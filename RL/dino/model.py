"""
model.py — Chrome Dino 真实游戏环境 (Selenium) + DQN + ReplayBuffer

依赖：
    pip install selenium torch numpy
    需要 Chrome 浏览器 + 对应版本 ChromeDriver（或 selenium-manager 自动下载）

游戏状态（10 维）：
    [dino_y_norm, dino_vel_norm, dino_jumping, dino_ducking,
     obs_dist_norm, obs_y_norm, obs_w_norm, obs_is_bird,
     speed_norm, second_obs_dist_norm]

动作（离散 3）：
    0 = 什么都不做
    1 = 跳跃 (↑)
    2 = 蹲下 (↓)

奖励设计：
    每步存活  +0.1
    跳跃动作  -0.15   ← 惩罚多余动作，迫使智能体只在必要时跳
    蹲下动作  -0.08   ← 蹲下惩罚稍低（有时需要蹲）
    碰撞死亡  -10.0
"""
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# ── 游戏内坐标常量（Chrome Dino 默认画布）────────────────
CANVAS_W      = 600          # 画布宽
CANVAS_H      = 150          # 画布高
DINO_GROUND_Y = 93.0         # 恐龙站立时 yPos（像素，从画布顶）
MAX_SPEED     = 13.0
MIN_SPEED     = 6.0

# ── 动作惩罚 ──────────────────────────────────────────────
ACTION_PENALTY = {0: 0.01, 1: -0.3, 2: -0.3}

# ── 提取游戏状态的 JS ─────────────────────────────────────
_JS_GET_STATE = """
try {
    var r = Runner.instance_;
    if (!r) return null;

    var obstacles = r.horizon.obstacles;
    function obsInfo(o) {
        if (!o) return {x: 1.0, y: 0.0, w: 0.0, bird: 0};
        var isBird = (o.typeConfig && o.typeConfig.type === 'PTERODACTYL') ? 1 : 0;
        return {
            x: o.xPos  / 600.0,
            y: o.yPos  / 150.0,
            w: o.width / 600.0,
            bird: isBird
        };
    }

    var o1 = obsInfo(obstacles[0] || null);
    var o2_dist = obstacles[1] ? obstacles[1].xPos / 600.0 : 1.0;

    return {
        y:        r.tRex.yPos / 150.0,
        vel:      (r.tRex.jumpVelocity || 0) / 15.0,
        jumping:  r.tRex.jumping  ? 1.0 : 0.0,
        ducking:  r.tRex.ducking  ? 1.0 : 0.0,
        speed:    r.currentSpeed / 13.0,
        crashed:  r.crashed,
        playing:  r.playing,
        score:    Math.ceil(r.distanceMeter.getActualDistance(r.distanceRan)),
        o1x:      o1.x,   o1y: o1.y, o1w: o1.w, o1bird: o1.bird,
        o2x:      o2_dist
    };
} catch(e) { return null; }
"""

# ── 动作 JS ──────────────────────────────────────────────
_JS_JUMP  = "document.dispatchEvent(new KeyboardEvent('keydown',{keyCode:38,bubbles:true}));"
_JS_DUCK  = "document.dispatchEvent(new KeyboardEvent('keydown',{keyCode:40,bubbles:true}));"
_JS_UNDUCK = "document.dispatchEvent(new KeyboardEvent('keyup',  {keyCode:40,bubbles:true}));"
_JS_SPACE = "document.dispatchEvent(new KeyboardEvent('keydown',{keyCode:32,bubbles:true}));"
_JS_RESTART = """
(function(){
    var r = Runner.instance_;
    if (r && r.restart) { r.restart(); return; }
    // 部分版本无 restart()，改用空格重新触发
    document.dispatchEvent(new KeyboardEvent('keydown',{keyCode:32,bubbles:true}));
})();
"""


class ChromeDinoEnv:
    """直接驱动 chrome://dino/ 的强化学习环境。"""

    STATE_DIM  = 10
    ACTION_DIM = 3

    def __init__(self,
                 headless: bool = False,
                 step_delay: float = 0.05):
        """
        headless:   True = 无界面训练（更快）
        step_delay: 每步等待时间（秒）。太小会导致动作被游戏跳过。
        """
        self.step_delay = step_delay

        options = Options()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--mute-audio")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=900,400")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        self.driver = webdriver.Chrome(options=options)
        self.driver.get("https://web.wetab.link/games/dino/index.html")
        time.sleep(2.0)

        # 触发一次空格，让游戏进入 playing 状态
        self.driver.execute_script(_JS_SPACE)
        time.sleep(0.5)

        self.score = 0

    # ─────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        """重置游戏，返回初始状态。"""
        self.driver.execute_script(_JS_RESTART)
        time.sleep(0.4)
        self.score = 0
        return self._get_state()

    def step(self, action: int):
        """
        执行动作，返回 (state, reward, done, info)。
        reward 包含动作惩罚：不必要地跳/蹲会被扣分。
        """
        self._send_action(action)
        time.sleep(self.step_delay)

        raw = self._raw_state()
        if raw is None:
            # JS 还未初始化，返回零状态
            return np.zeros(self.STATE_DIM, dtype=np.float32), 0.0, False, {}

        done   = bool(raw["crashed"])
        score  = int(raw["score"])

        # ── 奖励计算 ───────────────────────────────────
        if done:
            reward = -10.0
        else:
            reward = 0.1 + ACTION_PENALTY[action]   # 存活奖励 - 动作惩罚

        self.score = score

        state = self._raw_to_state(raw)
        return state, reward, done, {"score": score}

    def close(self):
        self.driver.quit()

    # ─────────────────────────────────────────────────────
    def _send_action(self, action: int):
        if action == 1:
            self.driver.execute_script(_JS_JUMP)
        elif action == 2:
            self.driver.execute_script(_JS_DUCK)
        else:
            # 释放蹲下键，保持站立
            self.driver.execute_script(_JS_UNDUCK)

    def _raw_state(self):
        return self.driver.execute_script(_JS_GET_STATE)

    def _get_state(self) -> np.ndarray:
        for _ in range(10):
            raw = self._raw_state()
            if raw:
                return self._raw_to_state(raw)
            time.sleep(0.1)
        return np.zeros(self.STATE_DIM, dtype=np.float32)

    @staticmethod
    def _raw_to_state(raw: dict) -> np.ndarray:
        return np.array([
            raw["y"],           # dino y（归一化）
            raw["vel"],         # 跳跃速度
            raw["jumping"],     # 是否在跳
            raw["ducking"],     # 是否在蹲
            raw["o1x"],         # 最近障碍物 x 距离
            raw["o1y"],         # 最近障碍物 y
            raw["o1w"],         # 最近障碍物宽度
            raw["o1bird"],      # 是否是鸟（0/1）
            raw["speed"],       # 游戏速度
            raw["o2x"],         # 第二个障碍物距离
        ], dtype=np.float32)


# ──────────────────────────────────────────────
# DQN 网络（Dueling DQN）
# ──────────────────────────────────────────────
class DQN(nn.Module):
    """
    Dueling DQN：分离 Value 流和 Advantage 流。
    对 action 数量较少时收敛更快。
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        # Value 流：估计状态价值 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Advantage 流：估计每个动作的优势 A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        v    = self.value_stream(feat)
        a    = self.advantage_stream(feat)
        # Q(s,a) = V(s) + A(s,a) - mean(A)
        return v + a - a.mean(dim=1, keepdim=True)


# ──────────────────────────────────────────────
# 优先经验回放
# ──────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)