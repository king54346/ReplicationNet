"""
model.py — DX-Ball DQN 环境 + 网络 + 缓冲区

游戏地址：https://dx-ball.ru/
依赖：pip install selenium torch numpy pillow pyautogui

控制方式：pyautogui 真实 OS 级鼠标（最可靠，游戏无法屏蔽）
状态：4 帧灰度 Canvas 截图叠加，每帧 84×84
动作（离散 3）：
    0 = 不动
    1 = 鼠标向左移动 MOVE_PX 像素
    2 = 鼠标向右移动 MOVE_PX 像素
奖励：
    每帧存活          +0.1
    分数增加 Δ        +(Δ × 0.5)
    丢失一条命        -5.0
    游戏结束          -10.0
    左/右移动惩罚     -0.01
"""
import base64
import io
import random
import time
from collections import deque

import numpy as np
import pyautogui
import torch
import torch.nn as nn
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# ── 关闭 pyautogui 安全限制（边角暂停） ─────────────────
pyautogui.FAILSAFE  = False
pyautogui.PAUSE     = 0.0

# ── 帧参数 ────────────────────────────────────────────────
FRAME_H = 84
FRAME_W = 84
STACK_N = 4

# ── 每步鼠标移动像素 ─────────────────────────────────────
MOVE_PX = 40

# ── 动作惩罚 ─────────────────────────────────────────────
ACTION_PENALTY = {0: 0.0, 1: -0.01, 2: -0.01}

# ── JS：Canvas 截图 ───────────────────────────────────────
_JS_CANVAS = """
try {
    var c = document.querySelector('canvas');
    if (!c) return null;
    return c.toDataURL('image/png');
} catch(e) { return null; }
"""

# ── JS：读游戏内部状态 ────────────────────────────────────
_JS_STATE = """
try {
    var g = window.game || window.Game || window.gameState || {};
    var score = g.score ?? g.points ?? window.score ?? 0;
    var lives = g.lives ?? g.lifes ?? window.lives ?? 3;
    var over  = !!(g.gameOver ?? g.isOver ?? g.over ?? window.gameOver ?? false);
    return {score: +score, lives: +lives, gameOver: over};
} catch(e) { return {score:0, lives:3, gameOver:false}; }
"""

# ── JS：Canvas 在页面中的坐标 ────────────────────────────
_JS_CANVAS_RECT = """
try {
    var c = document.querySelector('canvas');
    if (!c) return null;
    var r = c.getBoundingClientRect();
    return {left: r.left, top: r.top, width: r.width, height: r.height};
} catch(e) { return null; }
"""


class DXBallEnv:
    """pyautogui 鼠标驱动的 DX-Ball 游戏环境。"""

    STATE_SHAPE = (STACK_N, FRAME_H, FRAME_W)
    ACTION_DIM  = 3

    def __init__(self, headless: bool = False, step_delay: float = 0.05):
        if headless:
            raise ValueError("pyautogui 需要显示器，不支持 headless 模式。")
        self.step_delay = step_delay

        opts = Options()
        opts.add_argument("--mute-audio")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1100,800")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")

        self.driver = webdriver.Chrome(options=opts)
        # 把浏览器窗口放在屏幕左上，方便坐标计算
        self.driver.set_window_position(0, 0)
        self.driver.get("https://dx-ball.ru/")
        time.sleep(3.5)

        # 计算 Canvas 在屏幕上的绝对坐标
        self._update_canvas_coords()

        self.score       = 0
        self.lives       = 3
        self._idle_steps = 0
        self._frames: deque = deque(maxlen=STACK_N)

        self._start_game()
        self._fill_frames()

    # ─── 公开接口 ─────────────────────────────────────────
    def reset(self) -> np.ndarray:
        self.driver.refresh()
        time.sleep(3.5)
        self._update_canvas_coords()
        self.score       = 0
        self.lives       = 3
        self._idle_steps = 0
        self._frames.clear()
        self._start_game()
        self._fill_frames()
        return self._stack()

    def step(self, action: int):
        self._send_action(action)
        time.sleep(self.step_delay)

        frame     = self._grab_frame()
        gs        = self._game_state()
        new_score = gs["score"]
        new_lives = gs["lives"]
        done      = gs["gameOver"]

        # ── 奖励计算 ──────────────────────────────────
        reward = 0.1 + ACTION_PENALTY[action]

        delta = max(0, new_score - self.score)
        if delta > 0:
            reward += delta * 0.5
            self._idle_steps = 0
        else:
            self._idle_steps += 1

        if new_lives < self.lives:
            reward -= 5.0 * (self.lives - new_lives)
            self._idle_steps = 0   # 丢球后等待发射，重置计数

        if done:
            reward = -10.0

        # ── 自动处理卡屏 ───────────────────────────────
        # 连续静止：可能是「等待发球」或「High Score 输入框」
        if self._idle_steps >= 40:
            self._dismiss_blocking_screen()
            self._idle_steps = 0

        self.score = new_score
        self.lives = new_lives
        self._frames.append(frame)
        return self._stack(), reward, done, {"score": new_score, "lives": new_lives}

    def close(self):
        self.driver.quit()

    # ─── 内部 ─────────────────────────────────────────────
    def _update_canvas_coords(self):
        """计算 Canvas 在屏幕上的绝对坐标（考虑浏览器 chrome 高度）。"""
        rect = None
        for _ in range(10):
            rect = self.driver.execute_script(_JS_CANVAS_RECT)
            if rect and rect["width"] > 0:
                break
            time.sleep(0.5)
        if not rect:
            rect = {"left": 215, "top": 240, "width": 645, "height": 480}

        win    = self.driver.get_window_position()
        # 浏览器 chrome 高度（标题栏 + 地址栏 + 通知栏）
        chrome_h = self.driver.execute_script(
            "return window.outerHeight - window.innerHeight"
        )

        self._scr_left  = int(win["x"] + rect["left"])
        self._scr_top   = int(win["y"] + chrome_h + rect["top"])
        self._scr_right = int(self._scr_left + rect["width"])
        self._scr_bot   = int(self._scr_top  + rect["height"])

        # 挡板在 canvas 底部附近
        self._cx = (self._scr_left + self._scr_right) // 2   # 初始 X：canvas 中央
        self._cy = self._scr_bot - 30                         # Y：canvas 底部偏上一点
        self._mouse_x = self._cx

    def _start_game(self):
        """
        点击 canvas 中心跳过开始界面 / High Score 界面，进入游戏。
        逐步点击+Enter，确保无论处于哪个卡屏状态都能通过。
        """
        cx = (self._scr_left + self._scr_right) // 2
        cy = (self._scr_top  + self._scr_bot)  // 2

        for _ in range(5):
            pyautogui.click(cx, cy)
            time.sleep(0.3)
            pyautogui.typewrite("RL", interval=0.08)  # 防止 High Score 界面卡住
            time.sleep(0.1)
            pyautogui.press("enter")
            time.sleep(0.3)

        # 把鼠标移到挡板初始位置（canvas 底部中央）
        pyautogui.moveTo(self._mouse_x, self._cy)
        time.sleep(1.5)

    def _dismiss_blocking_screen(self):
        """
        处理所有可能的卡屏状态：
        - 等待发球        → 点击 canvas
        - High Score 输入 → 点击聚焦 → 输入名字 "RL" → Enter → 再点击
        - 排行榜/过渡页   → 点击 + Enter 关闭
        """
        cx = (self._scr_left + self._scr_right) // 2
        cy = (self._scr_top  + self._scr_bot)  // 2

        # 1. 先点击 canvas 确保它有键盘焦点
        pyautogui.click(cx, cy)
        time.sleep(0.4)

        # 2. 输入名字（High Score 界面需要至少一个字符才接受 Enter）
        pyautogui.typewrite("RL", interval=0.1)
        time.sleep(0.2)

        # 3. 按 Enter 提交名字 / 确认任何界面
        pyautogui.press("enter")
        time.sleep(0.5)

        # 4. 可能还有排行榜展示页，再点击+Enter 一次
        pyautogui.click(cx, cy)
        time.sleep(0.3)
        pyautogui.press("enter")
        time.sleep(0.5)

        # 5. 最后点一次发球
        pyautogui.click(cx, cy)
        time.sleep(0.5)

    def _send_action(self, action: int):
        """移动鼠标，挡板跟随。"""
        if action == 1:
            self._mouse_x = max(self._scr_left,  self._mouse_x - MOVE_PX)
        elif action == 2:
            self._mouse_x = min(self._scr_right, self._mouse_x + MOVE_PX)
        pyautogui.moveTo(self._mouse_x, self._cy, _pause=False)

    def _game_state(self) -> dict:
        try:
            s = self.driver.execute_script(_JS_STATE)
            if s:
                return s
        except Exception:
            pass
        return {"score": self.score, "lives": self.lives, "gameOver": False}

    def _grab_frame(self) -> np.ndarray:
        try:
            data_url = self.driver.execute_script(_JS_CANVAS)
            if data_url and data_url.startswith("data:image"):
                b64 = data_url.split(",", 1)[1]
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
                img = img.convert("L").resize((FRAME_W, FRAME_H), Image.BILINEAR)
                return np.array(img, dtype=np.float32) / 255.0
        except Exception:
            pass
        return np.zeros((FRAME_H, FRAME_W), dtype=np.float32)

    def _fill_frames(self):
        f = self._grab_frame()
        for _ in range(STACK_N):
            self._frames.append(f)

    def _stack(self) -> np.ndarray:
        return np.stack(list(self._frames), axis=0)


# ──────────────────────────────────────────────
# Dueling DQN（CNN）
# ──────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, in_channels: int, action_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        flat = 64 * 7 * 7
        self.value = nn.Sequential(
            nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x).flatten(1)
        v = self.value(feat)
        a = self.advantage(feat)
        return v + a - a.mean(dim=1, keepdim=True)


# ──────────────────────────────────────────────
# 经验回放（uint8 省内存）
# ──────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 30_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            (state  * 255).astype(np.uint8),
            int(action),
            float(reward),
            (next_state * 255).astype(np.uint8),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32) / 255.0,
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32) / 255.0,
            np.array(d,  dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
