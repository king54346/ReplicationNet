
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial

import torch
import torchaudio
import ffmpy
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


# ============================================================================
# 配置类 Configuration Class
# ============================================================================

@dataclass
class PreprocessConfig:
    """预处理配置"""
    
    # 路径配置
    lrs2_root: str = './lrs2/main'
    dataset_dir: str = './dataset'
    metadata_dir: str = './lrs2'
    tokenizer_file: str = 'tokenizer.json'
    
    # 分词器配置
    vocab_size: int = 500
    special_tokens: List[str] = None
    
    # 音频配置
    num_mel_bins: int = 80
    sample_rate: int = 16000
    
    # FFmpeg配置
    audio_codec: str = 'pcm_s16le'
    audio_channels: int = 1
    
    # 多进程配置
    num_workers: int = None
    max_workers: int = None
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = ['[UNK]', '[PAD]', '[BOS]', '[EOS]']
        
        if self.num_workers is None:
            cpu_count = multiprocessing.cpu_count()
            self.num_workers = cpu_count
            if self.max_workers is not None:
                self.num_workers = min(self.num_workers, self.max_workers)
        
        Path(self.dataset_dir).mkdir(parents=True, exist_ok=True)


# ============================================================================
# 全局变量 (每个进程独立)
# Global Variables (Independent per process)
# ============================================================================

# 每个进程会独立加载tokenizer
_process_tokenizer = None
_process_config = None


def init_worker(config: PreprocessConfig):
    """
    进程初始化函数
    在每个进程启动时调用一次，加载tokenizer
    
    Args:
        config: 预处理配置
    """
    global _process_tokenizer, _process_config
    
    # 每个进程加载自己的tokenizer
    _process_tokenizer = Tokenizer.from_file(config.tokenizer_file)
    _process_config = config
    
    # 设置进程级日志
    logging.basicConfig(
        level=logging.WARNING,  # 多进程时只记录警告和错误
        format=f'[Process-{os.getpid()}] %(levelname)s: %(message)s'
    )


# ============================================================================
# 音频处理函数 Audio Processing Functions
# ============================================================================

def load_wav(wav_filename: str) -> Tuple[torch.Tensor, int]:
    """加载WAV音频文件"""
    try:
        waveform, sample_rate = torchaudio.load(wav_filename)
        return waveform, sample_rate
    except Exception:
        try:
            import soundfile as sf
            audio, sample_rate = sf.read(wav_filename, dtype='float32')
            if audio.ndim == 1:
                audio = audio[:, None]
            audio = torch.from_numpy(audio).transpose(0, 1)
            return audio, sample_rate
        except Exception as e:
            raise RuntimeError(f"Failed to load {wav_filename}") from e


def convert_mp4_to_wav(
    mp4_filename: str, 
    wav_filename: str,
    config: PreprocessConfig
) -> bool:
    """将MP4转换为WAV"""
    try:
        output_options = [
            '-acodec', config.audio_codec,
            '-ac', str(config.audio_channels),
            '-ar', str(config.sample_rate),
            '-y'
        ]
        
        ff = ffmpy.FFmpeg(
            inputs={mp4_filename: None},
            outputs={wav_filename: ' '.join(output_options)}
        )
        ff.run()
        return True
    except Exception as e:
        logging.error(f"FFmpeg failed for {mp4_filename}: {e}")
        return False


def extract_mel_features(
    waveform: torch.Tensor, 
    config: PreprocessConfig
) -> torch.Tensor:
    """提取Mel频谱特征"""
    waveform_int16 = waveform * 32768
    audio_features = torchaudio.compliance.kaldi.fbank(
        waveform_int16,
        num_mel_bins=config.num_mel_bins,
        sample_frequency=config.sample_rate
    )
    return audio_features


# ============================================================================
# 文本处理 Text Processing
# ============================================================================

def load_record_txt(metaname: str, config: PreprocessConfig) -> str:
    """加载文本标注"""
    txt_filename = os.path.join(config.lrs2_root, f'{metaname}.txt')
    
    if not os.path.exists(txt_filename):
        raise FileNotFoundError(f"Text file not found: {txt_filename}")
    
    with open(txt_filename, 'r', encoding='utf-8') as fp:
        line = fp.readline().strip()
        if ':' in line:
            return line.split(':', 1)[1].strip()
        else:
            raise ValueError(f"Invalid format in {txt_filename}")


# ============================================================================
# 单样本处理 (进程独立)
# Single Sample Processing (Process-independent)
# ============================================================================

def process_single_sample_worker(metaname: str) -> Tuple[bool, Optional[str]]:
    """
    处理单个样本 (在独立进程中运行)
    
    注意: 这个函数在子进程中执行，访问全局变量 _process_tokenizer 和 _process_config
    
    Args:
        metaname: 样本名称
    
    Returns:
        (success, error_message): 成功标志和错误信息
    """
    global _process_tokenizer, _process_config
    
    try:
        # 使用进程级的tokenizer和config
        tokenizer = _process_tokenizer
        config = _process_config
        
        # 构建文件路径
        txt_filename = os.path.join(config.lrs2_root, f'{metaname}.txt')
        mp4_filename = os.path.join(config.lrs2_root, f'{metaname}.mp4')
        wav_filename = mp4_filename.replace('.mp4', '.wav')
        sample_file = os.path.join(config.dataset_dir, f'{metaname}.pt')
        
        # 检查是否已处理
        if os.path.exists(sample_file):
            return (False, None)  # 已存在
        
        # 读取并编码文本
        text = load_record_txt(metaname, config)
        encoded = tokenizer.encode(text)
        bos_id = tokenizer.token_to_id('[BOS]')
        eos_id = tokenizer.token_to_id('[EOS]')
        tokens = [bos_id] + encoded.ids + [eos_id]
        
        # 转换音频
        if not os.path.exists(wav_filename):
            if not convert_mp4_to_wav(mp4_filename, wav_filename, config):
                return (False, "FFmpeg conversion failed")
        
        # 加载音频
        waveform, sample_rate = load_wav(wav_filename)
        
        # 提取特征
        audio_features = extract_mel_features(waveform, config)
        
        # 保存样本
        sample = {
            'audio_features': audio_features,
            'sample_rate': sample_rate,
            'tokens': tokens,
            'text': text
        }
        
        os.makedirs(os.path.dirname(sample_file), exist_ok=True)
        torch.save(sample, sample_file)
        
        return (True, None)
        
    except Exception as e:
        return (False, str(e))


# ============================================================================
# 多进程数据处理 Multi-process Data Processing
# ============================================================================

def process_data_multiprocess(
    all_metas: Set[str],
    config: PreprocessConfig
) -> Tuple[int, List[str]]:
    """
    多进程批量处理数据
    
    Args:
        all_metas: 所有样本名称集合
        config: 预处理配置
    
    Returns:
        (processed_count, failed_samples): 处理数量和失败样本列表
    """
    print(f"Processing {len(all_metas)} samples with {config.num_workers} processes...")
    
    processed_count = 0
    failed_samples = []
    meta_list = list(all_metas)
    
    # 使用进程池，每个进程会调用init_worker初始化
    with ProcessPoolExecutor(
        max_workers=config.num_workers,
        initializer=init_worker,
        initargs=(config,)
    ) as executor:
        
        # 提交所有任务
        future_to_meta = {
            executor.submit(process_single_sample_worker, meta): meta
            for meta in meta_list
        }
        
        # 显示进度
        with tqdm(total=len(meta_list), desc="Processing samples") as pbar:
            for future in as_completed(future_to_meta):
                meta = future_to_meta[future]
                try:
                    success, error_msg = future.result()
                    
                    if success:
                        processed_count += 1
                    elif error_msg is not None:
                        failed_samples.append(f"{meta}: {error_msg}")
                        logging.error(f"Failed {meta}: {error_msg}")
                    
                except Exception as e:
                    failed_samples.append(f"{meta}: {str(e)}")
                    logging.error(f"Exception for {meta}: {e}")
                
                finally:
                    pbar.update(1)
    
    print(f"Successfully processed {processed_count} new samples")
    
    if failed_samples:
        print(f"Failed to process {len(failed_samples)} samples")
        with open('failed_samples_mp.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(failed_samples))
        print("Failed samples saved to failed_samples_mp.txt")
    
    return processed_count, failed_samples


# ============================================================================
# 分词器训练 Tokenizer Training
# ============================================================================

def load_metadata(filename: str) -> List[str]:
    """加载元数据文件"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Metadata not found: {filename}")
    
    records = []
    with open(filename, 'r', encoding='utf-8') as fp:
        for line in fp:
            record = line.strip().split()[0]
            if record:
                records.append(record)
    
    print(f"Loaded {len(records)} records from {filename}")
    return records


def train_tokenizer(all_metas: Set[str], config: PreprocessConfig) -> Tokenizer:
    """训练BPE分词器"""
    print("Training tokenizer...")
    
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        special_tokens=config.special_tokens
    )
    
    def iter_all_txt():
        for metaname in tqdm(all_metas, desc="Loading texts"):
            try:
                yield load_record_txt(metaname, config)
            except Exception as e:
                logging.warning(f"Failed to load text for {metaname}: {e}")
                continue
    
    tokenizer.train_from_iterator(
        iter_all_txt(), 
        trainer=trainer, 
        length=len(all_metas)
    )
    
    tokenizer.save(config.tokenizer_file, pretty=True)
    print(f"Tokenizer saved to {config.tokenizer_file}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    return tokenizer


# ============================================================================
# 主函数 Main Function
# ============================================================================

def main():
    """主执行流程"""
    import time
    
    # 配置
    config = PreprocessConfig(
        num_workers=None,  # 自动检测
        max_workers=8      # 限制最大进程数
    )
    
    print("=" * 60)
    print("Multi-process Preprocessing Pipeline")
    print("=" * 60)
    
    try:
        # 步骤1: 加载元数据
        print("Step 1: Loading metadata...")
        train_metas = load_metadata(os.path.join(config.metadata_dir, 'train.txt'))
        val_metas = load_metadata(os.path.join(config.metadata_dir, 'val.txt'))
        test_metas = load_metadata(os.path.join(config.metadata_dir, 'test.txt'))
        
        all_metas = set(train_metas + val_metas + test_metas)
        
        print(
            f"Dataset statistics: "
            f"train={len(train_metas)}, "
            f"val={len(val_metas)}, "
            f"test={len(test_metas)}, "
            f"total_unique={len(all_metas)}"
        )
        
        # 步骤2: 训练/加载分词器
        print("Step 2: Setting up tokenizer...")
        if not os.path.exists(config.tokenizer_file):
            tokenizer = train_tokenizer(all_metas, config)
        else:
            print(f"Tokenizer loaded from {config.tokenizer_file}")
        
        # 步骤3: 多进程处理(mel特征提取,audio,token,)
        print(f"Step 3: Processing with {config.num_workers} processes...")
        start_time = time.time()
        
        processed_count, failed_samples = process_data_multiprocess(
            all_metas, config
        )
        
        elapsed_time = time.time() - start_time
        
        # 性能统计
        print("=" * 60)
        print("Performance Statistics")
        print("=" * 60)
        print(f"Total samples: {len(all_metas)}")
        print(f"New processed: {processed_count}")
        print(f"Skipped: {len(all_metas) - processed_count - len(failed_samples)}")
        print(f"Failed: {len(failed_samples)}")
        print(f"Workers (processes): {config.num_workers}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        if processed_count > 0:
            print(f"Speed: {processed_count / elapsed_time:.2f} samples/sec")
        print("=" * 60)
        
        print("Preprocessing completed!")
        
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        raise


if __name__ == '__main__':
    # 多进程必须使用这个保护
    multiprocessing.set_start_method('spawn', force=True)
    main()