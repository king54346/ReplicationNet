"""
LRS2 Preprocessing for CTC
"""
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import torch
import torchaudio
import ffmpy
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


@dataclass
class PreprocessConfig:
    """预处理配置"""
    lrs2_root: str = './lrs2/main'
    dataset_dir: str = './dataset'
    metadata_dir: str = './lrs2'
    tokenizer_file: str = 'tokenizer_ctc.json'
    vocab_size: int = 500
    special_tokens: List[str] = None
    num_mel_bins: int = 80
    sample_rate: int = 16000
    audio_codec: str = 'pcm_s16le'
    audio_channels: int = 1
    num_workers: int = None
    max_workers: int = None
    
    def __post_init__(self):
        # CTC: blank必须是第一个（ID=0）
        if self.special_tokens is None:
            self.special_tokens = ['[BLANK]', '[PAD]', '[UNK]', '[BOS]', '[EOS]']
        
        if self.num_workers is None:
            cpu_count = multiprocessing.cpu_count()
            self.num_workers = cpu_count
            if self.max_workers is not None:
                self.num_workers = min(self.num_workers, self.max_workers)
        
        Path(self.dataset_dir).mkdir(parents=True, exist_ok=True)


_process_tokenizer = None
_process_config = None


def init_worker(config: PreprocessConfig):
    """进程初始化"""
    global _process_tokenizer, _process_config
    _process_tokenizer = Tokenizer.from_file(config.tokenizer_file)
    _process_config = config
    logging.basicConfig(level=logging.WARNING, 
                       format=f'[Process-{os.getpid()}] %(levelname)s: %(message)s')


def load_wav(wav_filename: str) -> Tuple[torch.Tensor, int]:
    """加载WAV"""
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


def convert_mp4_to_wav(mp4_filename: str, wav_filename: str, config: PreprocessConfig) -> bool:
    """MP4→WAV"""
    try:
        output_options = ['-acodec', config.audio_codec, '-ac', str(config.audio_channels),
                         '-ar', str(config.sample_rate), '-y']
        ff = ffmpy.FFmpeg(inputs={mp4_filename: None},
                         outputs={wav_filename: ' '.join(output_options)})
        ff.run()
        return True
    except Exception as e:
        logging.error(f"FFmpeg failed: {e}")
        return False


def extract_mel_features(waveform: torch.Tensor, config: PreprocessConfig) -> torch.Tensor:
    """提取Mel频谱"""
    waveform_int16 = waveform * 32768
    return torchaudio.compliance.kaldi.fbank(
        waveform_int16, num_mel_bins=config.num_mel_bins,
        sample_frequency=config.sample_rate)


def load_record_txt(metaname: str, config: PreprocessConfig) -> str:
    """加载文本"""
    txt_filename = os.path.join(config.lrs2_root, f'{metaname}.txt')
    if not os.path.exists(txt_filename):
        raise FileNotFoundError(f"Text not found: {txt_filename}")
    with open(txt_filename, 'r', encoding='utf-8') as fp:
        line = fp.readline().strip()
        if ':' in line:
            return line.split(':', 1)[1].strip()
        else:
            raise ValueError(f"Invalid format: {txt_filename}")


def process_single_sample_worker(metaname: str) -> Tuple[bool, Optional[str]]:
    """处理单个样本"""
    global _process_tokenizer, _process_config
    try:
        tokenizer = _process_tokenizer
        config = _process_config
        
        mp4_filename = os.path.join(config.lrs2_root, f'{metaname}.mp4')
        wav_filename = mp4_filename.replace('.mp4', '.wav')
        sample_file = os.path.join(config.dataset_dir, f'{metaname}.pt')
        
        if os.path.exists(sample_file):
            return (False, None)
        
        # 编码文本
        text = load_record_txt(metaname, config)
        encoded = tokenizer.encode(text)
        bos_id = tokenizer.token_to_id('[BOS]')
        eos_id = tokenizer.token_to_id('[EOS]')
        tokens = [bos_id] + encoded.ids + [eos_id]
        
        # 转换音频
        if not os.path.exists(wav_filename):
            if not convert_mp4_to_wav(mp4_filename, wav_filename, config):
                return (False, "FFmpeg failed")
        
        # 加载并提取特征
        waveform, sample_rate = load_wav(wav_filename)
        audio_features = extract_mel_features(waveform, config)
        
        # 保存
        sample = {
            'audio_features': audio_features,
            'sample_rate': sample_rate,
            'tokens': tokens,
            'text': text,
            'audio_length': audio_features.size(0),
            'token_length': len(tokens)
        }
        os.makedirs(os.path.dirname(sample_file), exist_ok=True)
        torch.save(sample, sample_file)
        return (True, None)
    except Exception as e:
        return (False, str(e))


def process_data_multiprocess(all_metas: Set[str], config: PreprocessConfig) -> Tuple[int, List[str]]:
    """多进程处理"""
    print(f"Processing {len(all_metas)} samples ({config.num_workers} workers)...")
    processed_count = 0
    failed_samples = []
    meta_list = list(all_metas)
    
    with ProcessPoolExecutor(max_workers=config.num_workers, initializer=init_worker,
                            initargs=(config,)) as executor:
        future_to_meta = {executor.submit(process_single_sample_worker, meta): meta
                         for meta in meta_list}
        with tqdm(total=len(meta_list), desc="Processing") as pbar:
            for future in as_completed(future_to_meta):
                meta = future_to_meta[future]
                try:
                    success, error_msg = future.result()
                    if success:
                        processed_count += 1
                    elif error_msg is not None:
                        failed_samples.append(f"{meta}: {error_msg}")
                except Exception as e:
                    failed_samples.append(f"{meta}: {str(e)}")
                finally:
                    pbar.update(1)
    
    print(f"✓ Processed {processed_count} new samples")
    if failed_samples:
        print(f"✗ Failed {len(failed_samples)} samples")
        with open('failed_samples.txt', 'w') as f:
            f.write('\n'.join(failed_samples))
    return processed_count, failed_samples


def load_metadata(filename: str) -> List[str]:
    """加载metadata"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Metadata not found: {filename}")
    records = []
    with open(filename, 'r') as fp:
        for line in fp:
            record = line.strip().split()[0]
            if record:
                records.append(record)
    print(f"  {len(records)} from {os.path.basename(filename)}")
    return records


def train_tokenizer(all_metas: Set[str], config: PreprocessConfig) -> Tokenizer:
    """训练CTC tokenizer"""
    print("Training tokenizer...")
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(vocab_size=config.vocab_size, 
                        special_tokens=config.special_tokens)
    
    def iter_all_txt():
        for metaname in tqdm(all_metas, desc="  Loading texts"):
            try:
                yield load_record_txt(metaname, config)
            except Exception as e:
                logging.warning(f"Failed {metaname}: {e}")
                continue
    
    tokenizer.train_from_iterator(iter_all_txt(), trainer=trainer, length=len(all_metas))
    tokenizer.save(config.tokenizer_file, pretty=True)
    
    blank_id = tokenizer.token_to_id('[BLANK]')
    print(f"✓ Tokenizer saved: {config.tokenizer_file}")
    print(f"  Vocab: {tokenizer.get_vocab_size()}, BLANK ID: {blank_id}")
    if blank_id != 0:
        print("  ⚠️ WARNING: BLANK is not ID 0!")
    return tokenizer


def main():
    """主流程"""
    import time
    config = PreprocessConfig(num_workers=None, max_workers=8)
    
    print("=" * 70)
    print("CTC Preprocessing")
    print("=" * 70)
    
    try:
        print("\n[1/3] Loading metadata...")
        train_metas = load_metadata(os.path.join(config.metadata_dir, 'train.txt'))
        val_metas = load_metadata(os.path.join(config.metadata_dir, 'val.txt'))
        test_metas = load_metadata(os.path.join(config.metadata_dir, 'test.txt'))
        all_metas = set(train_metas + val_metas + test_metas)
        print(f"  Total: Train={len(train_metas)}, Val={len(val_metas)}, Test={len(test_metas)}")
        
        print(f"\n[2/3] Tokenizer...")
        if not os.path.exists(config.tokenizer_file):
            train_tokenizer(all_metas, config)
        else:
            tokenizer = Tokenizer.from_file(config.tokenizer_file)
            print(f"  Loaded: {config.tokenizer_file}")
            print(f"  Vocab: {tokenizer.get_vocab_size()}, BLANK: {tokenizer.token_to_id('[BLANK]')}")
        
        print(f"\n[3/3] Processing ({config.num_workers} workers)...")
        start = time.time()
        processed, failed = process_data_multiprocess(all_metas, config)
        elapsed = time.time() - start
        
        print("\n" + "=" * 70)
        print(f"Done! Processed {processed}/{len(all_metas)} ({elapsed:.1f}s)")
        print(f"Speed: {processed/elapsed:.2f} samples/sec" if processed > 0 else "")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        raise


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()