import numpy as np
import torch
import torchaudio
import librosa

def load_audio(path):
    audio, sr = librosa.load(path)
    return torch.tensor(audio, dtype=torch.float32), sr


def create_resampler(orig_sr: int, new_sr: int):
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=new_sr)
    return resampler

def resample(audio: torch.Tensor, resampler) -> torch.Tensor:
    audio = resampler(audio)
    return audio


def peak_normalize(audio: torch.Tensor) -> torch.Tensor: 
    max_val = torch.max(torch.abs(audio))

    if max_val == 0:
        return audio
    
    return audio / max_val


def rms_normalize(audio: torch.Tensor) -> torch.Tensor: 
    rms = torch.sqrt(torch.mean(audio * audio))
    return audio / (rms + 1e-8)


def cut(audio: torch.Tensor, sr: int, lower_cut=0.5, upper_cut=9.5) -> torch.Tensor:
    start = int(lower_cut * sr)
    end = int(upper_cut * sr)
    audio = audio[start:end]
    return audio






