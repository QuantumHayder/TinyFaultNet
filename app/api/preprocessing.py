"""
Audio preprocessing for inference.
Supports two feature extraction pipelines:

  1. LEGACY (195 features) — pre-augmentation model
     - librosa-based MFCC extraction
     - np.diff for deltas
     - 5 summary stats (mean, std, max, min, IQR) × 13 coeffs × 3 groups

  2. AUGMENTED (158 features) — post-augmentation model
     - torchaudio-based MFCC + Mel spectrogram
     - torchaudio compute_deltas
     - 2 summary stats (mean, std) × 13 MFCC × 3 groups + 2 × 40 mel bins
"""

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
import librosa


# ═══════════════════════════════════════════════════════════════════════════════
#  LEGACY PIPELINE (195 features) — matches MFCCExtractor from original notebook
# ═══════════════════════════════════════════════════════════════════════════════

def _summarize_legacy(m: np.ndarray) -> np.ndarray:
    """
    Compute 5 summary statistics along the time axis (axis=0).
    Input:  (T, n_mfcc)
    Output: (5 * n_mfcc,)
    """
    return np.concatenate([
        np.mean(m, axis=0),
        np.std(m, axis=0),
        np.max(m, axis=0),
        np.min(m, axis=0),
        np.percentile(m, 75, axis=0) - np.percentile(m, 25, axis=0),  # IQR
    ])


def preprocess_audio_legacy(
    audio_path: str,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_mels: int = 40,
    n_fft: int = 512,
    duration_sec: float = 10.0,
) -> np.ndarray:
    """
    Legacy feature extraction pipeline.
    Replicates the original MFCCExtractor using librosa.
    Returns: np.ndarray of shape (195,)
    """
    hop_length = n_fft // 2

    # Load audio
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    # Pad or cut to fixed duration
    target_len = int(sr * duration_sec)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    elif len(y) > target_len:
        y = y[:target_len]

    # Extract MFCCs using librosa
    # librosa.feature.mfcc internally does: pre-emphasis, FFT, mel filterbank, DCT
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    # mfcc shape: (n_mfcc, T) → transpose to (T, n_mfcc) for np.diff
    mfcc = mfcc.T

    # Deltas using np.diff (matches original notebook, NOT librosa.feature.delta)
    delta = np.diff(mfcc, axis=0)           # (T-1, n_mfcc)
    delta_delta = np.diff(delta, axis=0)    # (T-2, n_mfcc)

    # Summarize: 5 stats × 13 coeffs × 3 groups = 195
    features = np.concatenate([
        _summarize_legacy(mfcc),
        _summarize_legacy(delta),
        _summarize_legacy(delta_delta),
    ])

    return features  # (195,)


# ═══════════════════════════════════════════════════════════════════════════════
#  AUGMENTED PIPELINE (158 features) — new model with mel + MFCC
# ═══════════════════════════════════════════════════════════════════════════════

def _create_resampler(orig_sr: int, new_sr: int):
    return torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=new_sr)


def _rms_normalize(audio: torch.Tensor) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(audio * audio))
    return audio / (rms + 1e-8)


def _cut(audio: torch.Tensor, sr: int, lower_cut=0.5, upper_cut=9.5) -> torch.Tensor:
    start = int(lower_cut * sr)
    end = int(upper_cut * sr)
    return audio[start:end]


def _extract_mel_spectrogram(audio: torch.Tensor, sr: int,
                              n_fft: int, hop_length: int, n_mels: int):
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, power=2.0,
    )
    mel = mel_transform(audio)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
    return mel_db.squeeze(0)


def _extract_mfcc(audio: torch.Tensor, sr: int, n_mfcc: int,
                   n_fft: int, hop_length: int, n_mels: int):
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr, n_mfcc=n_mfcc,
        melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels},
    )
    return mfcc_transform(audio).squeeze(0)


def _mfcc_stats_with_deltas(mfcc: torch.Tensor) -> torch.Tensor:
    delta = AF.compute_deltas(mfcc)
    delta2 = AF.compute_deltas(delta)
    return torch.cat([
        mfcc.mean(dim=1),   mfcc.std(dim=1),
        delta.mean(dim=1),  delta.std(dim=1),
        delta2.mean(dim=1), delta2.std(dim=1),
    ])


def preprocess_audio_augmented(
    audio_path: str,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_mels: int = 40,
    n_fft: int = 1024,
    duration_sec: float = 9.0,
) -> np.ndarray:
    """
    Augmented model feature extraction pipeline.
    Mirrors MachineAudioDatasetAugmented with train=False.
    Returns: np.ndarray of shape (158,)
    """
    hop_length = n_fft // 2

    # Load
    audio, file_sr = torchaudio.load(audio_path)
    audio = audio.mean(dim=0)

    # Resample
    if file_sr != sr:
        resampler = _create_resampler(file_sr, sr)
        audio = resampler(audio)

    # RMS normalize
    audio = _rms_normalize(audio)

    # Deterministic crop
    target_len = int(sr * duration_sec)
    audio_len = audio.shape[-1]
    if audio_len < target_len:
        audio = torch.nn.functional.pad(audio, (0, target_len - audio_len))
    elif audio_len > target_len:
        audio = _cut(audio, sr, lower_cut=0.5, upper_cut=0.5 + duration_sec)
        if audio.shape[-1] < target_len:
            audio = torch.nn.functional.pad(audio, (0, target_len - audio.shape[-1]))

    # Extract features
    mel = _extract_mel_spectrogram(audio, sr, n_fft, hop_length, n_mels)
    mfcc = _extract_mfcc(audio, sr, n_mfcc, n_fft, hop_length, n_mels)

    # Collapse to 1D
    mfcc_feat = _mfcc_stats_with_deltas(mfcc)
    mel_mean = mel.mean(dim=1)
    mel_std = mel.std(dim=1)
    mel_feat = torch.cat([mel_mean, mel_std])

    features = torch.cat([mfcc_feat, mel_feat]).numpy()
    return features


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTER — picks the right pipeline based on model config
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_audio(
    audio_path: str,
    expected_features: int,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_mels: int = 40,
    n_fft: int = 512,
) -> np.ndarray:
    """
    Route to the correct pipeline based on expected feature count.
    This auto-detects which model is loaded and uses the matching extraction.
    """
    if expected_features == 195:
        return preprocess_audio_legacy(
            audio_path, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft,
        )
    else:
        return preprocess_audio_augmented(
            audio_path, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft,
        )