import torchaudio
import torch
import torchaudio.functional as F

def extract_mfcc_torch(audio: torch.Tensor, sr, n_mfcc=13):
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": 1024,
            "hop_length": 512,
            "n_mels": 40
        }
    )

    return mfcc_transform(audio)


def mfcc_stats_torch(mfcc):

    mean = mfcc.mean(dim=1)
    std = mfcc.std(dim=1)

    return torch.cat([mean, std])


def mfcc_stats_with_deltas_torch(mfcc):

    delta = F.compute_deltas(mfcc)
    delta2 = F.compute_deltas(delta)

    return torch.cat([
        mfcc.mean(dim=1), mfcc.std(dim=1),
        delta.mean(dim=1), delta.std(dim=1),
        delta2.mean(dim=1), delta2.std(dim=1)
    ])


def extract_spectrogram_torch(audio: torch.Tensor, n_fft=1024, hop_length=512):
    # audio shape must be [1, T]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0
    )

    S = transform(audio)  # [1, freq_bins, time]

    S_db = torchaudio.transforms.AmplitudeToDB()(S)

    return S_db.squeeze(0)  # [freq_bins, time]    


def extract_mel_spectrogram_torch(audio: torch.Tensor, sr: int,
                                   n_fft=1024, hop_length=512, n_mels=40):

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )

    mel = mel_transform(audio)

    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

    return mel_db.squeeze(0)  # [n_mels, time]