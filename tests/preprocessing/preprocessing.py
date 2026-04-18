import pandas as pd
import matplotlib.pylab as plt
import librosa

from src.preprocessing.preprocessing import (
    load_audio, create_resampler,
    resample, peak_normalize,
    rms_normalize, cut
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

path = BASE_DIR / "data/raw/Machine 1/machine-data/Normal/2.wav"


audio, sr = load_audio(path)


plt.figure().set_figwidth(12)
librosa.display.waveshow(audio.numpy(), sr=sr)
plt.show()

resampler = create_resampler(sr, 16000)

audio = resample(audio, resampler)


plt.figure().set_figwidth(12)
librosa.display.waveshow(audio.numpy(), sr=16000)
plt.show()

audio = peak_normalize(audio)

audio = cut(audio, 16000)

plt.figure().set_figwidth(12)
librosa.display.waveshow(audio.numpy(), sr=16000)
plt.show()


