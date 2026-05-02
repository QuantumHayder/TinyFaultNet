# Machine Audio Classifier (SVM)

SVM (RBF kernel) trained on MFCC + Mel spectrogram features
for machine sound classification.

## Files
- `svm_model.pkl` — trained SVM model
- `scaler.pkl` — StandardScaler fitted on training data

## Features
- 78-dim MFCC stats (mean/std of static + delta + delta-delta)
- 80-dim Mel stats (mean/std of 40 mel bins)
- Total: 158-dimensional feature vector

## Preprocessing
- Sample rate: 16000 Hz
- Duration: 9.0 seconds
- RMS normalized
- n_fft=1024, hop_length=512, n_mels=40, n_mfcc=13