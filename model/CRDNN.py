
import torch
import torch.nn as nn
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
from speechbrain.processing.features import STFT
from speechbrain.processing.features import spectral_magnitude
from speechbrain.processing.features import Filterbank

# compute_STFT = STFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400)
# compute_fbanks = Filterbank(n_mels=40)

# signal_STFT = compute_STFT(signal)
# mag = spectral_magnitude(signal_STFT)
# fbanks = compute_fbanks(mag)


