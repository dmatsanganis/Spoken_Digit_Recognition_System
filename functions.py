import os
from matplotlib import pyplot as plt
import numpy as np
import librosa
from hmmlearn import hmm
from librosa.feature import mfcc


def find_se(signal, frame_length, hop_length):
    energy = librosa.feature.rms(
        signal, frame_length=frame_length, hop_length=hop_length)[0]
    return np.array(energy)


def zero_crossing_rate(signal, frame_size, hop_size):
    zcr = librosa.feature.zero_crossing_rate(
        signal, frame_length=frame_size, hop_length=hop_size)[0]
    return np.array(zcr)


def bf_classifier(se, zcr):
    classes = []
    se_thres = np.mean(se)
    zcr_thres = np.mean(zcr)
    for i in range(se.size):
        if zcr[i] <= zcr_thres and se[i] >= se_thres:
            classes.append(1)
        else:
            classes.append(0)
    return classes


def fir_band_pass(samples, fs, fL, fH, NL, NH, outputType):

    fH = fH / fs
    fL = fL / fs

    # Compute a low-pass filter with cutoff frequency fH.
    hlpf = np.sinc(2 * fH * (np.arange(NH) - (NH - 1) / 2.))
    hlpf *= np.blackman(NH)
    hlpf /= np.sum(hlpf)
    # Compute a high-pass filter with cutoff frequency fL.
    hhpf = np.sinc(2 * fL * (np.arange(NL) - (NL - 1) / 2.))
    hhpf *= np.blackman(NL)
    hhpf /= np.sum(hhpf)
    hhpf = -hhpf
    hhpf[int((NL - 1) / 2)] += 1
    # Convolve both filters.
    h = np.convolve(hlpf, hhpf)
    # Applying the filter to a signal s can be as simple as writing
    s = np.convolve(samples, h).astype(outputType)

    return s


def plots(signal, sr, frame_length, hop_length, se, zcr):
    # Figure 1 (Waveplot, RMSE, ZCR)
    fig, ax = plt.subplots(nrows=3, sharex=True,
                           sharey=True, constrained_layout=True)

    librosa.display.waveplot(signal, sr=sr, ax=ax[0])
    ax[0].set(title='Waveplot')
    ax[0].label_outer()

    frames = range(len(se))
    t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    librosa.display.waveplot(signal, sr=sr, alpha=0.5, ax=ax[1])
    ax[1].plot(t, se, color="r")
    ax[1].set_ylim((-1, 1))
    ax[1].set(title='RMSE')
    ax[1].label_outer()

    librosa.display.waveplot(signal, sr=sr, alpha=0.5, ax=ax[2])
    ax[2].plot(t, zcr, color="r")
    ax[2].set_ylim((-1, 1))
    ax[2].set(title="ZCR")
    ax[2].label_outer()

    plt.tight_layout()
    plt.show()

    # Figure 2 (Spectogram, Mel-Spectogram, MFCC)
    fig, ax = plt.subplots(nrows=3, sharex=False,
                           sharey=False, constrained_layout=True)

    y_to_db = librosa.amplitude_to_db(abs(librosa.stft(signal)))
    librosa.display.specshow(
        y_to_db, sr=sr, x_axis='time', y_axis='hz', ax=ax[0])
    ax[0].set(title='Spectrograph')
    ax[0].label_outer()

    mel_spectogram = librosa.feature.melspectrogram(
        signal, sr=sr, n_fft=frame_length, hop_length=hop_length)
    log_mel_spectogram = librosa.power_to_db(mel_spectogram)
    librosa.display.specshow(
        log_mel_spectogram, x_axis="time", y_axis="mel", sr=sr, ax=ax[1])
    ax[1].set(title='Mel-Spectrograph')

    mfccs = librosa.feature.mfcc(
        y=signal, sr=sr, n_fft=frame_length, hop_length=hop_length)
    librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax[2])
    ax[2].set(title='MFCC')

    plt.tight_layout()
    plt.show()
