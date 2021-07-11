from functions import bf_classifier, find_se, fir_band_pass, plots, zero_crossing_rate
import librosa
import librosa.display
import numpy as np
import os
from hmmlearn import hmm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def kati(bVSf):
    numbers = []
    start = 0
    end = FRAME_SIZE
    for i in range(1, len(bVSf)):
        if bVSf[i-1] == 1 and bVSf[i] == 1:
            end = i*HOP_SIZE+FRAME_SIZE
        elif bVSf[i-2] == 1 and bVSf[i-1] == 0 and bVSf[i] == 1 and i-2 >= 0:
            end = i*HOP_SIZE+FRAME_SIZE
        elif bVSf[i-1] == 0 and bVSf[i] == 1 and bVSf[i+1] == 1:
            start = (i-1)*HOP_SIZE
            end = start+FRAME_SIZE
        elif bVSf[i-1] == 1 and bVSf[i] == 0 and i+4 < len(bVSf) and (bVSf[i+1] == 1 or bVSf[i+2] == 1 or bVSf[i+3] == 1 or bVSf[i+4] == 1):
            end = i*HOP_SIZE+FRAME_SIZE
        elif bVSf[i-1] == 1 and bVSf[i] == 0 and i-1 >= 0:
            numbers.append(signal[start:end])
    #numbers = [x for x in numbers if not len(x)/SAMPLE_RATE < 0.100]
    return numbers


FRAME_SIZE = 256
HOP_SIZE = 256
SAMPLE_RATE = 8000
signal, sr = librosa.load(
    'test/correct/3_8_6_4_0.wav', sr=SAMPLE_RATE)
signal = fir_band_pass(signal, SAMPLE_RATE, 200, 4000, 100, 100, np.float32)

se = find_se(signal, FRAME_SIZE, HOP_SIZE)
zcr = zero_crossing_rate(signal, FRAME_SIZE, HOP_SIZE)
bVSf = bf_classifier(se, zcr)
numbers = kati(bVSf)

plots(signal, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE, se, zcr)

dir = 'training/panos+dimitris/'
train_dataset_x = []
train_dataset_y = []
for folder in os.listdir(dir):
    label = folder.split('_')[1]
    fileList = [f for f in os.listdir(
        dir+folder) if os.path.splitext(f)[1] == '.wav']
    for fileName in fileList:
        audio, sr = librosa.load(dir+folder+'/'+fileName, SAMPLE_RATE)
        audio = fir_band_pass(audio, SAMPLE_RATE, 200,
                              4000, 100, 100, np.float32)
        feature = librosa.feature.mfcc(
            audio, SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        feature = np.mean(feature, axis=1)
        train_dataset_x.append(feature)
        train_dataset_y.append(label)

rfc = RandomForestClassifier(n_estimators=150).fit(
    train_dataset_x, train_dataset_y)
# svm_model = SVC(kernel='rbf').fit(train_dataset_x, train_dataset_y)

features = []
for i, number in enumerate(numbers):
    output = number
    feature = librosa.feature.mfcc(
        output, SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    feature = np.mean(feature, axis=1)
    features.append(feature)

print('RESULT:')
print(rfc.predict(features))
