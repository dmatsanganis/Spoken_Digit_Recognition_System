import os
import numpy as np
import librosa
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# Preprocessing
def preprocessing(filename, sample_rate):
    print('[Process]: Preprocessing Started')
    # Read the input file with specific sampling rate
    signal, sr = librosa.load(filename, sr=sample_rate)
    # Apply the FIR Band Pass Filter
    signal = fir_band_pass(signal, sample_rate, 200,
                           4000, 100, 100, np.float32)
    print('[Process]: Preprocessing Completed')
    return signal


# FIR Band Pass Filter
def fir_band_pass(samples, fs, fL, fH, NL, NH, outputType):
    fH = fH / fs
    fL = fL / fs
    # Compute a low-pass filter with cutoff frequency fH.
    lpf = np.sinc(2 * fH * (np.arange(NH) - (NH - 1) / 2.))
    lpf *= np.blackman(NH)
    lpf /= np.sum(lpf)
    # Compute a high-pass filter with cutoff frequency fL.
    hpf = np.sinc(2 * fL * (np.arange(NL) - (NL - 1) / 2.))
    hpf *= np.blackman(NL)
    hpf /= np.sum(hpf)
    hpf = -hpf
    hpf[int((NL - 1) / 2)] += 1
    # Convolve both filters.
    h = np.convolve(lpf, hpf)
    # Applying the filter to a signal s can be as simple as writing
    s = np.convolve(samples, h).astype(outputType)
    return s


# Dataset creation
def build_dataset(dir, sample_rate, frame_length, hop_length):
    print('[Process]: Dataset Training Started')
    # List for the MFCC features of each .wav file in the dataset
    train_dataset_x = []
    # List with the labels of each .wav file in the dataset
    train_dataset_y = []
    # Read each folder inside the dataset folder
    for folder in os.listdir(dir):
        # Get the label of the digit from the folder name
        label = folder.split('_')[1]
        # Get all the .wav files inside the folder
        fileList = [f for f in os.listdir(
            dir+folder) if os.path.splitext(f)[1] == '.wav']
        # For each .wav file
        for fileName in fileList:
            # Read the file with specific sampling rate
            audio, sr = librosa.load(dir+folder+'/'+fileName, sample_rate)
            # Apply the FIR Band Pass Filter
            audio = fir_band_pass(audio, sample_rate, 200,
                                  4000, 100, 100, np.float32)
            # Extract the MFCC features
            feature = librosa.feature.mfcc(
                audio, sample_rate, n_fft=frame_length, hop_length=hop_length)
            # Find the mean values for these features
            feature = np.mean(feature, axis=1)
            # Add them to the list
            train_dataset_x.append(feature)
            # Add the label to the list
            train_dataset_y.append(label)
    # Create a Random Forest Classifier and insert the 2 lists for training
    rfc = RandomForestClassifier(n_estimators=150).fit(
        train_dataset_x, train_dataset_y)
    print('[Process]: Dataset Training Completed')
    return rfc


# Root Mean Square Energy
def rmse(signal, frame_length, hop_length):
    # Get the Root Mean Square Energy
    energy = librosa.feature.rms(
        signal, frame_length=frame_length, hop_length=hop_length)[0]
    print('[Process]: RMSE Calculated')
    return np.array(energy)


# Zero Crossing Rate
def zero_crossing_rate(signal, frame_length, hop_length):
    # Get the Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        signal, frame_length=frame_length, hop_length=hop_length)[0]
    print('[Process]: ZCR Calculated')
    return np.array(zcr)


# Background vs Foreground Classifier
def b_vs_f(signal, energy, zcr, frame_length, hop_length):
    print('[Process]: Background vs Foreground Classification Started')
    # List that seperates background from foreground
    bvsf = []
    # Thresholds for Energy and Zero Crossing Rate
    energy_threshold = np.mean(energy)
    zcr_threshold = np.mean(zcr)
    # For each frame add 1 to the list if Zero Crossing Rate is under it's threshold and Energy is over it's threshold
    # Otherwise add 0 to the list
    for i in range(energy.size):
        if zcr[i] <= zcr_threshold and energy[i] >= energy_threshold:
            bvsf.append(1)
        else:
            bvsf.append(0)

    # List for the recognised digits
    numbers = []
    # Starting sample
    start = 0
    # Ending sample
    end = frame_length
    # For every frame of the signal
    for i in range(1, len(bvsf)):
        # If the current and previous frame are voiced frames, update the ending sample
        if bvsf[i-1] == 1 and bvsf[i] == 1:
            end = i*hop_length+frame_length
        # Else if the current frame is voiced, the previous is unvoiced and the second previous is voiced, update the ending sample
        elif i-2 >= 0 and bvsf[i-2] == 1 and bvsf[i-1] == 0 and bvsf[i] == 1:
            end = i*hop_length+frame_length
        # Else if the current frame is voiced, the previous is unvoiced and the next is voiced, update the starting and ending sample
        elif bvsf[i-1] == 0 and bvsf[i] == 1 and bvsf[i+1] == 1:
            start = (i-1)*hop_length
            end = start+frame_length
        # Else if the current frame is unvoiced, the previous is unvoiced and one of the next 4 is voiced, update the ending sample
        elif bvsf[i-1] == 1 and bvsf[i] == 0 and i+4 < len(bvsf) and (bvsf[i+1] == 1 or bvsf[i+2] == 1 or bvsf[i+3] == 1 or bvsf[i+4] == 1):
            end = i*hop_length+frame_length
        # Else if the current frame is unvoiced and the previous is voiced, get a part of signal based on the calculated starting and ending samples
        elif bvsf[i-1] == 1 and bvsf[i] == 0:
            numbers.append(signal[start:end])
    print('[Process]: Background vs Foreground Classification Completed')
    return numbers


# Digit Recognition
def recognition(rfc, numbers, sample_rate, frame_length, hop_length):
    print('[Process]: Digits Recognition Started')
    # List for the MFCC features of each recognised digit
    features = []
    # For each digit of the recognised digits
    for number in numbers:
        # Extract the MFCC features
        feature = librosa.feature.mfcc(
            number, sample_rate, n_fft=frame_length, hop_length=hop_length)
        # Find the mean values for these features
        feature = np.mean(feature, axis=1)
        # Add them to the list
        features.append(feature)
    # Get the prediction from the Random Forest Classifier
    prediction = rfc.predict(features)
    print('[Process]: Digits Recognition Completed')
    print('[Result]: ' + str(prediction))
    return prediction


# Plot Graphs
def plots(signal, energy, zcr, sample_rate, frame_length, hop_length, ):
    # Figure 1 (Waveplot, RMSE, ZCR)
    fig, ax = plt.subplots(nrows=3, sharex=True,
                           sharey=True, constrained_layout=True)

    librosa.display.waveplot(signal, sr=sample_rate, ax=ax[0])
    ax[0].set(title='Waveplot')
    ax[0].label_outer()

    frames = range(len(energy))
    t = librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)

    librosa.display.waveplot(signal, sr=sample_rate, alpha=0.5, ax=ax[1])
    ax[1].plot(t, energy, color="r")
    ax[1].set_ylim((-1, 1))
    ax[1].set(title='RMSE')
    ax[1].label_outer()

    librosa.display.waveplot(signal, sr=sample_rate, alpha=0.5, ax=ax[2])
    ax[2].plot(t, zcr, color="r")
    ax[2].set_ylim((-1, 1))
    ax[2].set(title="ZCR")
    ax[2].label_outer()

    plt.show()

    # Figure 2 (Spectrogram, Mel-Spectrogram, MFCC)
    fig, ax = plt.subplots(nrows=3, sharex=False,
                           sharey=False, constrained_layout=True)

    y_to_db = librosa.amplitude_to_db(abs(librosa.stft(signal)))
    librosa.display.specshow(
        y_to_db, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax[0])
    ax[0].set(title='Spectrogram')
    ax[0].label_outer()

    mel_spectogram = librosa.feature.melspectrogram(
        signal, sr=sample_rate, n_fft=frame_length, hop_length=hop_length)
    log_mel_spectogram = librosa.power_to_db(mel_spectogram)
    librosa.display.specshow(
        log_mel_spectogram, x_axis="time", y_axis="mel", sr=sample_rate, ax=ax[1])
    ax[1].set(title='Mel-Spectrogram')

    mfccs = librosa.feature.mfcc(
        y=signal, sr=sample_rate, n_fft=frame_length, hop_length=hop_length)
    librosa.display.specshow(mfccs, x_axis='time', sr=sample_rate, ax=ax[2])
    ax[2].set(title='MFCC')

    plt.show()


# Get the accuracy score for the Background vs Foreground Classifier and Random Forest Classifier
def accuracy(dir, rfc, sample_rate, frame_length, hop_length):
    # List with the digits for all files
    all_digits = []
    # List with all the predictions from Random Forest Classifier
    predictions = []
    # List with the digits for all files that the Background vs Foreground Classifier was correct
    new_all_digits = []
    # Correct number of digits from Background vs Foreground Classifier
    correct_number_of_digits = 0
    # Correct recognised digits from Random Forest Classifier
    correct_recognised_digits = 0
    # Get all the .wav files inside the folder
    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    # For each .wav file
    for fileName in fileList:
        # List with digits of the file
        digits = []
        # Get the name of the file
        label = fileName.split('.wav')[0]
        # For loop for all the characters of the file name with step 2
        for i in range(0, len(label), 2):
            # Add each digit to the list
            digits.append(label[i])
        # Add the digits list to the list for all the files
        all_digits.append(digits)
        # Execute preprocessing for the input file
        signal = preprocessing(dir+fileName, sample_rate)
        # Find the Root Mean Square Energy of the input file
        energy = rmse(signal, frame_length, hop_length)
        # Find the Zero Crossing Rate of the input file
        zcr = zero_crossing_rate(signal, frame_length, hop_length)
        # Seperate background from foreground and get the splitted digits
        numbers = b_vs_f(signal, energy, zcr, frame_length, hop_length)
        # If the number of digits from the Background vs Foreground Classifier is equal to the file's number of digits
        if len(numbers) == len(digits):
            # For each digit
            for digit in digits:
                # Add it to the list
                new_all_digits.append(digit)
            # Add one to the correct number of digits
            correct_number_of_digits = correct_number_of_digits + 1
            # Predict the digits found from the background and foreground seperation
            prediction = recognition(
                rfc, numbers, sample_rate, frame_length, hop_length)
            # Add the predicted digits to the predictions list
            for digit in prediction:
                predictions.append(digit)
            # Add the correct recognised digits
            correct_recognised_digits = correct_recognised_digits + \
                accuracy_score(prediction, digits, normalize=False)

    # Print the percentage of the accuracy of Background vs Foreground Classifier
    print('[Result]: Background vs Foreground Classifier Accuracy: ' +
          str("{:.2f}".format((correct_number_of_digits/len(all_digits)) * 100)) + '%')
    # Print the percentage of the accuracy of Background vs Foreground Classifier
    print('[Result]: Random Forest Classifier Accuracy: ' +
          str("{:.2f}".format((correct_recognised_digits/len(new_all_digits))*100))+'%')
    # Figure for the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(
        new_all_digits, predictions), display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    disp.plot(include_values=True)
    plt.title('Confusion Matrix')
    plt.show()
