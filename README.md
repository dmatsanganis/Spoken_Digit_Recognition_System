# Spoken Digit Recognition System

This project is a spoken digit recognition system that uses audio signal processing and machine learning techniques to identify and output spoken digits from input audio files.

The system works by preprocessing the audio signal, extracting relevant features, and using a trained Random Forest Classifier to recognize the spoken digits. The system is capable of separating the audio signal into segments representing background (silence) and foreground (spoken digits), making it effective for continuous speech recognition.

## How it works

1. **Preprocessing**: This stage involves reading the input audio file with a specific sampling rate and applying a FIR (Finite Impulse Response) band-pass filter to the signal. This filter allows only a specific range of frequencies to pass through, which helps to eliminate noise and other irrelevant information from the signal.

2. **Training**: The system is trained using audio files of spoken digits, stored in a training dataset. For each audio file, the MFCC (Mel Frequency Cepstral Coefficients) features are extracted and used as input to a Random Forest Classifier. The labels (digits spoken in the audio files) are also provided to the classifier for training.

3. **Root Mean Square Energy & Zero Crossing Rate**: For the input audio file, the system calculates the Root Mean Square Energy and the Zero Crossing Rate. These features are used in the next step to separate the signal into segments representing background (silence) and foreground (spoken digits).

4. **Background vs. Foreground Classification**: Based on the energy and zero-crossing rate, each frame of the signal is classified as either background or foreground. Segments of the signal corresponding to foreground frames are considered as potential digit utterances.

5. **Digit Recognition**: For each identified digit utterance, MFCC features are again extracted and provided to the trained Random Forest Classifier for prediction.

6. **Accuracy & Confusion Matrix**: The system computes the accuracy of the Background vs. Foreground Classifier and the Random Forest Classifier using a test dataset. A confusion matrix is also plotted to visualize the performance of the digit recognition system.

## Contributors

- [x] [Dimitris Matsanganis](https://github.com/dmatsanganis)


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

