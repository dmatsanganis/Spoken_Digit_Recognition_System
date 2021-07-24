from functions import *

SAMPLE_RATE = 8000
FRAME_LENGTH = 256
HOP_LENGTH = 256

# Execute preprocessing for the input file
signal = preprocessing('testing/3_8_6_4_0.wav', SAMPLE_RATE)

# Training of the dataset
rfc = build_dataset('training/',
                    SAMPLE_RATE, FRAME_LENGTH, HOP_LENGTH)

# Find the Root Mean Square Energy of the input file
energy = rmse(signal, FRAME_LENGTH, HOP_LENGTH)

# Find the Zero Crossing Rate of the input file
zcr = zero_crossing_rate(signal, FRAME_LENGTH, HOP_LENGTH)

# Seperate background from foreground and get the splitted digits
numbers = b_vs_f(signal, energy, zcr, FRAME_LENGTH, HOP_LENGTH)

# Predict the digits found from the Background vs Foreground Classifier
prediction = recognition(rfc, numbers, SAMPLE_RATE, FRAME_LENGTH, HOP_LENGTH)


# plots(signal, energy, zcr, SAMPLE_RATE, FRAME_LENGTH, HOP_LENGTH)
# accuracy('testing/', rfc, SAMPLE_RATE, FRAME_LENGTH, HOP_LENGTH)
