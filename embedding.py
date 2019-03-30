# embedding.py
# Help: https://keras.io/layers/embeddings/
#   https://www.tensorflow.org/guide/embedding
#   https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526
#   https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
import tensorflow as tf
# pip3 install keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences # pad input data with 0's
from keras.preprocessing.text import one_hot # encode labels as integers

# pip3 install sklearn
from sklearn.preprocessing import LabelEncoder

import numpy as np

def main():
    print("Keras embedding")

    # Read in the data
    outputFolder = "C:/Users/koob8/Desktop/embeddings/output/"

    # length of peak array
    # max value in binned_peptide_0 was 6139
    dataLength = 7000

    # training set
    #-------------------------------------------------------------------------
    peptide_folder = outputFolder + "peptide_data/binned/"
    test_file = peptide_folder + "binned_peptide_0.ms2"
    test_label_file = peptide_folder + "label_peptide_0.ms2"

    # validation set
    #-------------------------------------------------------------------------

    # Create file readers for training set
    #-------------------------------------------------------------------------
    print("Creating training file objects")
    try:
    	# labels
        test_label_object = open(test_label_file, "r")
    	# input data
        test_file_object = open(test_file, "r")

    	# write to the output files
    # 	no_noise_output_file = open(filePath + no_noise_output_file, "w")
    # 	noise_output_file = open(filePath + noise_output_file, "w")
    except (OSError, IOError) as e:
    	print(e)
    	exit()

    # Create file readers for validation set
    #-------------------------------------------------------------------------
    # print("Creating validation file objects")
    # try:
    # 	utput_file = open(filePath + valid_no_noise_output_file, "w")
    # except (OSError, IOError) as e:
    # 	print(e)
    # 	exit()

    # Read and output training data
    #-------------------------------------------------------------------------
    test_label_data = []
    test_data = []
    tmp = []
    print("Reading training data")
    for line in test_label_object:
        tmp = list(line)
        del tmp[-1]
        line = "".join(tmp)

        test_label_data.append(line)


    for line in test_file_object:
        tmp = line.split(",")
        del tmp[-1]
        test_data.append(tmp)

    print(len(test_data))
    print(len(test_label_data))
    test_data = np.asarray(test_data)
    test_label_data = np.asarray(test_label_data)

    # Read and output validation data
    #-------------------------------------------------------------------------
    # print("Reading validation data")

    max_value = 6200
    input_dimension = 100
    arrayLength = 300
    dataLength = 6200

    # encoded_data = []
    # for line in test_data:
    #     encoded_data.append([one_hot(d, dataLength) for d in line])
    padded_data = pad_sequences(test_data, maxlen=arrayLength, dtype='int32', value=0, padding="post")

    encoder = LabelEncoder()
    encoder.fit(test_label_data)
    encoded_labels = encoder.transform(test_label_data)
    # convert integers to dummy variables

    # test_labels = []
    # for label in test_label_data:
    #     test_labels.append(one_hot(label, 1000000))

    model = Sequential()
    # max_value: number of different "words"
    # input_dimension: sizer of vector space in which to embed the words
    # dataLength: length of sentence (range of m/z values)
    model.add(Embedding(max_value, input_dimension, input_length=arrayLength))
    # the model will take as input an integer matrix of size (batch, input_length).
    # the largest integer (i.e. word index) in the input should be
    # no larger than 999 (vocabulary size).
    # now model.output_shape == (None, 10, 64), where None is the batch dimension.
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # input: range of numbers 0-999, array dimension of output 32 rows x 10 columns
    #input_array = np.random.randint(1000, size=(32, 10))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())

    # fit the model
    model.fit(padded_data, encoded_labels, epochs=10, verbose=1)

    # evaluate the models
    loss, accuracy = model.evaluate(padded_data, encoded_labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))

    # output_array = model.predict(noise_data)
    # assert output_array.shape == (32, 10, 64)




if __name__ == '__main__':
    main()
