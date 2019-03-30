# embedding.py
# Help: https://keras.io/layers/embeddings/
#   https://www.tensorflow.org/guide/embedding
#   https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526
#   https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
#   https://jovianlin.io/embeddings-in-keras/
import tensorflow as tf
# pip3 install keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences # pad input data with 0's
from keras.preprocessing.text import one_hot # encode labels as integers
import os # file structure
import sys # command line arguments

# pip3 install sklearn
from sklearn.preprocessing import LabelEncoder

import numpy as np

def main():
    print("Keras embedding")

    # Read in the data
    if len(sys.argv) > 1:
        filePath = str(os.getcwd()) + "/" + sys.argv[1] + "/"
    else:
        filePath = str(os.getcwd()) + "/binned/"

    filePath = "/Volumes/Jonah/embeddings/"
    splitPath = str(os.getcwd()) + "/split/"

    # Read in the data
    outputFolder = str(os.getcwd()) + "/output/"

    # training set
    #-------------------------------------------------------------------------
    # peptide_folder = outputFolder + "peptide_data/binned/"
    peptide_folder = filePath + "binned/"
    test_file = peptide_folder + "binned_train_noise.ms2"
    test_label_file = splitPath + "/train_protein.txt"

    # validation set
    #-------------------------------------------------------------------------

    # Create file readers for training set
    #-------------------------------------------------------------------------
    print("Creating training file objects")
    try:
    	# labels
        train_label_object = open(test_label_file, "r")
    	# input data
        train_file_object = open(test_file, "r")

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
    # 	output_file = open(filePath + valid_no_noise_output_file, "w")
    # except (OSError, IOError) as e:
    # 	print(e)
    # 	exit()

    num_lines = 1000
    count = 0
    maxLine = 0
    # Read and output training data
    #-------------------------------------------------------------------------
    train_label_data = []
    train_data = []
    tmp = []
    print("Reading training data")
    for line in train_label_object:
        tmp = list(line)
        del tmp[-1] # delete the new line
        line = "".join(tmp)

        train_label_data.append(line)

        count += 1
        # if count % 100 == 0:
        #     print("Count {}".format(count))
        if count == num_lines:
            break

    count = 0
    for line in train_file_object:
        tmp = line.split(",")
        del tmp[-1] # delete the new line
        int_val = []
        for i in range(len(tmp)):
            int_val.append(int(tmp[i]))

        train_data.append(int_val)

        if len(tmp) > maxLine:
            maxLine = len(tmp)

        count += 1
        # if count % 100 == 0:
        #     print("Count {}".format(count))
        if count == num_lines:
            break

    count = 0
    while count < 10:
        print("{} {}".format(train_label_data[count], train_data[count][1:5]))
        count += 1

    # print(maxLine)
    # print(len(train_data))
    # print(len(train_label_data))

    train_data = np.asarray(train_data)
    train_label_data = np.asarray(train_label_data)

    # Read and output validation data
    #-------------------------------------------------------------------------
    # print("Reading validation data")

    max_value = 6000
    input_dimension = 100
    arrayLength = 400
    dataLength = 2000

    labelencoder = LabelEncoder()
    encoded_label = labelencoder.fit_transform(train_label_data)

    # encode string labels as integers
    # encoded_label = []
    # for i in range(len(train_label_data)):
    #     encoded_label.append([])
    #     if i < 5:
    #         print("{}".format(train_label_data[i]))
    #     encoded_label[i] = one_hot(train_label_data[i], dataLength)

    print(encoded_label[1:5])

    # encode all
    padded_data = pad_sequences(train_data, maxlen=arrayLength, padding="post")
    print(padded_data[1:5])

    print(len(encoded_label))
    print(len(padded_data))

    model = Sequential()
    # max_value: number of different "words"
    # input_dimension: sizer of vector space in which to embed the words
    # dataLength: length of sentence (range of m/z values)
    model.add(Embedding(input_dim = max_value,
                        output_dim = max_value,
                        input_length=arrayLength))
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
    model.fit(x=padded_data, y=encoded_label, epochs=10, verbose=1)

    # evaluate the models
    loss, accuracy = model.evaluate(padded_data, train_label_data, verbose=0)
    print('Accuracy: %f' % accuracy)

    # output_array = model.predict(noise_data)
    # assert output_array.shape == (32, 10, 64)




if __name__ == '__main__':
    main()
