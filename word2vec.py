# embedding.py
# Help: https://keras.io/layers/embeddings/
#   https://www.tensorflow.org/guide/embedding
#   https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526
#   https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
#   https://jovianlin.io/embeddings-in-keras/
import tensorflow as tf
# pip3 install keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import os # file structure
import sys # command line arguments
import gensim

# pip3 install sklearn
from sklearn.preprocessing import LabelEncoder

import numpy as np

# read the file, split by ',' , and return the data
def readFile(fileObject, printLine = False):
    data = []
    tmp = []
    for line in fileObject:
        if printLine:
            print(line)
        tmp = list(line)
        del tmp[-1] # delete the new line
        line = "".join(tmp)

        if printLine:
            print(line)

        data.append(line)

    return data

def main():
    print("Word2Vec embedding")

    # Read in the data
    if len(sys.argv) > 1:
        filePath = str(os.getcwd()) + "/" + sys.argv[1] + "/"
    else:
        filePath = str(os.getcwd()) + "/random/"

    # Save model information
    output_dir = str(os.getcwd()) + "/model/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir + "word2vec.txt"
    # Read in the data
    outputFolder = str(os.getcwd()) + "/output/"

    # Data files
    #-------------------------------------------------------------------------
    train_file = filePath + "train_data.txt"
    train_label_file = filePath + "train_protein_data.txt"
    # Large dataset
    # train_file = "D:/embeddings/test_data.txt"
    # train_label_file = str(os.getcwd()) + "/split/test_protein.txt"

    test_file = filePath + "test_data.txt"
    test_label_file = filePath + "test_protein_data.txt"

    # Create file readers for training, testing set
    #-------------------------------------------------------------------------
    print("Creating training file objects")
    try:
    	# labels
        train_label_object = open(train_label_file, "r")
        test_label_object = open(test_label_file, "r")
    	# input data
        train_file_object = open(train_file, "r")
        test_file_object = open(test_file, "r")

    except (OSError, IOError) as e:
    	print(e)
    	exit()

    # Read and output training data
    #-------------------------------------------------------------------------
    train_label_data = []
    train_data = []

    train_label_data = readFile(train_label_object)
    train_data = readFile(train_file_object)

    train_data = np.asarray(train_data)
    train_label_data = np.asarray(train_label_data)

    # Read and output test data
    #-------------------------------------------------------------------------
    test_data = readFile(test_file_object)
    test_label_data = readFile(test_label_object)

    test_data = np.asarray(test_data)
    test_label_data = np.asarray(test_label_data)

    max_length = 0

    for i in train_data:
        if len(i) > max_length:
            max_length = len(i)

    print("Train data: {}".format(len(train_data)))
    print("Max Length: {}".format(max_length))
    # GENSIM MODEL
    #---------------------------------------------------------------------------

    model = gensim.models.Word2Vec(sentences=train_data, size=max_length, window=10, workers=4, min_count = 1)
    words = list(model.wv.vocab)

    print("Gensim Vocab: {}".format(len(words)))

    model.wv.save_word2vec_format(output_file, binary=False)

    # GENSIM MODEL END
    #---------------------------------------------------------------------------

    # read embeddings
    embeddings_index = {}
    f = open(output_file, encoding = "utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embeddings_index[word] = coefs
    f.close()

    # Tensorflow Embedding feeds to CNN network
    #---------------------------------------------------------------------------

    # Convert train "sentences" to list of integers
    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(train_data)
    sequences = tokenizer_obj.texts_to_sequences(train_data)

    # pad sequences to the same length
    train_max_length = max([len(s) for s in train_data])
    train_data = pad_sequences(sequences, maxlen=max_length, padding='post')

    # convert string labels to integers
    labelencoder = LabelEncoder()
    train_encoded_label = labelencoder.fit_transform(train_label_data)

    # Convert test "sentences" to list of integers
    encoded_data = tokenizer_obj.texts_to_sequences(test_data)
    # pad sequences to the same length as train
    test_data = pad_sequences(encoded_data, maxlen=train_max_length, padding='post')

    # convert test labels to integers
    test_encoded_label = labelencoder.fit_transform(test_label_data)

    # define vocabulary size (largest integer value)
    vocab_size = len(tokenizer_obj.word_index) + 1

    print("Vocabulary size: {}".format(vocab_size))

    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=train_max_length))
    model.add(Conv1D(filters=128, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())

    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(train_data, train_encoded_label, epochs=10, verbose=2)
    # evaluate
    loss, acc = model.evaluate(test_data, test_encoded_label, verbose=0)
    print('Test Accuracy: %f' % (acc*100))


    #sp|P38149|DUG2_YEAST
    #
    # string = "27,44,71,72,84,88,93,101,102,104,149,165,167,197,279,294,305,393,409,451,520,538,539,549,560,610,622,623,633,634,660,666,677,678,679,746,767,788,805,855,883,1012,1013,1032,1041,1114,1201,1218,1220,1244,1312,1331,1341,1357,1385,1418,1419,1420,1449,1450,1478,1495,1521,1533,1548,1565,1566,1583,1628,1630,1644,1667,1695,1696,1714,1715,1733,1754,1772,1782,1867,1885,1895,1912,1932,1933,1949,1981,2008,2026,2027,2068,2069,2088,2095,2113,2226,2279,2280,2297,2325,2370,2386,2388,2397,2498,2515,2527,2542,2543,2664,2681,2682,2741,2793,2869,2880,2906,2907,3007,3019,3052,3105,3120,3121,3122,3124,3233,3251,3260,3262,3356,3372,3373,3374,3398,3408,3426,3443"
    #
    # values = model.wv.most_similar_cosmul(positive = string.split(","))
    # print(values)



if __name__ == '__main__':
    main()