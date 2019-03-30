import random
import os
import numpy

def main():
    print("Generating peptide data from fasta")

    # Read in the data
    outputFolder = str(os.getcwd()) + "/random/"

    os.makedirs(outputFolder, exist_ok = True)

    proteinOutputFile = outputFolder + "train_protein_data.txt" # label
    ms2OutputFile = outputFolder + "train_ms2.txt" # ms2 data to go with the label

    protein_file = open("./split/train_protein.txt", "r")
    ms2_file = open("/Volumes/Jonah/embeddings/binned/binned_train_no_noise.ms2")

    protein_output = open(proteinOutputFile, "w")
    ms2_output = open(ms2OutputFile, "w")

    count = 10000
    lines = numpy.random.randint(0, 580000, size=count)

    lines = numpy.sort(lines)

    index = 0
    lineCount = 0
    index = lines[0]
    for line in protein_file:
        if index >= count:
            break
        if lineCount < lines[index]:
            lineCount += 1
            continue
        elif lineCount == lines[index]:
            # print this line
            if index < 5:
                print(index)
            protein_output.write(line)

            curVal = lines[index]
            index += 1
            while index < count and curVal == lines[index]:
                index += 1

    index = 0
    lineCount = 0
    index = lines[0]
    for line in ms2_file:
        if index >= count:
            break
        if lineCount < lines[index]:
            lineCount += 1
            continue
        elif lineCount == lines[index]:
            # print this line
            if index < 5:
                print(index)
            ms2_output.write(line)

            curVal = lines[index]
            index += 1
            while index < count and curVal == lines[index]:
                index += 1



if __name__ == '__main__':
    main()
