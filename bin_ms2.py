import os  # file system

# writes the data in a csv format to the file


def outputData(writer, data, index, length):
    # Error checking for empty data
    if len(data) == 0 or len(data[index]) == 0:
        print("Data is empty. Index: {}".format(index))
        return -1
    if not writer:
        print("Ouput file object is null")
        return -1

    # write the data to file
    i = 0
    while i < length - 1:
        writer.write(str(int(data[index][i])) + ",")
        i += 1

    writer.write(str(data[index][i]) + "\n")


def binLines(fileObject, outputObject, labelObject):
    if not fileObject:
        print("Input file failed")
        return

    peptide_count = 0
    # read all the data in the file
    for line in fileObject:

        # hold the peak data
        splitLine = line.split(",")
        if len(splitLine) == 1:
            continue

        data = []
        # Read and bin the line
        count = 0
        for item in splitLine:
            # print the peptide label
            if count == 0:
                labelObject.write(item + "\n")
                count += 1
                continue

            # cast data to a float, then round, and cast back to integer
            value = int(round(float(item)))

            data.append(value)

            count += 1

        # sort the m/z values
        data = sorted(data)
        outputString = []
        for i in data:
            outputString.append(str(i))
            outputString.append(",")

        # finished reading all the line data
        # remove the last ',' and replace with newline
        outputString[-1] = "\n"
        outputString = "".join(outputString)

        # write the line to file
        outputObject.write(outputString)

        peptide_count += 1

        if peptide_count % 100000 == 0:
            print("Wrote {} peptides".format(peptide_count))


def binFiles(folderPath):
    outputFolder = folderPath + "binned/"
    fileBase = "binned_"
    labelBase = "label_"

    # Create the output directory
    try:
        os.makedirs(outputFolder, exist_ok=True)
    except OSError:
        print("Failed creating dir: {}".format(outputFolder))
        return
    else:
        print("Created output dir: {}".format(outputFolder))

    fileCount = 0
    for file in os.listdir(folderPath):
        # skip if we aren't reading from a file (i.e. folders)
        if not os.path.isfile(os.path.join(folderPath, file)):
            continue

        print("Reading: {}".format(file))

        inputFile = open(folderPath + file, "r")
        # create output files
        outputFile = open(outputFolder + fileBase + file, "w")
        labelFile = open(outputFolder + labelBase + file, "w")

        binLines(inputFile, outputFile, labelFile)


def binMS2(fileObject, outputObject):
    if not fileObject:
        print("Input file failed")
        return

    peptide_count = 0
    data = []  # m/z array
    # read all the data in the file
    for line in fileObject:

        # hold the peak data
        splitLine = line.split()
        if len(splitLine) == 1:
            continue

        outputString = []
        # we are at a new spectrum, but not the first header
        if splitLine[0][0] == "S" and not len(data) == 0:
            for i in data:
                outputString.append(str(i))
                outputString.append(",")

            outputString[-1] = "\n"
            outputString = "".join(outputString)

            outputObject.write(outputString)

            data = []
            peptide_count += 1

            if peptide_count % 10000 == 0:
                print("Wrote {} peptides".format(peptide_count))
            continue

        # skip header information
        if splitLine[0][0] == "Z" or splitLine[0][0] == "S" or splitLine[0][0] == "H":
            continue

        # add the m/z value to the data list
        m_z = int(round(float(splitLine[0])))
        if m_z not in data:
            data.append(m_z)

def binMS2Files(folderPath):
    outputFolder = folderPath + "binned/"
    fileBase = "binned_"

    # Create the output directory
    try:
        os.makedirs(outputFolder, exist_ok=True)
    except OSError:
        print("Failed creating dir: {}".format(outputFolder))
        return
    else:
        print("Created output dir: {}".format(outputFolder))

    fileCount = 0
    for file in os.listdir(folderPath):
        # skip if we aren't reading from a file (i.e. folders)
        if not os.path.isfile(os.path.join(folderPath, file)):
            continue

        print("Reading: {}".format(file))

        inputFile = open(folderPath + file, "r")
        # create output files
        outputFile = open(outputFolder + fileBase + file, "w")

        binMS2(inputFile, outputFile)

def main():
    print("Bin MS2 script")

    # Default variables
    # folderPath = "C:/Users/koob8/Desktop/embeddings/output/peptide_data/"
    # folderPath = "C:/Users/koob8/Desktop/embeddings/split/peptide_data/"
    folderPath = "H:/embeddings/"

    # binFiles(folderPath)
    binMS2Files(folderPath)

if __name__ == '__main__':
    main()
