from Bio import SeqIO
import random
import os

# Set of peptides cleaved from the various proteins
peptide_data = {}

# Read the given fasta file
def readFasta(fastaPath, outputFilePath = "C:/", writeOutput = False):
    # open the fasta for reading
    fileObject = open(fastaPath, "r")

    fastaData = SeqIO.parse(fileObject, 'fasta')

    if writeOutput:
        outputFileObject = open(outputFilePath, "w")

        for fasta in fastaData:
            string = "{},{}\n".format(fasta.id, fasta.seq)
            outputFileObject.write(string)

# Break the string into all the various substrings (peptides)
def cleaveProtein(proteinSequence, minLength, maxLength):
    peptides = []

    for i in range(len(proteinSequence)):
        for j in range(minLength, maxLength):
            if j + i >= len(proteinSequence):
                break
            # print("peptide: {}".format(proteinSequence[i:j]))
            # get all the substrings
            if not j - i < 6:
                peptides.append(proteinSequence[i:j])

    return peptides

# call cleaveProtein on all the proteins.  This will generate all the peptide
# sequences and save them to peptide_data
def generatePeptide(proteinOutputFile, minLen = 6, maxLen = 50):
    global peptide_data

    # open the fasta for reading
    fileObject = open(proteinOutputFile, "r")

    proteinTotal = 0
    peptideTotal = 0

    for line in fileObject:
        splitLine = line.split(",")
        id = splitLine[0]
        sequence = splitLine[1]

        # print("Protein id: {}".format(id))
        # print("Sequence: {}".format(sequence))

        proteinTotal += 1

        cleaved_peptides = cleaveProtein(sequence, minLen, maxLen)

        peptideTotal += len(cleaved_peptides)

        for peptide in cleaved_peptides:
            # skip if the sequence has already been found
            if peptide in peptide_data:
                continue
            else:
                peptide_data[peptide] = [0]
                continue

        if proteinTotal % 100 == 0:
            print("Read {} proteins".format(proteinTotal))

# Read the fasta, randomly choose which file to output to (train, valid, test)
# and write the peptide sequence and protein label to each file
def writeProteinPeptide(fastaPath, outputDir, minLen = 6, maxLen = 50):
    # open the fasta for reading
    fileObject = open(fastaPath, "r")

    fastaData = SeqIO.parse(fileObject, 'fasta')

    minTrain = 30
    minValid = 10

    train_proteinObject = open(outputDir + "train_protein.txt", "w")
    train_peptideObject = open(outputDir + "train_peptide.txt", "w")
    valid_proteinObject = open(outputDir + "valid_protein.txt", "w")
    valid_peptideObject = open(outputDir + "valid_peptide.txt", "w")
    test_proteinObject = open(outputDir + "test_protein.txt", "w")
    test_peptideObject = open(outputDir + "test_peptide.txt", "w")

    count = 0
    cleaved_peptides = []
    for fasta in fastaData:
        cleaved_peptides = cleaveProtein(fasta.seq, minLen, maxLen)

        for peptide in cleaved_peptides:
            choice = random.randint(1, 101) # random value from 1 - 100
            if choice > minTrain:
                train_proteinObject.write("{}\n".format(fasta.id))
                train_peptideObject.write("{}\n".format(peptide))
            elif choice > minValid:
                valid_proteinObject.write("{}\n".format(fasta.id))
                valid_peptideObject.write("{}\n".format(peptide))
            else:
                test_proteinObject.write("{}\n".format(fasta.id))
                test_peptideObject.write("{}\n".format(peptide))

            if count % 10000 == 0:
                print("Wrote {} peptides".format(count))
            count += 1

# Write the peptide_data to the peptideOutputFile
def writePeptideData(peptideOutputFile):
    global peptide_data

    outputObject = open(peptideOutputFile, "w")

    string = ""
    for peptide, charges in peptide_data.items():
        outputObject.write(peptide + "\n")


def main():
    print("Generating peptide data from fasta")

    # Read in the data
    outputFolder = "C:/Users/koob8/Desktop/embeddings/split/"

    os.makedirs(outputFolder, exist_ok = True)

    proteinOutputFile = outputFolder + "proteinData.txt"
    peptideOutputFile = outputFolder + "peptideData.txt"
    fastaFile = "yeast.fasta"

    # Read fasta file with SeqIO api
    # readFasta(fastaFile, proteinOutputFile, True)

    # generate peptides (every substring of length 6 to 50)
    # generatePeptide(proteinOutputFile, 6, 50)

    # write strings to output file
    # writePeptideData(peptideOutputFile)

    # write protein as label and peptide as input data
    writeProteinPeptide(fastaFile, outputFolder)

if __name__ == '__main__':
    main()
