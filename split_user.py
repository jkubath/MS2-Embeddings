import pandas as pd
import os
import sys

def main():
	print("Start")
	folder = "C:/Users/koob8/Desktop/embeddings/output/"
	filename = "peptide_ms2.ms2"

	split_folder = "peptide_data/"
	split_name = "peptide_"

	if(len(sys.argv) > 1):
		filename = sys.argv[1]
		print("File: ", folder + filename)
	else:
		print("Default file: ", folder + filename)

	chunksize = 1000000

	f = open(folder + filename, "r")


	# Setup folder structure
	try:
	    # Create target Directory
	    os.mkdirs(folder + split_folder, exist_ok = True)
	    print("Directory " , folder + split_folder ,  " Created ")
	except FileExistsError:
	    print("Directory " , folder + split_folder ,  " already exists")


	count = 0
	fileCount = 0
	outputObject = 0
	# Read the lines of the file
	for line in f:
		if count % chunksize == 0 or fileCount == 0:
			outputObject = open(folder + split_folder + split_name + str(fileCount) + ".ms2", "w")
			print("\tWriting ", str(fileCount))
			fileCount += 1

		outputObject.write(line)

		count += 1

	print("\nDone")


if __name__ == '__main__':
	main()
