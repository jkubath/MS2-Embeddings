# MS2-Embeddings

# Summary
  In the realm of protein research, a common task is to match Tandem Mass Spectrum
  data (MS2) to a protein.  MS2-Embeddings attempts to use word embeddings as the
  first layer of a neural network.  The input to the embeddings layer is a "sentence"
  which is the integer values of the m/z data from the MS2 data.

  Example: "10,25,49,55,60" (Most sentences are more than 1400 integers long)

  The output from the embedding layer is sent to a shallow Convolutional layer
  which is followed by a few smaller dense layers ending with a single final output.

  The output is an integer encoded version of the protein id that matches the input.

# Files
  1. MS2_embedding.py is the main python script that runs the network
    A. Uses protein.py to read MS2 data and write data to appropriate files
  2. gen_peptide.py is used to create training data.  A fasta file is read with protein
    sequences, the peptide substrings are generated, and output to file
  3. choose_random.py is used to read large two large files and split the data into
    train, validation, and test files.  The two files are:
      1: theoretical peptide MS2 data
      2: the protein label to go with peptide
    The files are read and it is randomly chosen whether to write to train, validation,
    or test output.

  peptideToMS2.R can be used with R-Studio to generate MS2 data for peptides strings.
  In our study, we used MaSS-Simulator software written in Java to output MS2 data.
    source: https://github.com/pcdslab/MaSS-Simulator
