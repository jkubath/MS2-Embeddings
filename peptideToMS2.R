# MSnbase is used to generate MS2 data
# http://www.bioconductor.org/packages/release/bioc/html/MSnbase.html
rm(list=ls())

# Install software for MSnbase
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("MSnbase", version = "3.8")

# Generates the m/z data for the given peptides
library("MSnbase")

filePath = "C:/Users/koob8/Desktop/embeddings/output/peptideData.txt"
outputPath = "C:/Users/koob8/Desktop/embeddings/output/peptide_ms2.ms2"

# read the data
data = readLines(filePath)

# found = FALSE
# count = 0
printed = 0 # count of peptides written to file

# generate the ms2 data for each peptide and write the output file
for(i in data){
  #------------------------------------------------------------------------------------------
  # this code was used because m/z was cancelled halfway through the generation
  # DQDKAS was the current peptide in the data object when stopped
  # This code will loop through until DQDKAS is found and then continue generation from there
  #
  # if(i == "DQDKAS"){
  #   print("Found")
  #   found = TRUE
  # }
  # else if(found) {
  #   #print(i)
  #   count = count + 1
  # }
  # else {
  #   printed = printed + 1
  #   next
  # }
  #------------------------------------------------------------------------------------------
  
  # use MSnbase to calculate the m/z positions of the peptide sequence
  mass = calculateFragments(i)
  # get the m/z values from the returned object (the entire first column)
  list = mass[,1]
  # values are b-ions then y-ions, but we want them sorted by m/z value
  sort.list(list)

  # Add all the values to a string to be written to file
  line = i
  for(j in list) {
    value = sprintf("%.2f", j)
    line = paste(line, value, sep=",")
  }
  line = paste(line, "\n", sep="")

  # Write the m/z data to file
  write(line, file = outputPath, append = TRUE)
  
  printed = printed + 1
  
  if(printed %% 100000 == 0) {
    print("Wrote %d peptides", printed)
  }
  
} 


