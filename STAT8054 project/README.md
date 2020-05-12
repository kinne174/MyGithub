# STAT8054-project

This code is for my end of semester project for an applied statistics class in the Spring of 2017. The project was to develop a R package that could be implemented in CRAN. I chose to do my project as an application of [1] which is a classification method based on a dictionary method. The methodology in the paper is able to dually train the dictionary and reference matrix to use in prediction. The details are in my report. 

The code is done entirely in R and includes functions that are completely implementable. In my original package I included a dataset to test the functions with so please email me if you would like to use it to test the functions. I found it online here http://www.tc.umn.edu/~elock/TCGA_Breast_Data.Rdata but if the link is ever broken please let me know. 

At the time of finishing the project it was the only implimentation of the paper on R. The authors of [1] have done an implimentation of their own in MATLAB though if that language is more to your preference. My package is not currently on CRAN. 

[1] Aharon, Michal, Michael Elad, and Alfred Bruckstein. "$ rm k $-SVD: An algorithm for designing overcomplete dictionaries for sparse representation." IEEE Transactions on signal processing 54.11 (2006): 4311-4322.
