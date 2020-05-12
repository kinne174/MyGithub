# STAT8931-project
This code is for my end of semester project for a topics course in the statistics department which focused on estimating and using precision matrices in high dimensional settings. I took the course in the Fall of 2017. 

My project was to do sentiment analysis of tweets from the Sentiment140 dataset which was set up by students at Stanford and can be found at http://help.sentiment140.com/for-students/. I used the algorithm proposed by Le and Mikolov [1] that is implimented in the Python library gensim called doc2vec to transform the tweets from text data to numeric feature vectors. Then I compared classification rates using different methods of computing a precision matrix to supply to an LDA/QDA classifier. My findings and complete methodology are in the paper supplied but the short synopsis is that the combination of an inverse sample covariance matrix and LDA was the best result with a classification rate of around 73%.

My code is in two parts. The Python code is written to generate the feature vectors doc2vec from the twitter dataset. I also do a lot of cleaning of the tweets in the code. The other part of the code is in R which is to do the precision matrix estimates and do LDA/QDA classification. The Python code is based on a tutorial found at https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb and the R code is largely Adam Rothman's (the professor of the topics course) who allowed me to manipulate his code posted on the course website. There will write outs and read ins from folders on my computer in the code that was necessary since I wasn't able to find a good way of doing everything in one console. 

Thank you and please email me with any questions! kinne174 at umn.edu

[1] Le, Quoc, and Tomas Mikolov. "Distributed representations of sentences and documents." Proceedings of the 31st International Conference on Machine Learning (ICML-14). 2014.
