# CSCI8363-project

This code is for my end of semester project for a topics course in computer science that focused on using linear algebra in data exploration. I took the course in the Fall of 2017. In the project I wanted to test different methods to do sentiment analysis of short texts such as tweets. My goal was to compare different kinds of feature vectors that I could supply to a linear SVM and see which one the SVM was most able to seperate given the labels. The different feature vectors I tested were doc2vec, Bag-of-Words, and some dimension reduction/expansion of both doc2vec and Bag-of-Words. The details are in my report.

The short synopsis of my report is that I found while doc2vec did well the Bag-of-Words did best. Neither was super close to a state of the art classification rate but this project was more as practice and for feeling out doc2vec than anything. I hope to continue to use doc2vec in other applications to see if I can do better.

The code is entirely in Python using the gensim package to do the doc2vec and the libLinear package to do linear SVM. 
