from sklearn.feature_extraction import text as fet
from nltk.tokenize import TreebankWordTokenizer
from scipy.sparse.linalg import svds
from scipy.sparse.csr import csr_matrix
import numpy as np
import time

#from training/testing data create BOW matrix and determine if full BOW will be used for feature vectors or
#if PCA should be done
def BOW_vectors(train_docs, test_docs, do_PCA, num_PCA):

    print "Beginning to train BOW..."
    start = time.clock()

    train_sentences = [" ".join(doc.words) for doc in train_docs]
    test_sentences = [" ".join(doc.words) for doc in test_docs]

    count_vectorizer = fet.CountVectorizer(analyzer="word", lowercase=True, binary=True, stop_words=None, tokenizer=TreebankWordTokenizer().tokenize)
    BOW_train = count_vectorizer.fit_transform(train_sentences)
    BOW_test = count_vectorizer.transform(test_sentences)

    if do_PCA == True:
        u, s, vt = svds(A=csr_matrix.transpose(BOW_train.asfptype()),k=num_PCA,which="LM",return_singular_vectors=True)
        #train_vectors = projected_BOW(BOW_train, u, s)
        train_vectors = vt.transpose()
        test_vectors = projected_BOW(BOW_test, u, s)
    else:
        train_vectors = BOW_train.todense()
        test_vectors = BOW_test.todense()

        print "BOW matrix has %d unique terms" % (BOW_train.shape[1])

    train_labels = [doc.sentiment for doc in train_docs]
    test_labels = [doc.sentiment for doc in test_docs]

    end = time.clock()
    print "Ending BOW training... took %f seconds" % (end - start)

    return train_vectors, train_labels, test_vectors, test_labels

#get query terms projected into eigen space of svd of training data
def projected_BOW(BOW, u, s):
    k = len(s)
    num_docs = BOW.shape[0]
    PCA_return = np.array([[0.]*num_docs]*k)

    for i in range(k):
        u_col = u[:,i]
        u_col_mat = np.array([u_col.tolist()]*num_docs)
        PCA_col = np.array(BOW.multiply(u_col_mat).sum(1) * (s[i])**-1).T
        PCA_return[i] = PCA_col

    return PCA_return.T







