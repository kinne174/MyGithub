from sklearn.feature_extraction import text as fet
from nltk.tokenize import TreebankWordTokenizer
from scipy.sparse.linalg import svds
from scipy.sparse.csr import csr_matrix
import numpy as np
import time

def BOW_vectors(classifying_docs, do_PCA, num_PCA):

    print "Beginning to train BOW..."
    start = time.clock()

    classifying_sentences = [" ".join(doc.words) for doc in classifying_docs]

    count_vectorizer = fet.CountVectorizer(analyzer="word", lowercase=True, binary=True, stop_words=None, tokenizer=TreebankWordTokenizer().tokenize)
    BOW_classifying = count_vectorizer.fit_transform(classifying_sentences)

    if do_PCA == True:
        u, s, _ = svds(A=csr_matrix.transpose(BOW_classifying.asfptype()),k=num_PCA,which="LM",return_singular_vectors=True)
        classifying_vectors = projected_BOW(BOW_classifying, u, s)
    else:
        classifying_vectors = BOW_classifying

        print "BOW matrix has %d unique terms" % (BOW_classifying.shape[1])

    classifying_labels = [doc.sentiment for doc in classifying_docs]

    end = time.clock()
    print "Ending BOW training... took %f seconds" % (end - start)

    return classifying_vectors, classifying_labels


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







