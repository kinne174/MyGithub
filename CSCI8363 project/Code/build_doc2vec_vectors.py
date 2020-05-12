import statsmodels.api as sm
from sklearn import preprocessing

#from current doc2vec model get document vectors for the training and testing datasets
#doc2vec models are trained with testing documents so can get the vectors straight from the model or can infer them
#the infer default parameters are untested
#scaling for doing svm
def doc2vec_vectors(current_model, train_docs, test_docs, tag_decoder, infer = True, infer_steps = 5, infer_alpha = 0.025):
    train_labels, train_vectors = zip(*[(doc.sentiment, current_model.docvecs[tag_decoder[doc.tags[0]]]) for doc in train_docs])
    min_max_scaler = preprocessing.MinMaxScaler()
    train_vectors = min_max_scaler.fit_transform(train_vectors)
    train_vectors = sm.add_constant(train_vectors)

    if infer:
        test_vectors = [current_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in test_docs]
        test_labels = [doc.sentiment for doc in test_docs]
    else:
        test_labels, test_vectors = zip(*[(doc.sentiment, current_model.docvecs[tag_decoder[doc.tags[0]]]) for doc in test_docs])

    test_vectors = min_max_scaler.transform(test_vectors)
    test_vectors = sm.add_constant(test_vectors)

    return train_vectors, train_labels, test_vectors, test_labels



