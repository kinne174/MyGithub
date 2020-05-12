import statsmodels.api as sm
from sklearn import preprocessing

#from current doc2vec model get document vectors for the classifyinging and testing datasets
#doc2vec models are classifyinged with testing documents so can get the vectors straight from the model or can infer them
#the infer default parameters are untested
#scaling for doing svm
def doc2vec_vectors(current_model, classifying_docs, scale = True):
    classifying_labels, classifying_vectors = zip(*[(doc.sentiment, current_model.docvecs[doc.tags[0]]) for doc in classifying_docs])

    if scale:
        min_max_scaler = preprocessing.MinMaxScaler()
        classifying_vectors = min_max_scaler.fit_transform(classifying_vectors)
        classifying_vectors = sm.add_constant(classifying_vectors)

    return classifying_vectors, classifying_labels



