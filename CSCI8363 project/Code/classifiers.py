from liblinearutil import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise as pw
from sklearn import model_selection as ms
import numpy as np

def log_regression(train_vectors, train_labels):
    num_class1 = np.sum(np.array(train_labels) < .5)
    num_class2 = len(train_labels) - num_class1
    best_C, _ = train(train_labels, train_vectors, '-C -q -s 0 -w1 {0}'.format(float(num_class2)/num_class1))
    m = train(train_labels, train_vectors, '-c {0} -q -s 0 -w1 {1}'.format(best_C, float(num_class2)/num_class1))
    return m, best_C

def svm_predictor_linear(train_vectors, train_labels):
    num_class1 = np.sum(np.array(train_labels) < .5)
    num_class2 = len(train_labels) - num_class1
    best_C, _ = train(train_labels, train_vectors, '-C -q -s 2 -w1 {0}'.format(float(num_class2)/num_class1))
    m = train(train_labels, train_vectors, '-c {0} -q -s 2 -w1 {1}'.format(best_C, float(num_class2)/num_class1))
    return m, best_C

def train_knn(train_vectors, train_labels):
    best_nn = -1
    train_acc = 0
    best_model = -1

    for nn_iter in range(1,11):
        knn_model = KNeighborsClassifier(n_neighbors=nn_iter)
        score = ms.cross_val_score(knn_model, train_vectors, train_labels, cv = 5, scoring = 'accuracy')
        current_acc = score.mean()
        if current_acc > train_acc:
            train_acc = current_acc
            best_nn = nn_iter

    return best_nn, train_acc

#this is the main function that can handle finding the error rate given a certain classifier for training and testing
def find_error_rate(train_vectors, train_labels, test_vectors, test_labels, classifier = "svm"):

    if classifier == "svm":
        svm_model, best_C = svm_predictor_linear(train_vectors, train_labels)

        test_predictions, test_acc, _ = predict(test_labels, test_vectors, svm_model, '-q')
        train_predictions, train_acc, _ = predict(train_labels, train_vectors, svm_model, '-q')

        param = best_C

        test_acc = float(test_acc[0])/100
        train_acc = float(train_acc[0])/100



    elif classifier == "knn":
        nn, train_acc = train_knn(train_vectors, train_labels)
        knn_model = KNeighborsClassifier(n_neighbors=nn)
        knn_model.fit(train_vectors, train_labels)
        test_acc = knn_model.score(test_vectors, test_labels)

        param = nn

    elif classifier == "cos":
        similarity = pw.cosine_similarity(test_vectors, train_vectors)
        n_v = -1
        test_acc = -1
        train_acc = -1
        for num_votes in range(1,14,2):
            max_indices = [np.argpartition(arr, -1*num_votes)[(-1*num_votes):] for arr in similarity]
            votes = [np.array(train_labels)[max_indices[v]] + 1 for v in range(len(max_indices))]
            votes_added_up = [np.bincount(vot).argmax() - 1 for vot in votes]
            test_acc_temp = float(sum(np.equal(votes_added_up, test_labels)))/len(test_labels)
            if test_acc_temp > test_acc:
                test_acc = test_acc_temp
                n_v = num_votes

        param = n_v

    elif classifier == "log":
        log_model, best_C = log_regression(train_vectors, train_labels)

        test_predictions, test_acc, _ = predict(test_labels, test_vectors, log_model, '-q')
        train_predictions, train_acc, _ = predict(train_labels, train_vectors, log_model, '-q')

        param = best_C

        test_acc = float(test_acc[0])/100
        train_acc = float(train_acc[0])/100

    else:
        return "do a better job of selecting a classifier"


    return test_acc, train_acc, param

