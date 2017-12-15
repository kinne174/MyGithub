from liblinearutil import *
from build_doc2vec_vectors import doc2vec_vectors
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise as pw

#function to take best model selected and get final classification rate on validation set using training+testing as the new
#training set
def validation_testing(train_docs, test_docs, validation_docs, best_param, feature_generator, classifier_in, tag_decoder, model = "none"):
    if feature_generator == "doc2vec" and model == "none":
        return "need a model"

    training_docs = train_docs + test_docs

    validation_acc_infer = -1
    validation_acc = -1

    if feature_generator == "doc2vec":
        _, _, validation_vectors, validation_labels = doc2vec_vectors(model, training_docs, validation_docs, tag_decoder, infer=False)
        training_vectors, training_labels, validation_vectors_infer, validation_labels_infer = doc2vec_vectors(model, training_docs, validation_docs, tag_decoder)


    elif feature_generator == "princomp":
        pass

    if classifier_in == "svm":
        num_class1 = np.sum(np.array(training_labels) < .5)
        num_class2 = len(training_labels) - num_class1
        svm_m = train(training_labels, training_vectors, '-c {0} -q -s 2 -w1 {1}'.format(best_param, float(num_class2)/num_class1))

        _, validation_acc, _ = predict(validation_labels, validation_vectors, svm_m, '-q')

        validation_acc = float(validation_acc[0]) / 100

        _, validation_acc_infer, _ = predict(validation_labels_infer, validation_vectors_infer, svm_m, '-q')

        validation_acc_infer = float(validation_acc_infer[0]) / 100

    elif classifier_in == "knn":
        knn_m = KNeighborsClassifier(n_neighbors=best_param)
        knn_m.fit(training_vectors, training_labels)
        validation_acc = knn_m.score(validation_vectors, validation_labels)
        validation_acc_infer = knn_m.score(validation_vectors_infer, validation_labels_infer)

    elif classifier_in == "cos":
        similarity = pw.cosine_similarity(validation_vectors, training_vectors)
        max_indices = [np.argpartition(arr, -1 * best_param)[(-1 * best_param):] for arr in similarity]
        votes = [training_labels[max_indices[v]] for v in range(len(max_indices))]
        votes_added_up = [np.bincount(vot).argmax() for vot in votes]
        validation_acc = float(sum(votes_added_up == validation_labels)) / len(validation_labels)

        similarity = pw.cosine_similarity(validation_vectors_infer, training_vectors)
        max_indices = [np.argpartition(arr, -1 * best_param)[(-1 * best_param):] for arr in similarity]
        votes = [training_labels[max_indices[v]] for v in range(len(max_indices))]
        votes_added_up = [np.bincount(vot).argmax() for vot in votes]
        validation_acc_infer = float(sum(votes_added_up == validation_labels_infer)) / len(validation_labels_infer)

    print "The validation error is %f and the validation_inferred error is %f" %(validation_acc, validation_acc_infer)

