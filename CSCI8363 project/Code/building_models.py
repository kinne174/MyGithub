from random import shuffle
import datetime
from classifiers import find_error_rate
from build_doc2vec_vectors import doc2vec_vectors
from feature_generator import feature_generator

#train and test classifying models to see which parameter is best to use in training the model
#also trains a doc2vec model over 20 passes
#alpha_list is made up of tuples that is the starting learning rate and the ending learning rate

def choose_best_model(models_by_name, all_docs, train_docs, test_docs, tag_decoder, deg, classifier_in = "svm", alpha_list= "none"):

    print "START all models and learning rates %s" % str(datetime.datetime.now())

    best_alpha_acc = 0

    alpha_list = [(0.025,0.001)] if alpha_list == "none" else alpha_list #list of tuples
    all_docs_copy = all_docs[:]

    for alpha_start, alpha_min in alpha_list:
        passes = 20
        alpha = alpha_start
        alpha_delta = (alpha_start - alpha_min) / passes

        best_model_acc = 0

        models_by_name_copy = models_by_name.copy()

        print "START %s learning rate max: %f min %f" % (str(datetime.datetime.now()), alpha_start, alpha_min)
        for epoch in range(passes):
            shuffle(all_docs_copy)#need all_docs in console

            for model_name, model in models_by_name_copy.items(): #need models_by_name in console
                model.alpha, model.min_alpha = alpha, alpha
                model.train(all_docs_copy, total_examples = len(all_docs_copy), epochs = 1)

                if epoch == (passes - 1):

                    train_vectors, train_labels, test_vectors, test_labels = doc2vec_vectors(model, train_docs, test_docs, tag_decoder=tag_decoder, infer= False)

                    if deg > -1:
                        train_vectors = feature_generator(train_vectors, deg)
                        test_vectors = feature_generator(test_vectors, deg)

                    current_test_acc, current_train_acc, current_param = find_error_rate(train_vectors, train_labels, test_vectors, test_labels, classifier=classifier_in)

                    best_so_far = ""

                    if current_test_acc > best_model_acc:
                        best_model_acc = current_test_acc
                        best_param = current_param
                        best_model = model
                        best_model_name = model_name
                        best_so_far = "*"

                    print "%s%f: testing from %s on pass %d" % (best_so_far, current_test_acc, model_name, epoch)
                    if not classifier_in == "cos":
                        print "%f: training from %s on pass %d" % (current_train_acc, model_name, epoch)

                    if epoch % (passes-1) == 0:
                        train_vectors, train_labels, test_vectors_infer, test_labels_infer = doc2vec_vectors(model, train_docs, test_docs, tag_decoder=tag_decoder)

                        if deg > -1:
                            train_vectors = feature_generator(train_vectors, deg)
                            test_vectors_infer = feature_generator(test_vectors_infer, deg)

                        current_test_acc_infer, current_train_acc, current_param_infer = find_error_rate(train_vectors, train_labels, test_vectors_infer, test_labels_infer, classifier=classifier_in)

                        best_so_far = ""

                        if current_test_acc_infer > best_model_acc:
                            best_so_far = "*"

                        print "%s%f: testing from %s on pass %d" % (best_so_far, current_test_acc_infer, model_name + "_inferred", epoch)

            alpha -= alpha_delta

        print "END %s learning rate max: %f min %f" % (str(datetime.datetime.now()), alpha_start, alpha_min)

        if best_model_acc > best_alpha_acc:
            best_alpha_acc = best_model_acc
            best_alpha_min = alpha_min
            best_alpha_start = alpha_start
            best_alpha_model = best_model
            best_alpha_param = best_param
            best_alpha_name = best_model_name

    print "Best model is %s with an accuracy of %f using alpha_start %f and alpha_min %f and using classifier parameter %f" % (best_alpha_name, best_alpha_acc, best_alpha_start, best_alpha_min, best_alpha_param)

    return best_alpha_model, best_alpha_param

