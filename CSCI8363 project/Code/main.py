from clean_data import ALL_clean_dataset, SOME_clean_dataset
from create_named_documents import get_named_documents
from building_doc2vec import build_doc2vec
from building_models import choose_best_model
from validation_testing import validation_testing
from create_BOW import BOW_vectors
from classifiers import find_error_rate

#####################################################################################################
#main function to run from top to bottom, clean tweets, build feature vectors, and classify
#options are using regular doc2vec, using BOW, and doc2vec expansion using half Gaussian kernel
#####################################################################################################
def doc2vec_classifier(alpha_list= "none", classifier_in = "svm", num_samples_wanted = int(200000), file_name_list = ["testdata.manual.2009.06.14", "training.1600000.processed.noemoticon"], use_current_tweet_iterate = True, use_neutral_tweets = False, size_in = 1000, deg_in = -1):

    if not use_current_tweet_iterate:
        split_list, sentiment_list, tag_decoder_list = ALL_clean_dataset(file_name_list, use_neutral_tweets, num_samples_wanted)
    else:
        split_list, sentiment_list, tag_decoder_list = SOME_clean_dataset(file_name_list, use_neutral_tweets, num_samples_wanted)

    all_docs, train_docs, test_docs, validation_docs = get_named_documents(sentiment_list, split_list, num_samples_wanted)

    models_by_name = build_doc2vec(all_docs, size_in)

    best_model, best_param = choose_best_model(models_by_name, all_docs, train_docs, test_docs, tag_decoder_list, deg_in, classifier_in, alpha_list)

    validation_testing(train_docs, test_docs, validation_docs, best_param, feature_generator="doc2vec", classifier_in = classifier_in, tag_decoder=tag_decoder_list, model= best_model)

    return best_model, best_param

# doc2vec_classifier(size_in=1001)
# doc2vec_classifier(size_in=286)
# doc2vec_classifier(num_samples_wanted=int(150000),size_in=1001)
# doc2vec_classifier(num_samples_wanted=int(150000),size_in=286)
# doc2vec_classifier(num_samples_wanted=int(100000),size_in=1001)
# doc2vec_classifier(num_samples_wanted=int(100000),size_in=286)

def prinComp_classifier(use_PCA = True, num_PCA = 10, classifier_in = "svm", num_samples_wanted = int(200000), file_name_list = ["testdata.manual.2009.06.14", "training.1600000.processed.noemoticon"], use_current_tweet_iterate = True, use_neutral_tweets = False):
    if not use_current_tweet_iterate:
        split_list, sentiment_list, tag_decoder_list = ALL_clean_dataset(file_name_list, use_neutral_tweets, num_samples_wanted)
    else:
        split_list, sentiment_list, tag_decoder_list = SOME_clean_dataset(file_name_list, use_neutral_tweets, num_samples_wanted)

    _, train_docs, test_docs, validation_docs = get_named_documents(sentiment_list, split_list, num_samples_wanted)

    train_vectors, train_labels, test_vectors, test_labels = BOW_vectors(train_docs + test_docs, validation_docs, use_PCA, num_PCA)

    test_acc, train_acc, param = find_error_rate(train_vectors,train_labels,test_vectors,test_labels, classifier_in)

    print "The testing accuracy was %f \n The training accuracy was %f \n Both used parameter %f" % (test_acc, train_acc, param)

#prinComp_classifier(use_PCA=False)
#prinComp_classifier(use_PCA=False,num_samples_wanted=int(150000))
#prinComp_classifier(use_PCA=False,num_samples_wanted=int(100000))
#prinComp_classifier(num_PCA=500)
prinComp_classifier(num_PCA=100,num_samples_wanted=int(150000))
prinComp_classifier(num_PCA=100,num_samples_wanted=int(100000))
#prinComp_classifier(num_PCA=1000)
prinComp_classifier(num_PCA=200,num_samples_wanted=int(150000))
prinComp_classifier(num_PCA=200,num_samples_wanted=int(100000))


def doc2vec_classifier_half_gaussian(alpha_list= "none", classifier_in = "svm", num_samples_wanted = int(200000), file_name_list = ["testdata.manual.2009.06.14", "training.1600000.processed.noemoticon"], use_current_tweet_iterate = True, use_neutral_tweets = False, size_in = 1000, deg_in = -1):

    if not use_current_tweet_iterate:
        split_list, sentiment_list, tag_decoder_list = ALL_clean_dataset(file_name_list, use_neutral_tweets, num_samples_wanted)
    else:
        split_list, sentiment_list, tag_decoder_list = SOME_clean_dataset(file_name_list, use_neutral_tweets, num_samples_wanted)

    all_docs, train_docs, test_docs, validation_docs = get_named_documents(sentiment_list, split_list, num_samples_wanted)

    models_by_name = build_doc2vec(all_docs, size_in)

    best_model, best_param = choose_best_model(models_by_name, all_docs, train_docs, test_docs, tag_decoder_list, deg_in, classifier_in, alpha_list)

    validation_testing(train_docs, test_docs, validation_docs, best_param, feature_generator="doc2vec", classifier_in = classifier_in, tag_decoder=tag_decoder_list, model= best_model)

    return best_model, best_param

# doc2vec_classifier_half_gaussian(num_samples_wanted=int(100000),size_in=10,deg_in=3)
# doc2vec_classifier_half_gaussian(num_samples_wanted=int(100000),size_in=10,deg_in=4)
# doc2vec_classifier_half_gaussian(num_samples_wanted=int(150000),size_in=10,deg_in=3)
# doc2vec_classifier_half_gaussian(num_samples_wanted=int(150000),size_in=10,deg_in=4)
# doc2vec_classifier_half_gaussian(size_in=10,deg_in=3)
# doc2vec_classifier_half_gaussian(size_in=10,deg_in=4)