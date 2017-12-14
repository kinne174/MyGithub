from clean_data import ALL_clean_dataset
from build_named_documents import build_named_documents
from initializing_doc2vec import build_doc2vec
from build_doc2vec_vectors import doc2vec_vectors
from build_BOW_vectors import BOW_vectors
from writing_csv import write_csv
import numpy as np
import datetime


def output_doc2vec_vectors(file_name, num_samples_wanted, doc_size, which_model):
    #which_model == 0 for DM or == 1 for BOW
    print "STARTING at %s" % str(datetime.datetime.now())


    sentiment_score_list = ALL_clean_dataset(file_name, num_samples_wanted)

    classifying_docs = build_named_documents(sentiment_score_list, num_samples_wanted)

    trained_model = build_doc2vec(classifying_docs, doc_size, which_model)

    classifying_doc2vec_vectors, classifying_doc2vec_labels = doc2vec_vectors(trained_model, classifying_docs, scale=False)

    classifying_BOW_vectors, classifying_BOW_labels = BOW_vectors(classifying_docs,do_PCA=False,num_PCA=50)

    write_csv(np.hstack((np.array(classifying_doc2vec_vectors), np.array([classifying_doc2vec_labels]).T)), classifying_BOW_vectors, np.array([classifying_BOW_labels]).T, num_samples_wanted, doc_size, which_model)

    print "ENDING at %s" % str(datetime.datetime.now())




output_doc2vec_vectors(["training.1600000.processed.noemoticon"], num_samples_wanted=200000,doc_size=250, which_model=0)
output_doc2vec_vectors(["training.1600000.processed.noemoticon"], num_samples_wanted=200000,doc_size=150, which_model=0)
#output_doc2vec_vectors(["training.1600000.processed.noemoticon"], num_samples_wanted=100000,doc_size=50, which_model=0)
# output_doc2vec_vectors(["training.1600000.processed.noemoticon"], num_samples_wanted=50000,doc_size=250, which_model=1)
# output_doc2vec_vectors(["training.1600000.processed.noemoticon"], num_samples_wanted=50000,doc_size=1000, which_model=0)
# output_doc2vec_vectors(["training.1600000.processed.noemoticon"], num_samples_wanted=50000,doc_size=1000, which_model=1)
# output_doc2vec_vectors(["training.1600000.processed.noemoticon"], num_samples_wanted=100000,doc_size=250, which_model=0)
# output_doc2vec_vectors(["training.1600000.processed.noemoticon"], num_samples_wanted=100000,doc_size=250, which_model=1)
# output_doc2vec_vectors(["training.1600000.processed.noemoticon"], num_samples_wanted=100000,doc_size=1000, which_model=0)
# output_doc2vec_vectors(["training.1600000.processed.noemoticon"], num_samples_wanted=100000,doc_size=1000, which_model=1)





