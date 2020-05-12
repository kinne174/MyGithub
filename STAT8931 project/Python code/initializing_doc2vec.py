from gensim.models import Doc2Vec
import gensim.models.doc2vec
import multiprocessing
from random import shuffle

#which doc2vec models the classifier should train on
#unfortunately not optimized right now to allow full control from function line
#has to be manually edited within the code to get different models
def build_doc2vec(all_docs, doc_size, which_model):
    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

    models_list = [
        # Doc2Vec(dm = 1, dm_mean= 1, size = 400, window = 10, negative = 11, hs = 0, min_count= 1, workers= cores),
        #Doc2Vec(dm=1, dm_mean=1, size=500, window=10, negative=11, hs=0, min_count=1, workers=cores),
        Doc2Vec(dm=1, dm_mean=1, size=doc_size, window=3, negative=11, hs=0, min_count=1, workers=cores),
        # Doc2Vec(dm=0, size=400, negative=11, hs=0, min_count=1, workers=cores),
        #Doc2Vec(dm=0, size=500, negative=11, hs=0, min_count=1, workers=cores),
        Doc2Vec(dm=0, size=doc_size, negative=11, hs=0, min_count=1, workers=cores)
    ]

    # classifying_docs should be in console before running this
    models_list[0].build_vocab(all_docs)
    print models_list[0]
    for model in models_list[1:]:
        model.reset_from(models_list[0])
        print model

    passes = 20
    alpha_start = 0.025
    alpha_min = 0.001
    alpha_delta = (alpha_start - alpha_min)/passes

    all_docs_copy = all_docs[:]

    alpha = alpha_start

    for model in models_list:
        for epoch in range(passes):
            shuffle(all_docs_copy)
            model.alpha, model.min_alpha = alpha, alpha
            model.train(all_docs_copy, total_examples = len(all_docs_copy), epochs = 1)

            alpha -= alpha_delta

    #which_model = 0 #0 for distributed memory, 1 for BOW

    return models_list[which_model]