from gensim.models import Doc2Vec
import gensim.models.doc2vec
import multiprocessing
from collections import OrderedDict
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

#which doc2vec models the classifier should train on
#unfortunately not optimized right now to allow full control from function line
#has to be manually edited within the code to get different models
def build_doc2vec(all_docs, size_in):
    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

    models_list = [
        # Doc2Vec(dm = 1, dm_mean= 1, size = 400, window = 10, negative = 11, hs = 0, min_count= 1, workers= cores),
        #Doc2Vec(dm=1, dm_mean=1, size=500, window=10, negative=11, hs=0, min_count=1, workers=cores),
        Doc2Vec(dm=1, dm_mean=1, size=size_in, window=3, negative=11, hs=0, min_count=1, workers=cores),
        # Doc2Vec(dm=0, size=400, negative=11, hs=0, min_count=1, workers=cores),
        #Doc2Vec(dm=0, size=500, negative=11, hs=0, min_count=1, workers=cores),
        Doc2Vec(dm=0, size=500, negative=11, hs=0, min_count=1, workers=cores)
    ]

    # classifying_docs should be in console before running this
    models_list[0].build_vocab(all_docs)
    print models_list[0]
    for model in models_list[1:]:
        model.reset_from(models_list[0])
        print model

    models_by_name = OrderedDict((str(model), model) for model in models_list)

    models_by_name['dbow+dmm_500']= ConcatenatedDoc2Vec([models_list[0], models_list[1]])
    #models_by_name['dbow+dmm_500'] = ConcatenatedDoc2Vec([models_list[1], models_list[4]])
    #models_by_name['dbow+dmm_600'] = ConcatenatedDoc2Vec([models_list[2], models_list[5]])

    return models_by_name