import numpy as np
import os

import scipy.io

def write_csv(doc2vec_array, BOW_array, BOW_labels, num_samples, doc_size, which_model):
    model_names = ["DM", "BOW"]
    out_file = "C:\Users\Mitch\Documents\UofM\Fall 2017\STAT 8931\Project\Data"
    np.savetxt(os.path.join(out_file, "doc2vec_vectors_{0}_{1}_{2}.csv".format(num_samples, model_names[which_model], doc_size)), doc2vec_array, delimiter=",")
    np.savetxt(os.path.join(out_file, "BOW_labels_{0}_{1}_{2}.csv".format(num_samples, model_names[which_model], doc_size)), BOW_labels, delimiter=",")

    scipy.io.mmwrite(os.path.join(out_file, "BOW_vectors_{0}_{1}_{2}.mtx".format(num_samples, model_names[which_model], doc_size)), BOW_array)