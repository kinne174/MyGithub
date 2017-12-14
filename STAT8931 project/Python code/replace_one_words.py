from sklearn.feature_extraction import text as fet
from nltk.tokenize import TreebankWordTokenizer
import numpy as np

#def replace_one_words(train_sentences, test_sentences, validation_sentences):
def replace_one_words(sentences):

    #all_new_list = train_sentences + test_sentences + validation_sentences
    all_new_list = sentences

    count_vectorizer = fet.CountVectorizer(analyzer="word", lowercase=False, binary=True, stop_words=None, tokenizer=TreebankWordTokenizer().tokenize)

    X = count_vectorizer.fit_transform(all_new_list)
    feature_names = count_vectorizer.get_feature_names()

    word_counts = np.array(X.sum(0))[0]
    one_indices = np.where(word_counts == 1)[0]
    one_words = [feature_names[ind] for ind in one_indices]

    for ind in range(len(one_indices)):
        one_word_column = X[:,one_indices[ind]]
        one_word_sentence_ind = one_word_column.nonzero()[0].tolist()[0]
        all_new_list[one_word_sentence_ind] = all_new_list[one_word_sentence_ind].replace(" " + one_words[ind] + " ", " UNKNOWN_WORD ")

    return "\n".join(all_new_list)