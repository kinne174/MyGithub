import csv
import os
import smart_open
import numpy as np
import locale
import io
from replace_one_words import replace_one_words
import time

locale.setlocale(locale.LC_ALL, 'C')

directory = "Data"

#function to take a tweet and replace unwanted characters and lowercase letters
def clean_line(line, query_term):
    line += " "

    line = line.lower()

    url_char = "http"
    if url_char in line:
        url_index = line.index(url_char)
        url_first_space_after = line[url_index:].index(" ") + url_index
        url_text_to_replace = line[url_index:url_first_space_after]
        line = line.replace(url_text_to_replace, "URL")

    BOM_chars = ["\xef\xbf\xbd", "\xcf\x9b"]
    for bc in BOM_chars:
        line = line.replace(bc, " ")

    unique_letters = "".join(set(line))
    for uc in unique_letters:
        while uc + uc + uc in line:
            line = line.replace(uc + uc + uc, uc + uc)

    html_code = ["&quot;", '&amp;', '&lt;', '&gt;']
    in_ascii = ['"', '&', '<', '>']
    for hc_ind in range(len(html_code)):
        line = line.replace(html_code[hc_ind], in_ascii[hc_ind])

    replace_emoticons = [":)", ";)", ":(", ";(", ":')", ":'(", ";')", ";'(", ":-)", ":-("]
    for re in replace_emoticons:
        line = line.replace(re, "")

    space_chars = ['"', ',', '/', '_', "#", "[", "]", "&", "*", "-"]
    for sc in space_chars:
        line = line.replace(sc, " " + sc + " ")

    new_space_chars = [".", "'", ";", ":", "(", ")", "!", "?"]
    for nsc in new_space_chars:
        if nsc in line:
            for i in range(line.count(nsc)):
                nsc_index = [ind for ind, ltr in enumerate(line) if ltr == nsc]
                if not (line[nsc_index[i] + 1] == nsc or line[nsc_index[i] - 1] == nsc):
                    if not line[nsc_index[i] - 1] == " ":
                        line = line[:nsc_index[i]] + " " + line[nsc_index[i]] + " " + line[(nsc_index[i]+1):]
                elif line[nsc_index[i] + 1] == nsc:
                    if not line[nsc_index[i] - 1] == " ":
                        line = line[:(nsc_index[i])] + " " + line[(nsc_index[i]):]
                elif line[nsc_index[i] - 1] == nsc:
                    if not line[nsc_index[i] + 1] == " ":
                        line = line[:(nsc_index[i] + 1)] + " " + line[(nsc_index[i] + 1):]

    while "@" in line:
        at_index = line.index("@")
        at_first_space_after = line[at_index:].index(" ") + at_index
        at_text_to_replace = line[at_index:at_first_space_after]
        if at_text_to_replace[1:] == query_term:
            line = line.replace(at_text_to_replace, "QUERY_TERM")
        else:
            line = line.replace(at_text_to_replace, "USERNAME")

    line = line.replace(query_term, "QUERY_TERM")

    try:
        line = line.decode('utf-8')
    except:
        line = line.decode('latin-1')

    line = u' '.join(line.split()).strip()

    line += '\n'

    return line

# def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
#     # csv.py doesn't do Unicode; encode temporarily as UTF-8:
#     csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
#                             dialect=dialect, **kwargs)
#     for row in csv_reader:
#         # decode UTF-8 back to Unicode, cell by cell:
#         yield row
#
# def utf_8_encoder(unicode_csv_data):
#     for line in unicode_csv_data:
#         yield line.encode('utf-8')

def ALL_clean_dataset(file_name_list = file_name_list, use_neutral_tweets = False, num_samples_wanted = int(166667)):

    start = time.clock()

    np.random.seed(8363)
    mod_val = (1600000 / num_samples_wanted) + 2

    classifier_iterator = 0

    doc2vec_string = u''
    classifying_sentences_string = u''

    sentiment_score_list = [0] * num_samples_wanted
    classifiers_to_word2vec_tags = [0] * num_samples_wanted
    split_list = [0] * num_samples_wanted
    len_doc2vec = 0

    for file_no, file_name in enumerate(file_name_list):
        #with io.open(os.path.join(directory, file_name + ".csv"), encoding='utf-8') as csvfile:
        with smart_open.smart_open(os.path.join(directory, file_name + ".csv"), 'rb') as csvfile:
            csvreader = csv.reader(csvfile, dialect=csv.excel)
            #csvreader = unicode_csv_reader(csvfile)
            for line_no, line in enumerate(csvreader):
                if file_no == 0 or line_no % mod_val == 0:
                    tweet_sentiment = int(line[0])
                    if not tweet_sentiment == 2 or use_neutral_tweets:
                        #add to doc2vec_string
                        tweet_id = line[1]
                        query_word = line[3]
                        dirty_tweet = line[5]
                        clean_tweet = clean_line(dirty_tweet, query_word)

                        doc2vec_string += clean_tweet

                        if not tweet_sentiment == 2:
                            classifying_sentences_string += clean_tweet
                            sentiment_score = (tweet_sentiment - 2)/2
                            #sentiment_score_list += [sentiment_score]
                            sentiment_score_list[classifier_iterator] = sentiment_score
                            #split_list += [["test", "train", "validation"][file_no*(1*(np.random.uniform(0,1) > 0.95) + 1)]]
                            #split_list[len_doc2vec] = ["test", "train", "validation"][file_no*(1*(np.random.uniform(0,1) > 0.95) + 1)]
                            split_list[classifier_iterator] = ["train","train","train","train","train", "train","train","test","test","validation"][np.random.choice(range(10))]

                            #classifiers_to_word2vec_tags += [len_doc2vec]
                            classifiers_to_word2vec_tags[classifier_iterator] = len_doc2vec

                            classifier_iterator += 1

                        len_doc2vec += 1

                    if classifier_iterator % 10000 == 0:
                        print classifier_iterator

                    if classifier_iterator >= num_samples_wanted:
                        break

    classifying_splitlines = classifying_sentences_string.splitlines()
    doc2vec_splitlines = doc2vec_string.splitlines()
    #train_sentences = [classifying_splitlines[ind] for ind in range(len(split_list)) if split_list[ind] == "train"]
    #test_sentences = [classifying_splitlines[ind] for ind in range(len(split_list)) if split_list[ind] == "test"]
    #validation_sentences = [classifying_splitlines[ind] for ind in range(len(split_list)) if split_list[ind] == "validation"]

    #new_classifying_sentences_string = replace_one_words(train_sentences, test_sentences, validation_sentences)
    new_classifying_sentences_string = replace_one_words(classifying_splitlines)
    new_doc2vec_string = replace_one_words(doc2vec_splitlines)

    #new_sentiment_score_list = [sentiment_score_list[ind] for ind in range(len(split_list)) if split_list[ind] == "train"] + [sentiment_score_list[ind] for ind in range(len(split_list)) if split_list[ind] == "test"]  + [sentiment_score_list[ind] for ind in range(len(split_list)) if split_list[ind] == "validation"]
    #new_split_list = [split for split in split_list if split == "train"] + [split for split in split_list if split == "test"] + [split for split in split_list if split == "validation"]

    with smart_open.smart_open("all-doc2vec-data-{0}.txt".format(num_samples_wanted), 'wb') as open_file:
        for line_no, line in enumerate(new_doc2vec_string.splitlines()):
            new_line = u'*{0} {1}\n'.format(line_no, line)
            open_file.write(new_line.encode("utf-8"))

    with smart_open.smart_open("all-classifyingSentences-data-{0}.txt".format(num_samples_wanted), "wb") as open_file:
        #for line_no, line in enumerate(classifying_sentences_string.splitlines()):
        for line_no, line in enumerate(new_classifying_sentences_string.splitlines()):
            new_line = u'*{0} {1}\n'.format(line_no, line)
            open_file.write(new_line.encode("utf-8"))

    end = time.clock()

    print "Cleaning the data using %d samples took %f seconds" % (num_samples_wanted, end - start)

    return split_list, sentiment_score_list, classifiers_to_word2vec_tags

'''
Just getting sentiment_score_list, classifiers_to_word2vec, split_list

'''

def SOME_clean_dataset(file_name_list = file_name_list, use_neutral_tweets = False, num_samples_wanted = int(166667)):

    start = time.clock()

    np.random.seed(8363)

    mod_val = (1600000 / num_samples_wanted) + 2

    classifier_iterator = 0

    #doc2vec_string = u''
    #classifying_sentences_string = u''

    sentiment_score_list = [0] * num_samples_wanted
    classifiers_to_word2vec_tags = [0] * num_samples_wanted
    split_list = [0] * num_samples_wanted
    len_doc2vec = 0

    for file_no, file_name in enumerate(file_name_list):
        #with io.open(os.path.join(directory, file_name + ".csv"), encoding='utf-8') as csvfile:
        with smart_open.smart_open(os.path.join(directory, file_name + ".csv"), 'rb') as csvfile:
            csvreader = csv.reader(csvfile, dialect=csv.excel)
            #csvreader = unicode_csv_reader(csvfile)
            for line_no, line in enumerate(csvreader):
                if file_no == 0 or line_no % mod_val == 0:
                    tweet_sentiment = int(line[0])
                    if not tweet_sentiment == 2 or use_neutral_tweets:
                        #add to doc2vec_string
                        #tweet_id = line[1]
                        #query_word = line[3]
                        #dirty_tweet = line[5]
                        #clean_tweet = clean_line(dirty_tweet, query_word)

                        #doc2vec_string += clean_tweet

                        if not tweet_sentiment == 2:
                            #classifying_sentences_string += clean_tweet
                            sentiment_score = (tweet_sentiment - 2)/2
                            #sentiment_score_list += [sentiment_score]
                            sentiment_score_list[classifier_iterator] = sentiment_score
                            #split_list += [["test", "train", "validation"][file_no*(1*(np.random.uniform(0,1) > 0.95) + 1)]]
                            #split_list[len_doc2vec] = ["test", "train", "validation"][file_no*(1*(np.random.uniform(0,1) > 0.95) + 1)]
                            split_list[classifier_iterator] = ["train", "train", "train", "train", "train", "train", "train", "test", "test","validation"][np.random.choice(range(10))]

                            #classifiers_to_word2vec_tags += [len_doc2vec]
                            classifiers_to_word2vec_tags[classifier_iterator] = len_doc2vec

                            classifier_iterator += 1

                        len_doc2vec += 1

                    if classifier_iterator % 10000 == 0:
                        print classifier_iterator

                    if classifier_iterator >= num_samples_wanted:
                        break

    #new_sentiment_score_list = [sentiment_score_list[ind] for ind in range(len(split_list)) if split_list[ind] == "train"] + [sentiment_score_list[ind] for ind in range(len(split_list)) if split_list[ind] == "test"]  + [sentiment_score_list[ind] for ind in range(len(split_list)) if split_list[ind] == "validation"]
    #new_split_list = [split for split in split_list if split == "train"] + [split for split in split_list if split == "test"] + [split for split in split_list if split == "validation"]

    end = time.clock()
    print "Cleaning the data using %d samples took %f seconds" % (num_samples_wanted, end - start)

    return split_list, sentiment_score_list, classifiers_to_word2vec_tags