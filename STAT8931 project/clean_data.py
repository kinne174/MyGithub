#Cleaning the tweets

import csv
import os
import smart_open
import locale
from replace_one_words import replace_one_words
import time

locale.setlocale(locale.LC_ALL, 'C')

directory = "Data"

#function to reduce and replace unwanted characters
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

#function to run through tweets and save clean tweets to external file
def ALL_clean_dataset(file_name_list, num_samples_wanted):

    start = time.clock()

    mod_val = (1600000 / num_samples_wanted) + 2

    classifier_iterator = 0

    classifying_sentences_string = u''

    sentiment_score_list = [0] * num_samples_wanted

    for file_no, file_name in enumerate(file_name_list):
        with smart_open.smart_open(os.path.join(directory, file_name + ".csv"), 'rb') as csvfile:
            csvreader = csv.reader(csvfile, dialect=csv.excel)
            for line_no, line in enumerate(csvreader):
                if line_no % mod_val == 0:
                    tweet_sentiment = int(line[0])
                    if not tweet_sentiment == 2:
                        #add to doc2vec_string
                        query_word = line[3]
                        dirty_tweet = line[5]
                        clean_tweet = clean_line(dirty_tweet, query_word)

                        classifying_sentences_string += clean_tweet
                        sentiment_score = (tweet_sentiment/4) + 1
                        sentiment_score_list[classifier_iterator] = sentiment_score

                        classifier_iterator += 1

                    if classifier_iterator % 10000 == 0:
                        print classifier_iterator

                    if classifier_iterator >= num_samples_wanted:
                        break

    classifying_splitlines = classifying_sentences_string.splitlines()

    new_classifying_sentences_string = replace_one_words(classifying_splitlines)

    with smart_open.smart_open(os.path.join(directory,"all-classifyingSentences-data-{0}.txt".format(num_samples_wanted)), "wb") as open_file:
        for line_no, line in enumerate(new_classifying_sentences_string.splitlines()):
            new_line = u'*{0} {1}\n'.format(line_no, line)
            open_file.write(new_line.encode("utf-8"))

    end = time.clock()

    print "Cleaning the data using %d samples took %f seconds" % (num_samples_wanted, end - start)

    return sentiment_score_list