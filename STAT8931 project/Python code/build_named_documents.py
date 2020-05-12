from collections import namedtuple
import gensim
import io
import os

#function to collect tweets in an organized way
def build_named_documents(sentiment_score_list, num_samples_wanted):

    directory = "Data"

    classifying_documents = []

    SentimentDocument = namedtuple("SentimentDocument", "words tags sentiment")

    with io.open(os.path.join(directory,"all-classifyingSentences-data-{0}.txt".format(num_samples_wanted)), encoding="utf-8") as allClassifyingSentences:
        for line_no, line in enumerate(allClassifyingSentences):
            tokens = gensim.utils.to_unicode(line).split()
            words = tokens[1:]
            tags = [line_no]
            # sentiment_list has to be in console for now, will do better when I get a main()
            sentiment = sentiment_score_list[line_no]
            classifying_documents.append(SentimentDocument(words, tags, sentiment))

    print " There are %i total documents in classification." % (len(classifying_documents))

    sentiment_split_train = float(sum([(doc.sentiment - 1) for doc in classifying_documents]))/len(classifying_documents)
    print "The sentiment split in classification is %f" % sentiment_split_train

    return classifying_documents

