from collections import namedtuple
import gensim
import io

#organize classifying documents to collect all information about the individual tweets
def get_named_documents(sentiment_score_list, split_list, num_samples_wanted):

    classifying_documents = []

    SentimentDocument = namedtuple("SentimentDocument", "words tags sentiment split")

    with io.open("all-classifyingSentences-data-{0}.txt".format(num_samples_wanted), encoding="utf-8") as allClassifyingSentences:
        for line_no, line in enumerate(allClassifyingSentences):
            tokens = gensim.utils.to_unicode(line).split()
            words = tokens[1:]
            tags = [line_no]
            # sentiment_list has to be in console for now, will do better when I get a main()
            sentiment = sentiment_score_list[line_no]
            split = split_list[line_no]
            classifying_documents.append(SentimentDocument(words, tags, sentiment, split))

    train_docs = [doc for doc in classifying_documents if doc.split == "train"]
    test_docs = [doc for doc in classifying_documents if doc.split == "test"]
    validation_docs = [doc for doc in classifying_documents if doc.split == "validation"]

    print " There are %i total documents in classification. \n There are %i documents in train_docs. \n There are %i documents in test_docs. \n There are %i documents in validation_docs. \n The split is %.2f %.2f %.2f" % (
    len(classifying_documents), len(train_docs), len(test_docs), len(validation_docs),
    float(len(train_docs)) / len(classifying_documents), float(len(test_docs)) / len(classifying_documents),
    float(len(validation_docs)) / len(classifying_documents))

    sentiment_split_train = float(sum([(doc.sentiment + 1)/2 for doc in train_docs]))/len(train_docs)
    print "The sentiment split in training is %f" % sentiment_split_train

    sentiment_split_test = float(sum([(doc.sentiment + 1)/2 for doc in test_docs]))/len(test_docs)
    print "The sentiment split in testing is %f" % sentiment_split_test

    Doc2VecDocument = namedtuple("Doc2VecDocument", "words tags")

    all_docs = []

    with io.open("all-doc2vec-data-{0}.txt".format(num_samples_wanted), encoding="utf-8") as allDoc2VecSentences:
        for line_no, line in enumerate(allDoc2VecSentences):
            tokens = gensim.utils.to_unicode(line).split()
            words = tokens[1:]
            tags = [line_no]
            all_docs.append(Doc2VecDocument(words, tags))

    print " There are %i total docuemnts that will be used in model building." % (len(all_docs))

    #all_docs_copy = all_docs[:]

    return all_docs, train_docs, test_docs, validation_docs

