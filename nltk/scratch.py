#!/usr/bin/env python

# import libraries
import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union
from nltk.corpus import wordnet
from nltk.corpus import movie_reviews
import random
import pickle

import sys # for argument vector
import argparse # to parse arguments
import re
#from BeautifulSoup import BeautifulSoup # to force to unicode
from BeautifulSoup import BeautifulSoup

# describe what the script does
what_i_do = "Examples of natural language processing with nltk"

# initialize the parser
parser = argparse.ArgumentParser(description=what_i_do)
parser.add_argument("-i", "--input_files", type=str, dest="input_files", default=None,
                   required=True, nargs='*', help='file to print out to the screen [Required]')
parser.add_argument("-o", "--output_file", type=str, dest="output_file", default=None,
                   required=False, nargs=1, help='file to print out to the screen [Required]')

sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')

def getGutenbrugLines(lines):
    # scan lines and pull out relevant lines from a pg file
    pg_lines = []
    in_pg = False
    for l in lines:
        l = l.strip()
        if "*** START OF THIS PROJECT GUTENBERG" in l:
            in_pg = True
            continue
        elif "*** END OF THIS PROJECT GUTENBERG" in l:
            in_pg = False
            continue

        if l == "":
            continue
        if in_pg:
            pg_lines.append(l)

    return pg_lines

def word_tokenizer(sents_text, pattern = r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*"):
    # could also use custom trained tokenizer
    # custom_sent_tokenizer = PunktsentenceTokenizer(train_text)
    return nltk.regexp_tokenize(sents_text, pattern)


def process_input(lines):
    # get the text
    raw_text = " ".join(getGutenbrugLines(lines))

    # turn to unicode (dammit)
    soup = BeautifulSoup(raw_text)
    raw_text = str(soup)
    raw_text = raw_text.decode('utf-8')

    # isolate sentenses
    sents_text = " ".join(sent_tokenizer.tokenize(raw_text))

    # extract the tokens
    tokens = word_tokenizer(sents_text)

    # convert to lowercase and stem
    wnl = nltk.WordNetLemmatizer()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';',
                      '(', ')', '[', ']', '{', '}', '--'])

    # words = [porter.stem(w.lower()) for w in tokens]
    words = [wnl.lemmatize(w.lower()) for w in tokens if w.lower() not in stop_words]

    return words

def process_content():

    train_text = state_union.raw("2005-GWBush.txt")
    sample_text = state_union.raw("2006-GWBush.txt")

    custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
    tokenized = custom_sent_tokenizer.tokenize(sample_text)

    try:
        for i in tokenized:
            words  = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            # namedEnt = nltk.ne_chunk(tagged, binary=True)
            # print namedEnt

            # any form of an adverb RB
            # chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            # chunkGram = r"""Chunk: {<.*>+}
            #                         }<VB.?|IN|DT|TO>{"""
            #
            # chunkParser = nltk.RegexpParser(chunkGram)
            # chunked = chunkParser.parse(tagged)
            # chunked.draw()

    except Exception as e:
        print str(e)

def demo_wordnet():
    syno_list = []
    anto_list = []
    for syn in wordnet.synsets("good"):
        for l in syn.lemmas():
            syno_list.append(l.name())
            if l.antonyms():
                anto_list.append(l.antonyms()[0].name())

    print set(syno_list)
    print set(anto_list)

    # semantic similarity
    w1 = wordnet.synset("ship.n.01")
    w2 = wordnet.synset("boat.n.01")
    print w1.wup_similarity(w2)

    # semantic similarity
    w1 = wordnet.synset("ship.n.01")
    w2 = wordnet.synset("car.n.01")
    print w1.wup_similarity(w2)

    # semantic similarity
    w1 = wordnet.synset("ship.n.01")
    w2 = wordnet.synset("cat.n.01")
    print w1.wup_similarity(w2)

def find_features(document, word_features):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words) # one-hot encoding

    return features

def document_class():
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            document_category_pair = ( list(movie_reviews.words(fileid)), category )
            documents.append(document_category_pair)

    random.shuffle(documents)

    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)
    # print all_words.most_common(15)
    # print all_words["stupid"]
    word_features = list(all_words.keys())[:3000]
    # print find_features(movie_reviews.words('neg/cv000_29416.txt'), word_features)
    featuresets = [(find_features(rev, word_features), category) for (rev, category) in documents]

    training_set = featuresets[:1900]
    testing_set = featuresets[1900:]

    # posterior = prior occurences x liklihood / evidence
    # classifier = nltk.NaiveBayesClassifier.train(training_set)
    classifier_f = open("naivebayes.pickle", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    print "Naive Bayes Algo Accuracy:", nltk.classify.accuracy(classifier, testing_set)
    print classifier.show_most_informative_features(15)
    # save_classifier_fh = open("naivebayes.pickle", "wb")
    # pickle.dump(classifier, save_classifier_fh)
    # save_classifier_fh.close()


# the main function of the script
def main():
    args = vars(parser.parse_args())
    # process_content()
    # demo_wordnet()
    document_class()

    # for f in args["input_files"]:
    #     # read lines
    #     print f
    #     fh = open(f, 'r')
    #     lines = fh.readlines()
    #     fh.close()
    #
    #     words = process_input(lines)
    #
    #     print nltk.pos_tag(words[1:200])
        # vocab = sorted(set(words))
        # print f, ":", len(vocab)

if __name__ == "__main__":
    main()
