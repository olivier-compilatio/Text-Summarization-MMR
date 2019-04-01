import os
import re
import sys
import json
from termcolor import colored
from string import punctuation

from ntpath import split as ntsplit
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.stem.snowball import FrenchStemmer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import operator


def getVectorSpace(cleanSet):
    vocab = {}
    for data in cleanSet:
        for word in data.split():
            vocab[data] = 0
    return vocab.keys()


def calculateSimilarity(sentence, doc):
    if doc == []:
        return 0
    vocab = {}
    for word in sentence:
        vocab[word] = 0

    docInOneSentence = ""
    for t in doc:
        docInOneSentence += t + " "
        for word in t.split():
            vocab[word] = 0

    cv = CountVectorizer(vocabulary=vocab.keys())

    docVector = cv.fit_transform([docInOneSentence])
    sentenceVector = cv.fit_transform([sentence])
    return cosine_similarity(docVector, sentenceVector)[0][0]


def load_stopwords(stopwords_file):
    with open(stopwords_file) as f:
        return json.load(f)["fr"]


def cleanData(sentence):
    # remove stopwords
    # sentence = re.sub('[^A-Za-z0-9 ]+', '', sentence)
    # sentence filter(None, re.split("[.!?", setence))
    ret = []
    sentence = stemmer.stem(sentence)
    for word in sentence.split():
        if not word in stopwords:
            ret.append(word)
    return " ".join(ret)


def stemmize(fs, sent, stopwords):
    clean_sent = []
    for word in sent:
        stem = fs.stem(word)
        if word not in stopwords and word not in punctuation:
            clean_sent.append(stem)
    return clean_sent


def load_sentences(text_file, stopwords):
    path, f = ntsplit(text_file)
    reader = PlaintextCorpusReader(path, f)
    sentences = [sent for sent in reader.sents()]
    clean = []
    originalSentenceOf = {}
    fs = FrenchStemmer()
    # Data cleansing
    for sent in sentences:
        s = stemmize(fs, sent, stopwords)
        clean.append(" ".join(s))
        originalSentenceOf[clean[-1]] = sent
    setClean = set(clean)
    print(clean)

    return setClean, originalSentenceOf, sentences, clean


def load_data(stopwords_file, text_file):
    stopwords = load_stopwords(stopwords_file)
    setClean, originalSentenceOf, sentences, clean = load_sentences(
        text_file, stopwords
    )

    return stopwords, setClean, originalSentenceOf, sentences, clean


def compute_similarity_scores(clean, setClean):
    # calculate Similarity score each sentence with whole documents
    scores = {}
    for data in clean:
        temp_doc = setClean - set([data])
        score = calculateSimilarity(data, list(temp_doc))
        scores[data] = score
    return scores


def summarize_mmr(scores, sentences):
    # calculate MMR
    n = 30 * len(sentences) / 100
    alpha = 0.5
    summarySet = []
    while n > 0:
        mmr = {}
        for sentence in scores.keys():
            if not sentence in summarySet:
                mmr[sentence] = alpha * scores[sentence] - (
                    1 - alpha
                ) * calculateSimilarity(sentence, summarySet)
        selected = max(mmr.items(), key=operator.itemgetter(1))[0]
        summarySet.append(selected)
        n -= 1
    return summarySet


def show_results(summarySet, originalSentenceOf, clean):

    print("\nSummary:\n")
    for sentence in summarySet:
        print(" ".join(originalSentenceOf[sentence]))
    print()

    print("=============================================================")
    print("\nOriginal Passages:\n")
    for sentence in clean:
        if sentence in summarySet:
            print(colored(" ".join(originalSentenceOf[sentence]), "red"))
        else:
            print(" ".join(originalSentenceOf[sentence]))


def main():
    stopwords_file = "stopwords.json"
    text_file = "sample.txt"
    stopwords, setClean, originalSentenceOf, sentences, clean = load_data(
        stopwords_file, text_file
    )
    scores = compute_similarity_scores(clean, setClean)
    summary = summarize_mmr(scores, sentences)
    show_results(summary, originalSentenceOf, clean)


if __name__ == "__main__":
    main()
