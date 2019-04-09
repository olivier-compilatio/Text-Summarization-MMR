import os
import re
import sys
import json
from termcolor import colored
from string import punctuation
import argparse
from ntpath import split as ntsplit
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.stem.snowball import FrenchStemmer, SnowballStemmer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import operator


# def getVectorSpace(cleanSet):
#     vocab = {}
#     for data in cleanSet:
#         for word in data.split():
#             print("woard", word)
#             vocab[data] = 0
#     return vocab.keys()


def calculateSimilarity(sentence, doc):
    if doc == []:
        return 0
    vocab = {}
    for word in sentence.split(" "):
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


def load_stopwords(stopwords_file, lang):
    with open(stopwords_file) as f:
        return json.load(f)[lang]


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


def stemmize(stemmer, sent, stopwords):
    clean_sent = []
    for word in sent:
        stem = stemmer.stem(word)
        if word not in stopwords and word not in punctuation:
            clean_sent.append(stem)
    return clean_sent


def load_sentences(text_file, stopwords, lang):
    path, f = ntsplit(text_file)
    reader = PlaintextCorpusReader(path, f)
    sentences = [sent for sent in reader.sents()]
    clean = []
    originalSentenceOf = {}
    if lang == "fr":
        stemmer = FrenchStemmer()
    elif lang == "en":
        stemmer = SnowballStemmer("english")
    # Data cleansing
    for sent in sentences:
        s = stemmize(stemmer, sent, stopwords)
        clean.append(" ".join(s))
        originalSentenceOf[clean[-1]] = sent
    setClean = set(clean)
    return setClean, originalSentenceOf, sentences, clean


def load_data(stopwords_file, text_file, lang):
    stopwords = load_stopwords(stopwords_file, lang)
    setClean, originalSentenceOf, sentences, clean = load_sentences(
        text_file, stopwords, lang
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


def summarize_mmr(scores, sentences, percent):
    # calculate MMR
    n = percent * len(sentences) / 100
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


def run_mmr(stopwords_file, text_file, lang):
    stopwords, setClean, originalSentenceOf, sentences, clean = load_data(
        stopwords_file, text_file, lang
    )
    scores = compute_similarity_scores(clean, setClean)
    summary = summarize_mmr(scores, sentences, percent)
    show_results(summary, originalSentenceOf, clean)


def main():
    def parse_arguments():
        parser = argparse.ArgumentParser(description="simple mmr for French")
        parser.add_argument(
            "-t", "--text", help="file to summarize", default="sample.txt"
        )
        parser.add_argument(
            "-s",
            "--stopwords",
            help="json file containing stopwords for multiple languages ",
            default="stopwords.json",
        )
        parser.add_argument(
            "-p",
            "--percent",
            type=int,
            help="percentage of sentences to use for summary",
            default=20,
        )
        parser.add_argument(
            "-l", "--lang", type=str, help="language of the input text", default="fr"
        )

        args = parser.parse_args()

        if args.lang.lower() in ("french", "fr"):
            args.lang = "fr"
        elif args.lang.lower() in ("english", "en"):
            args.lang = "en"
        else:
            raise Exception("Unknown lang ", args.lang)

        return args

    args = parse_arguments()
    stopwords_file = args.stopwords
    text_file = args.text
    percent = args.percent

    lang = args.lang
    print("text file :", text_file, file=sys.stderr)
    print("stopwords file", stopwords_file, file=sys.stderr, end="\n\n\n")

    run_mmr(stopwords_file, text_file, lang)


if __name__ == "__main__":
    main()
