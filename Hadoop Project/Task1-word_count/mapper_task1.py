#! /usr/bin/env python3

import sys
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

# store stop words in english
stop_words = set(stopwords.words('english'))
# initialize lemmatizer
lem = WordNetLemmatizer()

def process_text(line):
    # tokenize the words using nltk word_tokenize
    tokenized_words = word_tokenize(line)
    processed_words = []

    for word in tokenized_words:
        # remove stop words using nltk
        if word not in stop_words:
            # remove punctuations in a word
            word = word.translate(word.maketrans('', '', string.punctuation))
            # remove empty strings if present
            if word is not '':
                processed_words.append(lem.lemmatize(word))

    return processed_words


def read_input(file):
    for line in file:
        yield process_text(line.lower())


def main(separator='\t'):
    # read and store input after processing
    data = read_input(sys.stdin)
    for words in data:
        for word in words:
            print("%s%s%d" % (word, separator, 1))


if __name__ == "__main__":
    main()
