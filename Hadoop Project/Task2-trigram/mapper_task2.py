#! /usr/bin/env python3

import sys
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem.wordnet import WordNetLemmatizer

# store stop words in english
stop_words = set(stopwords.words('english'))
key_words = ['science', 'sea', 'fire']
# initialize lemmatizer
lem = WordNetLemmatizer()


def process_text(line):
    # tokenize the words using nltk word_tokenize
    tokenized_words = word_tokenize(line)
    processed_words = []
    for word in tokenized_words:
        # remove punctuations in a word
        word = word.translate(word.maketrans('', '', string.punctuation))
        # remove empty strings if present and lemmatize non empty strings(words)
        if word is not '':
            processed_words.append(lem.lemmatize(word))

    return processed_words


def process_trigrams(tokens):
    processed_trigrams = []
    # remove the keywords and replace with $
    for index, token in enumerate(tokens):
        if token in key_words:
            tokens[index] = '$'

    # create trigrams
    trigrams = ngrams(tokens, 3)

    # add only trigrams with $ and return the list
    for trigram in trigrams:
        if trigram[0] is '$' or trigram[1] is '$' or trigram[2] is '$':
            processed_trigrams.append(trigram)

    return processed_trigrams


def read_input(file):
    processed_words = []
    for line in file:
        # process each line
        processed_line = process_text(line.lower())
        processed_words = processed_words + processed_line
    # create and process trigrams
    processed_trigrams = process_trigrams(processed_words)
    return processed_trigrams


def main(separator='\t'):
    # read and store input after processing
    data = read_input(sys.stdin)
    for trigram in data:
        print('%s_%s_%s%s%d' % (trigram[0], trigram[1], trigram[2], separator, 1))


if __name__ == "__main__":
    main()
