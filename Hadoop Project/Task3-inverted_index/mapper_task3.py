#! /usr/bin/env python3

import sys
import string
import nltk
import os
import ntpath
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
# initialize lemmatizer
lem = WordNetLemmatizer()


# processing the data which involves removing stop words and also lemmatizing that is only having root words..
def process_text(line):
    tokenized_words = word_tokenize(line)
    processed_words = []
    for word in tokenized_words:
        if word not in stop_words:
            word = word.translate(word.maketrans('', '', string.punctuation))  # removing punctuations
            if word is not '':
                # this lines make sure we lemmatize it , i.e., only taking root words
                processed_words.append(lem.lemmatize(word))
    return processed_words


# function to read data line by line from file and process it
def read_input(file):
    for line in file:
        processed_line = process_text(line.lower())
        yield processed_line


def main(separator='\t'):
    # reading file name
    file_name = os.getenv('map_input_file')
    file_name = ntpath.basename(file_name)
    # reading the data by calling read_input
    data = read_input(sys.stdin)
    for words in data:
        for word in words:
            print('%s%s%s' % (word, separator, file_name))


if __name__ == "__main__":
    main()
