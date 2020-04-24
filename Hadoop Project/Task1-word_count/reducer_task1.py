#! /usr/bin/env python3

import sys
from itertools import groupby
from operator import itemgetter


def read_mapper_output(file, separator='\t'):
    for line in file:
        # read a line strip spaces at the end and split the line into words based on separator
        word = line.strip().split(separator, 1)
        yield word


def main(separator='\t'):
    # read the complete data
    data = read_mapper_output(sys.stdin, separator)
    # group the data based on first value(word)
    for current_word, group in groupby(data, itemgetter(0)):
        try:
            # calculate the total count of words
            total_count = sum(int(count) for current_word, count in group)
            print("%s%s%d" % (current_word, separator, total_count))
        except ValueError:
            pass


if __name__ == "__main__":
    main()
