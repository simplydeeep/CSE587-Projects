#! /usr/bin/env python3

import sys
from itertools import groupby
from operator import itemgetter


def read_mapper_output(file, separator='\t'):
    for line in file:
        # read a line strip spaces at the end and split the line into words based on separator
        yield line.strip().split(separator)


def main(separator='\t'):
    result = {}
    # read the complete data
    data = read_mapper_output(sys.stdin, separator)
    # group the data based on first value(word-dummy key in this case)
    for current_word, group in groupby(data, itemgetter(1)):
        try:
            total_count = sum(int(count) for key, current_word, count in group)
            result[current_word] = total_count
        except ValueError:
            pass

    if result:
        # sort the key value pair based on values and print the first 10
        result_sorted = sorted(result.items(), key=lambda item: item[1], reverse=True)
        for index, item in enumerate(result_sorted):
            print("%s%s%d" % (item[0], separator, item[1]))
            if index == 9:
                break


if __name__ == "__main__":
    main()
