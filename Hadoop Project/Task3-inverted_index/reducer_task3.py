#! /usr/bin/env python3

from itertools import groupby
from operator import itemgetter
import sys


def read_mapper_output(file, separator='\t'):
    for line in file:
        yield line.strip().split(separator, 1)


def main(separator='\t'):
    data = read_mapper_output(sys.stdin, separator)
    for current_word, group in groupby(data, itemgetter(0)):
        try:
            file_list = set([file_name for current_word, file_name in group])
            print("%s%s%s" % (current_word, separator, file_list))
        except ValueError:
            pass


if __name__ == "__main__":
    main()