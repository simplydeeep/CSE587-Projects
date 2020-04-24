#! /usr/bin/env python3

import sys
import string


def read_input(file, separator='\t'):
    for line in file:
        # read a line strip spaces at the end and split the line into words
        yield line.strip().split(separator)


def main(separator='\t'):
    # read input and process them
    line = read_input(sys.stdin, separator)
    for data in line:
        print("key\t%s\t%s" % (data[0], data[1]))


if __name__ == "__main__":
    main()
