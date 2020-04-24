#! /usr/bin/env python3

import sys
from itertools import groupby
from operator import itemgetter


def read_mapper_output(file, separator='\t'):
    for line in file:
        yield line.strip().split(separator)


def main(separator='\t'):
    k = 3
    data = read_mapper_output(sys.stdin, separator)
    # grouping the data by keys which is row numbers, This will give my number of mappers*k points
    for index1, group in groupby(data, itemgetter(0)):
        labels = {}                                        # dictionary to maintain count of labels
        try:
            result = []
            for index, distance, label in group:
                result.append((distance, label))
                labels[label] = 0                   # initializing every labels count as zero
            prediction = sorted(result)            # sorting the list based on distances
            max1 = labels[prediction[0][1]]

            ans = prediction[0][1]
            for i in range(k):    # having the only first k points

                labels[prediction[i][1]] = labels[prediction[i][1]] + 1  # incrementing the label count if it occurs

                if max1 < labels[prediction[i][1]]:     # checking if it exceeded the highest frequency label
                    max1 = labels[prediction[i][1]]     # if yest then set max1 to the frequency of the highest
                    # frequency label
                    ans = prediction[i][1]              # and setting ans to the label with highest frequency
            # printing ans after for loop is completed which prints key as test row number and value as prediction
            print("%d\t%d" % (int(index1), int(ans)))
        except ValueError:
            pass


if __name__ == "__main__":
    main()
