#! /usr/bin/env python3

import sys
import pandas as pd


def main():
    test_dict = {}  # having a dictionary in which key will be the test row number and value will be a list of pairs
    # which have the distances with all train data with that key ie test row number and the train row label
    # with which distance is being calculated
    k = 3  # having k value as 3
    count = 0
    tests = pd.read_csv("/home/cse587/knn_data/TestNN.csv").values.tolist()
    for i in range(len(tests)):
        test_dict[i] = []  # initializing my dictionary with empty list for all test values
    for line in sys.stdin:
        if count == 0:  # Skipping the header
            count += 1
            continue

        line = line.strip()
        # making features ie extracting column values of a train row by splitting it with comma
        features = line.split(',')
        for index, test in enumerate(tests):
            avg = 0.0
            for i in range(len(features) - 1):
                if features[i] != '':
                    avg += (float(features[i]) - float(
                        test[i])) ** 2  # calculating the square of the distance between the train row and the test row
            dist = avg
            # making a pair in which first element is distance between the train row and test row and second
            pair = (dist, features[len(features) - 1])
            # element is the label with which the distance is being calculated
            test_dict[index].append(pair)  # pushing the pair in the list which is the value of the key
    for key, value in test_dict.items():
        value = sorted(value)
        # printing the first k values of the sorted list as per distance, which shows we are taking nearest k points,
        for i in range(k):
            # this will also be done in reducer so we will not miss any distances.
            print('%s\t%f\t%d' % ( key, float(value[i][0]), int(value[i][1])))
            # printing the value in which key is test row number and values are distances


if __name__ == "__main__":
    main()
