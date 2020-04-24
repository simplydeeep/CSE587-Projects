#! /usr/bin/env python3

import sys

separator = '\t'

for line in sys.stdin:
    line = line.strip()  # remove leading and trailing whitespace
    splits = line.split(",")  # csv file,  values will be separated by comma

    employeeId = "-1"  # by default all values are stored as -1
    employeeName = "-1"
    salary = "-1"
    country = "-1"
    code = "-1"

    if len(splits) == 2:  # join1 data
        if splits[0] != "Employee ID":
            employeeId = splits[0]
            employeeName = splits[1]

    elif len(splits) == 5:  # join2 data (if country name doesn't have comma)
        employeeId = splits[0]
        salary = splits[1] + "," + splits[2]
        country = splits[3]
        code = splits[4]

    elif len(splits) == 6:  # join2 data (if country name also has comma)
        employeeId = splits[0]
        salary = splits[1] + "," + splits[2]
        country = splits[3] + ", " + splits[4]
        code = splits[5]

    if employeeId != "-1":  # to make sure all values sent to reducer have employee Id present
        print('%s%s%s%s%s%s%s%s%s' % (employeeId, separator, employeeName, separator, salary, separator, country,
                                      separator, code))
