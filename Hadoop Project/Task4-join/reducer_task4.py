#! /usr/bin/env python3

import sys

employeeDict = {}  # join1
salaryDict = {}  # join2
flag = False

for line in sys.stdin:
    line = line.strip()  # remove leading and trailing whitespaces
    employeeId, employeeName, salary, country, code = line.split('\t')  # separating the terms received from mapper

    if salary == "-1":  # if the data is coming from Join1 table
        employeeDict[employeeId] = employeeName
    else:  # if the data is coming from Join2 table
        salaryDict[employeeId] = [employeeId, salary, country, code]

for employeeId in salaryDict.keys():  # joining the tables for keys present in Join2 table
    employeeName = employeeDict[
        salaryDict[employeeId][0]]  # employee name will be retrieved from employeeDict dictionary
    salary = salaryDict[employeeId][1]  # other details from salaryDict dictionary
    country = salaryDict[employeeId][2]
    code = salaryDict[employeeId][3]

    print('%s\t%s\t%s\t%s\t%s' % (employeeId, employeeName, salary, country, code))

# this loop is for printing details of those keys which are not present in Join2 table but are there in Join1 table
for employeeId in employeeDict.keys():
    for employeeId1 in salaryDict.keys():  # every key is first checked if already printed
        if employeeId == employeeId1:
            flag = True

    if not flag:  # if key is not already printed, its details are extracted from employeeDict and printed
        employeeName = employeeDict[employeeId]
        salary = "-1"
        country = "-1"
        code = "-1"

        print('%s\t%s\t%s\t%s\t%s' % (employeeId, employeeName, salary, country, code))
    flag = False
