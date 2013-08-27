import datetime
import csv
import sys
import pylab
import operator

def main():
    data_file = sys.argv[1]
    input_file = csv.DictReader(open(data_file))
    days = []
    deltas = []
    for line in input_file:
       days.append(int(line['Date'].split('-')[2]))
    for i in range(0, len(days)-1):
        if days[i] >= days[i+1] and days[i+1] != 1:
            print i


if __name__ == '__main__':
    main()
