from collections import defaultdict
import datetime
import csv
import sys
import pylab
import operator
import math

def output_form_regions(regions):
    for region in regions:
        counts = []
        counts_fem = []
        counts_male = []
        counts_ages = defaultdict(list)
        input_file = csv.DictReader(open("./generated_csv/Data%s.csv" %(region,)))

        for line in input_file:
            for entry in line.keys():
                if 'Patient Count Ages' in entry:
                    counts_ages[entry].append(int(line[entry]))
            counts_male.append(int(line['Patient Count Male']))
            counts_fem.append(int(line['Patient Count Female']))
        output_form_single(region, counts_fem, 'female')
        output_form_single(region, counts_male, 'male')
        
        for day in ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'):
            counts = []
            input_file = csv.DictReader(open("./generated_csv/Data%s_%s.csv" %(region,day,)))
            for line in input_file:
                counts.append(line['Patient Count'])
            output_form_single(region, counts, day)
        for day in ('weekday', 'weekend'):
            counts = []
            input_file = csv.DictReader(open("./generated_csv/Data%s_%s.csv" %(region,day,)))
            for line in input_file:
                counts.append(line['Patient Count'])
            output_form_single(region, counts, day)


def output_form_single(region, counts, output_type):
    """  """
    name = "./autoregression_output/output_%s_%s.csv" %(region, output_type,)

    writer = csv.writer(open(name, 'wt'))
    length = 0
    while(length<len(counts)):
        writer.writerow([counts[i] for i in range(length, -1, - 1)])
        length += 1

                    

def main():
    regions = ('OregonEugeneSpringfield', 'WashingtonLongview', 'WashingtonWhatcom', 'test')
    output_form_regions(regions)

if __name__ == '__main__':
    main()
