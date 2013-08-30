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
        input_file = csv.DictReader(open("Data%s.csv" %(region,)))

        for line in input_file:
            for entry in line.keys():
                if 'Patient Count Ages' in entry:
                    counts_ages[entry].append(int(line[entry]))
            counts_male.append(int(line['Patient Count Male']))
            counts_fem.append(int(line['Patient Count Female']))
        output_form_single(region, counts_fem, 'female')
        output_form_single(region, counts_male, 'male')
        for age_key in counts_ages.keys():
            if len(counts_ages[age_key]) == 0: # No need for extra files if the data is empty. Maybe threshold this to like 10?
                continue
            output_form_single(region, counts=counts_ages[age_key], age_key=age_key)

def output_form_single(region, counts, gender=None, age_key=None):
    """  """
    if gender and gender in ('female', 'male',):
        name = "output_%s_%s.csv" %(region, gender,)
    elif age_key:
        name = "output_%s_%s.csv" %(region, age_key,)
    writer = csv.writer(open(name, 'wt'))
    for i in range(0, int(math.ceil(len(counts))/2.0) + 1):
        writer.writerow([counts[j] for j in range(i, len(counts) - i)]) # prints correct portion of data for auto regression

                    

def main():
    regions = ('OregonEugeneSpringfield', 'WashingtonLongview', 'WashingtonWhatcom', )
    output_form_regions(regions)

if __name__ == '__main__':
    main()
