from collections import defaultdict
import datetime
import csv
import sys
import pylab
import operator
import math

#def output_form_single(data_file):
#    counts = []
#    counts_fem = []
#    counts_male = []
#    input_file = csv.DictReader(open(data_file))
#
#    for line in input_file:
#        counts_male.append(int(line['Patient Count Male']))
#        counts_fem.append(int(line['Patient Count Female']))
#    
#    name = 'output_test_male.csv'
#    writer = csv.writer(open(name, 'wt'))
#    for i in range(0, int(math.ceil(len(counts_male))/2.0) + 1):
#        writer.writerow([counts_male[j] for j in range(i, len(counts_male) - i)])
#    name = 'output_test_female.csv'
#    writer = csv.writer(open(name, 'wt'))
#    for i in range(0, int(math.ceil(len(counts_male))/2.0) + 1):
#        writer.writerow([counts_fem[j] for j in range(i, len(counts_fem) - i)])

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
        
        name = "output_%s_male.csv" %(region,)
        writer = csv.writer(open(name, 'wt'))
        for i in range(0, int(math.ceil(len(counts_male))/2.0) + 1):
            writer.writerow([counts_male[j] for j in range(i, len(counts_male) - i)]) # prints correct portion of data for auto regression
        name = "output_%s_female.csv" %(region,)
        writer = csv.writer(open(name, 'wt'))
        for i in range(0, int(math.ceil(len(counts_male))/2.0) + 1):
            writer.writerow([counts_fem[j] for j in range(i, len(counts_fem) - i)]) 
        for entry in counts_ages.keys():
            if len(counts_ages[entry]) == 0: # No need for extra files if the data is empty. Maybe threshold this to like 10?
                continue
            name = "output_%s_%s.csv" %(region, entry)
            writer = csv.writer(open(name, 'wt'))
            for i in range(0, int(math.ceil(len(counts_ages[entry]))/2.0) + 1):
                writer.writerow([counts_ages[entry][j] for j in range(i, len(counts_ages[entry]) - i)])


                    

def main():
    #if len(sys.argv) == 2:
    #    data_file = sys.argv[1]
    #    output_form_single(data_file)
    #else:
    regions = ('OregonEugeneSpringfield', 'WashingtonLongview', 'WashingtonWhatcom', )
    output_form_regions(regions)

if __name__ == '__main__':
    main()
