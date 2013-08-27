import datetime
import csv
import sys
import pylab
import operator
import math

def output_form_single(data_file):
    counts = []
    counts_fem = []
    counts_male = []
    input_file = csv.DictReader(open(data_file))

    for line in input_file:
        counts_male.append(int(line['Count Male']))
        counts_fem.append(int(line['Count Female']))
    
    name = 'output_test_male.csv'
    writer = csv.writer(open(name, 'wt'))
    for i in range(0, int(math.ceil(len(counts_male))/2.0) + 1):
        writer.writerow([counts_male[j] for j in range(i, len(counts_male) - i)])
        #print [counts_male[j] for j in range(i, len(counts_fem) - i)]
    name2 = 'output_test_female.csv'
    writer = csv.writer(open(name2, 'wt'))
    for i in range(0, int(math.ceil(len(counts_male))/2.0) + 1):
        writer.writerow([counts_fem[j] for j in range(i, len(counts_fem) - i)])
        #print [counts_fem[j] for j in range(i, len(counts_fem) - i)]

def output_form_regions(regions):
    for region in regions:
        counts = []
        counts_fem = []
        counts_male = []
        input_file = csv.DictReader(open("Data%s.csv" %(region,)))

        for line in input_file:
            counts_male.append(int(line['Count Male']))
            counts_fem.append(int(line['Count Female']))
        
        name = "output_%s_male.csv" %(region,)
        writer = csv.writer(open(name, 'wt'))
        for i in range(0, int(math.ceil(len(counts_male))/2.0) + 1):
            writer.writerow([counts_male[j] for j in range(i, len(counts_male) - i)])
            #print [counts_male[j] for j in range(i, len(counts_fem) - i)]
        name2 = "output_%s_female.csv" %(region,)
        writer = csv.writer(open(name2, 'wt'))
        for i in range(0, int(math.ceil(len(counts_male))/2.0) + 1):
            writer.writerow([counts_fem[j] for j in range(i, len(counts_fem) - i)])
            #print [counts_fem[j] for j in range(i, len(counts_fem) - i)]
                    

def main():
    if len(sys.argv) == 2:
        data_file = sys.argv[1]
        output_form_single(data_file)
    else:
        regions = ('OregonEugeneSpringfield', 'WashingtonLongview', 'WashingtonWhatcom', )
        output_form_regions(regions)

if __name__ == '__main__':
    main()
