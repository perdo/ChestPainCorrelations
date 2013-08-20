import datetime
import csv
import sys
import pylab
import operator

def generate_delta_plot(data_file, day_length=1, start=0):
    input_file = csv.DictReader(open(data_file))
    counts = []
    deltas = []
    for line in input_file:
        counts.append(int(line['Count']))
    left_over_values = (len(counts)-start) % day_length
    deltas = [counts[pos] - counts[pos-day_length] for pos in range(day_length+start, len(counts) - left_over_values)]

    pylab.hist(deltas, normed=True)
    pylab.title("Spaced by %s days" % (day_length,))
    #pylab.show() # We want to just output to file instead of showing
    pylab.savefig("%s_%s_days.png" %(data_file.split('.')[0], day_length,))  #strips .csv and uses that as name base
    pylab.clf()

def choose_best_cycle(data_file):
    input_file = csv.DictReader(open(data_file))
    counts = []
    cycle_errors = {}
    for line in input_file:
        counts.append(int(line['Count']))
    #print counts
    for k in range(1, 31):
    #for k in range(1, 3):
        start_error = {}
        error = 0
        for start in range(0, k): #issue, strips off anything that doesn't fit on left side but not right side maybe I made it other side issue? 
            left_over_values = (len(counts)-start) % k
            error_terms = [1.0 * abs(counts[pos] - counts[pos-k]) for pos in range(k+start, len(counts)-left_over_values)]
            #error_terms = [1.0 * abs(counts[pos] - counts[pos-k]) for pos in range(k+start, len(counts))]
            error = 1.0 * sum(error_terms) / len(error_terms)
            start_error[start] = error
        best_start_key = min(start_error.iteritems(), key=operator.itemgetter(1))[0]
        #print sorted(start_error.items(), key=lambda x: x[1])
        cycle_errors[(k,best_start_key)] = start_error[best_start_key]
    print sorted(cycle_errors.items(), key=lambda x: x[1])[:5]
    return sorted(cycle_errors.items(), key=lambda x: x[1])[:5]
    #print sorted(cycle_errors.items())


def main():
    data_file = sys.argv[1]
    best_10_cycles = choose_best_cycle(data_file)
    for entry in best_10_cycles:
        generate_delta_plot(data_file, day_length=int(entry[0][0]), start=int(entry[0][1]))
    
if __name__ == '__main__':
    main()
