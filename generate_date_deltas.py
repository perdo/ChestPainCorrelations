import datetime
import csv
import sys
import pylab
import operator

def generate_delta_plot(data_file, day_length=1):
    input_file = csv.DictReader(open(data_file))
    counts = []
    deltas = []
    for line in input_file:
        counts.append(line['Count'])
    for i in range(day_length, len(counts) - day_length, day_length): # start: date length, end, length - day_length, step day_length
        #print "pos %s - pos %s" %(i, i-1)
        deltas.append(int(counts[i])- int(counts[i-1])) #subtract boundaries of day length groupings

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
    print counts
    #for k in range(1, 31):
    for k in range(1, 3):
        w = []
        start_error = {}
        error = 0
        for start in range(0, k):  
            for i in range(start, len(counts) - k + 1, k):
                w.append(sum(counts[i:i+k])/k) 
                print "countsB" 
                print counts[i:i+k] 
                print "countsE"
            error = sum([abs(w[pos] - w[pos-1]) for pos in range(1, len(w))]) 
            error = sum([abs(w[pos] - w[pos-1]) for pos in range(1, len(w))]) 
            print "W"
            print w
            print "WE"
            #print [(w[pos],w[pos-k]) for pos in range(k + start, len(w)-k)]
            #print [pos for pos in range(k + start, len(w)-k)]
            start_error[start] = error
        best_start_key = min(start_error.iteritems(), key=operator.itemgetter(1))[0]
        print sorted(start_error.items(), key=lambda x: x[1])
        cycle_errors[(k,best_start_key)] = start_error[best_start_key]
    print sorted(cycle_errors.items(), key=lambda x: x[1])
    #print sorted(cycle_errors.items())


def main():
    data_file = sys.argv[1]
    choose_best_cycle(data_file)
    
if __name__ == '__main__':
    main()
