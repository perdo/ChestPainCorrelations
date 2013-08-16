import datetime
import csv
import sys
import pylab

def generate_delta_plot(data_file, day_length=1):
    input_file = csv.DictReader(open(data_file))
    counts = []
    deltas = []
    for line in input_file:
        counts.append(line['Count'])
    
    for i in range(day_length, len(counts) - day_length, day_length): # start: date length, end, length - day_length, step day_length
        #print "pos %s - pos %s" %(i, i-1)
        deltas.append(int(counts[i])- int(counts[i-1])) #subtract boundaries of day length groupings

    
    #for i, item in enumerate(counts):
    #    if len(counts) <= i + day_length:
    #        break
    #    else:
    #        deltas.append(int(counts[i+day_length])- int(counts[i]))
    #print deltas
    pylab.hist(deltas, normed=True)
    pylab.title("Spaced by %s days" % (day_length,))
    #pylab.show()
    pylab.savefig("%s_%s_days.png" %(data_file.split('.')[0], day_length,))  #strips .csv and uses that as name base
    pylab.clf()

def main():
    data_file = sys.argv[1]
    generate_delta_plot(data_file, 1)
    generate_delta_plot(data_file, 7)
    generate_delta_plot(data_file, 14)
    generate_delta_plot(data_file, 21)
    generate_delta_plot(data_file, 30)
    
if __name__ == '__main__':
    main()
