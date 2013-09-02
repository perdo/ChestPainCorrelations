import csv
import sys

def read_file(data_file):
    reader = csv.DictReader(open(data_file, 'rt'), delimiter='\t')
    main_data = []
    for line in reader:
        year = line['year'].strip()
        period = line['period'].strip()
        if year not in ('2011', '2012', '2013',):
            continue
        series_id = line['series_id'].strip()
        data_series_code = series_id[0].strip()
        srd_code = series_id[4:7].strip() # S41 Oregon, S53 Washington
        if srd_code not in ('S41', 'S53',):
            continue
        if period[0] != 'M':
            continue
        #industryb_code = series_id[4:7].strip()
        #irc_code = series_id[7:12].strip()
        value = line['value'].strip()
        dict_values = {
                        'value': value,
                        'srd_code': srd_code, 
                        'year': year,
                        'series_id':series_id,
                        'period': period,
                        }
        main_data.append(dict_values)
    print len(main_data)
    print main_data[0]
    writer = csv.writer(open('bugleo.csv', 'wt'))
    writer.writerow(('series_id', 'period', 'srd_code', 'year', 'value'))
    for entry in main_data:
        writer.writerow((entry['series_id'], entry['period'], entry['srd_code'], entry['year'], entry['value']))

def main():
    data_file = sys.argv[1]
    read_file(data_file)

if __name__ == '__main__':
    main()
