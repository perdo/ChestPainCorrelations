import pandas as pd
import datetime
import csv
import sys
import re

def bucket_key(value, bucket_size):
    for i in range(0, 130, bucket_size):
        if value in range(i+1, i + bucket_size + 1):
            return "Patient Count Ages %s-%s" %(i + 1, i + bucket_size)

def partition(seq, bucket_size, key=bucket_key):
    d = {}
    for i in range(0, 130, bucket_size):
        d["Patient Count Ages %s-%s" % (i + 1, i + bucket_size)] = 0
    for x in seq:
        d[key(x, bucket_size)] += 1
    return d

def load_data(data_file):
    input_file = csv.DictReader(open(data_file))
    working_list = []
    for line in input_file:
        split_date = line['ORDERED'].split('/')
        year =  int(split_date[2][:-5])
        month = int(split_date[0])
        day = int(split_date[1])
        age = line['Patient Age']
        working_list.append(
                                {'Region': line['Region'], 
                                'Patient Sex': line['Patient Sex'], 
                                'Patient Age': age, 
                                'Ordered': datetime.date(year=year, month=month, day=day)
                                }
                            )
    return working_list

def load_generic_data(regions, data, bucket_size=25):
    for region in regions:
        dated = load_single_region_data(region, data, bucket_size)
        name = re.sub(r'[()/ ]', '', "Data%s.csv" %(region))
        writer = csv.writer(open(name, 'wt'))
        #writer.writerow( ('Region', 'Date', 'Day', 'Patient Count', 'Patient Count Female', 'Patient Count Male', 'Patient Age', 'Time Point') )
        sorted_header = tuple(sorted(tuple(dated[0].keys()))) + ('Time Point',)
        writer.writerow(sorted_header)
        time_point = 1
        for line in dated:
            my_tuple = tuple(line[x] for x in sorted_header if x != 'Time Point') + (time_point,)
            writer.writerow(my_tuple)
            #writer.writerow((line['Region'], line['Date'], line['Day'], line['Count'], line['Count Female'], line['Count Male'], line['Patient Age'], time_point))
            time_point += 1

def load_single_region_data(input_region, data, bucket_size=25):
    int_week_to_string = {
                            0: 'Monday',
                            1: 'Tuesday',
                            2: 'Wednesday',
                            3: 'Thursday',
                            4: 'Friday',
                            5: 'Saturday',
                            6: 'Sunday',
                        }
    found_dates = {}
    data_dated = []
    marker = 0
    for entry in data:
        marker += 1
        count_patients = 1
        count_female = 0
        count_age = []
        region = entry['Region']
        if entry['Patient Sex'] == 'F':
            count_female = 1
        if input_region != region:
            continue
        if entry['Patient Age'][-1] != 'Y':
            continue
        try:
            count_age.append(int(entry['Patient Age'][0:-1]))
        except:
            print entry
        my_date = entry['Ordered']
        unique_pair = (region, my_date)
        if unique_pair in found_dates:
            continue
        else:
            found_dates[unique_pair] = 1
        for sub_entry in data[marker:]:
            sub_unique_pair = (sub_entry['Region'], sub_entry['Ordered'])
            if unique_pair == sub_unique_pair and sub_entry['Patient Age'][-1] == 'Y':
                count_patients += 1
                if sub_entry['Patient Sex'] == 'F':
                    count_female += 1
                try:
                    count_age.append(int(sub_entry['Patient Age'][0:-1]))
                except:
                    print sub_entry
        else:
            count_age_buckets = partition(count_age, bucket_size)
            input_dict =    {'Region': region, 
                                'Date': my_date, 
                                'Day': int_week_to_string[my_date.weekday()], 
                                'Patient Count': count_patients,
                                'Patient Count Female': count_female,
                                'Patient Count Male': count_patients - count_female,
                                } 
            input_dict.update(count_age_buckets)
            data_dated.append(input_dict)
                                                                
    data_dated = sorted(data_dated, key= lambda x: x['Date'])
    return data_dated
        

if __name__ == '__main__':
    data_file = sys.argv[1]
    regions = ('Oregon (Eugene/Springfield)', 'Washington (Longview)', 'Washington (Whatcom)', )
    bucket_size = 25
    #regions = ('Oregon (Eugene/Springfield)')
    #regions = ('Washington (Longview)', )
    #regions = ('Washington (Whatcom)', )
    #load_gender_data(regions, load_data(data_file))
    load_generic_data(regions, load_data(data_file), bucket_size)
