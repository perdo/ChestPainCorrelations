import pandas as pd
from collections import defaultdict
import datetime
import csv
import sys
import re

class CsvGenerator:
    """ lah """

    def __init__(self, data_file, bucket_size=0, regions=0):
        self.bucket_size = bucket_size
        self.data_file = data_file
        self.regions = regions
        
    #def bucket_key(self, value):
    #    for i in range(0, 130, self.bucket_size):
    #        if value in range(i+1, i + self.bucket_size + 1):
    #            return "Patient Count Ages %s-%s" %(i + 1, i + self.bucket_size)

    #def partition(self, seq):
    #    d = {}
    #    for i in range(0, 130, self.bucket_size):
    #        d["Patient Count Ages %s-%s" % (i + 1, i + self.bucket_size)] = 0
    #    for x in seq:
    #        d[self.bucket_key(x)] += 1
    #    return d

    def load_data(self):
        input_file = csv.DictReader(open(self.data_file))
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
        self.load_generic_data(working_list)

    def load_generic_data(self, data):
        day_to_weekday_weekend={
                        'Monday':'weekday',
                        'Tuesday':'weekday',
                        'Wednesday':'weekday',
                        'Thursday':'weekday',
                        'Friday':'weekday',
                        'Saturday':'weekend',
                        'Sunday':'weekend',
                }
        for region in self.regions:
            dated = self.load_single_region_data(region, data)
            name = re.sub(r'[()/ ]', '', "Data%s.csv" %(region))
            name = 'generated_csv/' + name
            writer = csv.writer(open(name, 'wt'))
            sorted_header = tuple(sorted(tuple(dated[0].keys()))) + ('Time Point',) # Since we don't know age buckets at runtime this is dynamic
            writer.writerow(sorted_header)
            time_point = 1
            for line in dated:
                my_tuple = tuple(line[x] for x in sorted_header if x != 'Time Point') + (time_point,) # See above comment
                writer.writerow(my_tuple)
                time_point += 1
            for day in day_to_weekday_weekend.keys():
                time_point = 1
                name = re.sub(r'[()/ ]', '', "Data%s_%s.csv" %(region, day))
                name = 'generated_csv/' + name
                writer = csv.writer(open(name, 'wt'))
                writer.writerow(sorted_header)
                for line in dated:
                    if line['Day'] != day:
                        continue
                    my_tuple = tuple(line[x] for x in sorted_header if x != 'Time Point') + (time_point,)
                    writer.writerow(my_tuple)
                    time_point += 1
            for day in ('weekday', 'weekend'):
                time_point = 1
                name = re.sub(r'[()/ ]', '', "Data%s_%s.csv" %(region, day))
                name = 'generated_csv/' + name
                writer = csv.writer(open(name, 'wt'))
                writer.writerow(sorted_header)
                next_row = False
                week_sum = 0
                for line in dated:
                    if day == 'weekday' and line['Day'] in ('Saturday', 'Sunday'):
                        next_row = True 
                        continue
                    elif day == 'weekend' and line['Day'] not in ('Saturday', 'Sunday'):
                        next_row = True
                        continue
                    if next_row and week_sum != 0:
                        my_tuple = tuple(line[x] if x != 'Patient Count' else week_sum for x in sorted_header if x != 'Time Point') + (time_point,)
#                        my_tuple['Patient Count'] = week_sum
                        writer.writerow(my_tuple)
                        time_point += 1
                        next_row = False
                        week_sum = 0
                        week_sum += line['Patient Count'] 
                    else:
                        next_row = False
                        week_sum += line['Patient Count'] 
                    

    def load_single_region_data(self, input_region, data):
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
            #count_age = []
            region = entry['Region']
            if entry['Patient Sex'] == 'F':
                count_female = 1
            if input_region != region:
                continue
            #if entry['Patient Age'][-1] != 'Y':
            #    continue
            #try:
            #    count_age.append(int(entry['Patient Age'][0:-1]))
            #except:
            #    print entry
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
                    #try:
                    #    count_age.append(int(sub_entry['Patient Age'][0:-1]))
                    #except:
                    #    print sub_entry
            else:
                #count_age_buckets = self.partition(count_age)
                input_dict =    {'Region': region, 
                                    'Date': my_date, 
                                    'Day': int_week_to_string[my_date.weekday()], 
                                    'Patient Count': count_patients,
                                    'Patient Count Female': count_female,
                                    'Patient Count Male': count_patients - count_female,
                                    } 
                #input_dict.update(count_age_buckets)
                data_dated.append(input_dict)
                                                                    
        data_dated = sorted(data_dated, key= lambda x: x['Date'])
        return data_dated
        

if __name__ == '__main__':
    data_file = sys.argv[1]
    regions = ('Oregon (Eugene/Springfield)', 'Washington (Longview)', 'Washington (Whatcom)', )
    bucket_size = 25
    #regions = ('Oregon (Eugene/Springfield)',)
    #regions = ('Washington (Longview)', )
    #regions = ('Washington (Whatcom)', )
    csvgen = CsvGenerator(data_file, bucket_size, regions)
    csvgen.load_data()
