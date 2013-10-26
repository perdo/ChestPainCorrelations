import pandas as pd
from collections import defaultdict
import datetime
import csv
import sys
import re

class CsvGenerator:
    """ lah """

    def __init__(self, data_file, regions):
        self.data_file = data_file
        self.regions = regions
        self.data = []
        self.day_to_weekday_weekend={
                        'Monday':'weekday',
                        'Tuesday':'weekday',
                        'Wednesday':'weekday',
                        'Thursday':'weekday',
                        'Friday':'weekday',
                        'Saturday':'weekend',
                        'Sunday':'weekend',
                }

        
    def read_data(self):
        input_file = csv.DictReader(open(self.data_file))
        for line in input_file:
            split_date = line['ORDERED'].split('/')
            year =  int(split_date[2][:-5])
            month = int(split_date[0])
            day = int(split_date[1])
            age = line['Patient Age']
            self.data.append(
                                {'Region': line['Region'], 
                                'Patient Sex': line['Patient Sex'], 
                                'Patient Age': age, 
                                'Ordered': datetime.date(year=year, month=month, day=day)
                                }
                            )

    def write_data(self):
        for region in self.regions:
            self.write_weekday_weekend(region)
            self.write_day(region)
            self.write_naive(region)
                                            
    def write_weekday_weekend(self, region):
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
                        writer.writerow(my_tuple)
                        time_point += 1
                        next_row = False
                        week_sum = 0
                        week_sum += line['Patient Count'] 
                    else:
                        next_row = False
                        week_sum += line['Patient Count'] 


    def write_day(self, region):
        for day in self.day_to_weekday_weekend.keys():
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


    def write_naive(self, region):
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


    def load_single_region_data(self, input_region):
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
        for entry in self.data:
            marker += 1
            count_patients = 1
            count_female = 0
            region = entry['Region']
            if entry['Patient Sex'] == 'F':
                count_female = 1
            if input_region != region:
                continue
            my_date = entry['Ordered']
            unique_pair = (region, my_date)
            if unique_pair in found_dates:
                continue
            else:
                found_dates[unique_pair] = 1
            for sub_entry in self.data[marker:]:
                sub_unique_pair = (sub_entry['Region'], sub_entry['Ordered'])
                if unique_pair == sub_unique_pair and sub_entry['Patient Age'][-1] == 'Y':
                    count_patients += 1
                    if sub_entry['Patient Sex'] == 'F':
                        count_female += 1
            else:
                input_dict =    {'Region': region, 
                                    'Date': my_date, 
                                    'Day': int_week_to_string[my_date.weekday()], 
                                    'Patient Count': count_patients,
                                    'Patient Count Female': count_female,
                                    'Patient Count Male': count_patients - count_female,
                                    } 
                data_dated.append(input_dict)
                                                                    
        data_dated = sorted(data_dated, key= lambda x: x['Date'])
        return data_dated
        

if __name__ == '__main__':
    data_file = sys.argv[1]
    regions = ('Oregon (Eugene/Springfield)', 'Washington (Longview)', 'Washington (Whatcom)', )
    #regions = ('Oregon (Eugene/Springfield)',)
    #regions = ('Washington (Longview)', )
    #regions = ('Washington (Whatcom)', )
    csvgen = CsvGenerator(data_file, regions)
    csvgen.read_data()
    csvgen.write_data()
