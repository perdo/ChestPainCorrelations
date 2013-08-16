import pandas as pd
import datetime
import csv
import sys
import re

def load_data(data_file):
    input_file = csv.DictReader(open(data_file))
    working_list = []
    for line in input_file:
        split_date = line['ORDERED'].split('/')
        year =  int(split_date[2][:-5])
        month = int(split_date[0])
        day = int(split_date[1])
        age = line['Patient Age'][:-1]
        working_list.append(
                                {'Region': line['Region'], 
                                'Patient Sex': line['Patient Sex'], 
                                'Patient Age': age, 
                                'Ordered': datetime.date(year=year, month=month, day=day)
                                }
                            )
    return working_list

def load_gender_data(regions, data):
    for gender in ('M', 'F'):
        gender_full = [x for x in data if x['Patient Sex'] == gender]
        for region in regions:
            gender_dated = load_single_region_data(region, gender_full)
            name = re.sub(r'[()/ ]','',"%sData%s.csv" %(gender, region))
            writer = csv.writer(open(name, 'wt'))
            writer.writerow( ('Region', 'Date', 'Day', 'Count', 'Time Point') )
            for line in gender_dated:
                writer.writerow((line['Region'], line['Date'], line['Day'], line['Count'], line['Time_Point']))

def load_generic_data(regions, data):
    for region in regions:
        dated = load_single_region_data(region, data)
        name = re.sub(r'[()/ ]', '', "Data%s.csv" %(region))
        writer = csv.writer(open(name, 'wt'))
        writer.writerow( ('Region', 'Date', 'Day', 'Count', 'Time Point') )
        for line in dated:
            writer.writerow((line['Region'], line['Date'], line['Day'], line['Count'], line['Time_Point']))

def load_single_region_data(input_region, data):
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
    time_point = 1
    marker = 0
    for entry in data:
        marker += 1
        count_patients = 1
        region = entry['Region']
        if input_region != region:
            continue
        my_date = entry['Ordered']
        unique_pair = (region, my_date)
        if unique_pair in found_dates:
            continue
        else:
            found_dates[unique_pair] = 1
        for sub_entry in data[marker:]:
            sub_unique_pair = (sub_entry['Region'], sub_entry['Ordered'])
            if unique_pair == sub_unique_pair:
                count_patients += 1
        else:
            data_dated.append(
                                {'Region': region, 
                                'Date': my_date, 
                                'Day': int_week_to_string[my_date.weekday()], 
                                'Count': count_patients, 
                                'Time_Point': time_point, }
                                )
            time_point += 1
    return data_dated
        

if __name__ == '__main__':
    data_file = sys.argv[1]
    regions = ('Oregon (Eugene/Springfield)', 'Washington (Longview)', 'Washington (Whatcom)', )
    load_gender_data(regions, load_data(data_file))
    load_generic_data(regions, load_data(data_file))
