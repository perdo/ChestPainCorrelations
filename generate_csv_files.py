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

#def load_gender_data(regions, data):
#    for gender in ('M', 'F'):
#        gender_full = [x for x in data if x['Patient Sex'] == gender]
#        for region in regions:
#            gender_dated = load_single_region_data(region, gender_full)
#            name = re.sub(r'[()/ ]','',"%sData%s.csv" %(gender, region))
#            writer = csv.writer(open(name, 'wt'))
#            writer.writerow( ('Region', 'Date', 'Day', 'Count', 'Time Point') )
#            for line in gender_dated:
#                writer.writerow((line['Region'], line['Date'], line['Day'], line['Count'], line['Time_Point']))

def load_generic_data(regions, data):
    for region in regions:
        dated = load_single_region_data(region, data)
        name = re.sub(r'[()/ ]', '', "Data%s.csv" %(region))
        writer = csv.writer(open(name, 'wt'))
        writer.writerow( ('Region', 'Date', 'Day', 'Count', 'Count Female', 'Count Male', 'Time Point') )
        time_point = 1
        for line in dated:
            writer.writerow((line['Region'], line['Date'], line['Day'], line['Count'], line['Count Female'], line['Count Male'], time_point))
            time_point += 1

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
    marker = 0
    for entry in data:
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
        for sub_entry in data[marker:]:
            sub_unique_pair = (sub_entry['Region'], sub_entry['Ordered'])
            if unique_pair == sub_unique_pair:
                count_patients += 1
                if sub_entry['Patient Sex'] == 'F':
                    count_female += 1
        else:
            data_dated.append(
                                {'Region': region, 
                                'Date': my_date, 
                                'Day': int_week_to_string[my_date.weekday()], 
                                'Count': count_patients,
                                'Count Female': count_female,
                                'Count Male': count_patients - count_female,}
                                )
    data_dated = sorted(data_dated, key= lambda x: x['Date'])
    #for i in range(1,len(data_dated)):
    #    current_date = data_dated[i]['Date']
    #    previous_date = data_dated[i-1]['Date']
    #    if int(current_date.day) <= int(previous_date.day) and int(current_date.day != 1):
    #        if int(current_date.month) != previous_date.month:
    #            for missing_day in range(1, int(current_date.day)):
    #                try:
    #                    new_date = datetime.date(year=previous_date.year, month=current_date.month, day=missing_day)
    ###                except:
    #                    print previous_date
    #                    print current_date
    #                    raise
    #                data_dated[i]['Time_Point'] +=1
    #                data_dated.insert(i + missing_day,                                
    #                                {'Region': region, 
    #                                'Date': new_date, 
    #                                'Day': int_week_to_string[new_date.weekday()], 
    #                                'Count': 0, 
    #                                'Time_Point': i, })
#
#            else:
#                try:
#                    new_date = datetime.date(year=previous_date.year, month=previous_date.month, day=previous_date.day+1)
#                except:
#                    print previous_date
#                    print current_date
#                    raise
#                data_dated[i]['Time_Point'] +=1
#                data_dated.insert(i,                                
#                                    {'Region': region, 
#                                    'Date': new_date, 
#                                    'Day': int_week_to_string[new_date.weekday()], 
#                                    'Count': 0, 
#                                    'Time_Point': i, })
    return data_dated
        

if __name__ == '__main__':
    data_file = sys.argv[1]
    regions = ('Oregon (Eugene/Springfield)', 'Washington (Longview)', 'Washington (Whatcom)', )
    #regions = ('Oregon (Eugene/Springfield)')
    #regions = ('Washington (Longview)', )
    #regions = ('Washington (Whatcom)', )
    #load_gender_data(regions, load_data(data_file))
    load_generic_data(regions, load_data(data_file))
