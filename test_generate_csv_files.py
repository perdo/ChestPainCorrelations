import generate_csv_files
import unittest

class TestGenerateCsvFiles:
    """ blah """

    def setUp(self):
        self.regions = ('Oregon (Eugene/Springfield)', 'Washington (Longview)', 'Washington (Whatcom)', )
        self.test_data_file ='test.csv'
        self.csvgen = CsvGenerator(self.test_data_file, self.regions)


    def check_read_data(self):
        self.csvgen.read_data()
        result = [
                            {'Region': 'Washington (Whatcom)', 
                            'Patient Sex': 'F', 
                            'Patient Age': '67Y', 
                            'Ordered': datetime.date(year=2011, month=1, day=1)
                            },
                            {'Region': 'Washington (Whatcom)', 
                            'Patient Sex': 'M', 
                            'Patient Age': '68Y', 
                            'Ordered': datetime.date(year=2011, month=1, day=1)
                            },
                            {'Region': 'Oregon (Eugene/Springfield)', 
                            'Patient Sex': 'M', 
                            'Patient Age': '59Y', 
                            'Ordered': datetime.date(year=2011, month=1, day=1)
                            },
                            {'Region': 'Oregon (Eugene/Springfield)', 
                            'Patient Sex': 'M', 
                            'Patient Age': '83Y', 
                            'Ordered': datetime.date(year=2011, month=1, day=1)
                            },
                            {'Region': 'Washington (Longview)', 
                            'Patient Sex': 'F', 
                            'Patient Age': '74Y', 
                            'Ordered': datetime.date(year=2011, month=1, day=1)
                            },
                            {'Region': 'Washington (Longview)', 
                            'Patient Sex': 'M', 
                            'Patient Age': '88Y', 
                            'Ordered': datetime.date(year=2011, month=1, day=1)
                            },
                            {'Region': 'Washington (Longview)', 
                            'Patient Sex': 'F', 
                            'Patient Age': '88Y', 
                            'Ordered': datetime.date(year=2011, month=1, day=2)
                            },
                            {'Region': 'Washington (Longview)', 
                            'Patient Sex': 'F', 
                            'Patient Age': '50Y', 
                            'Ordered': datetime.date(year=2011, month=1, day=1)
                            },
                      ]
        assertEqual(result, self.csv.gen.data)

    def check_load_single_region_data(self):
        result = [
                            {'Region': 'Washington (Longview)', 
                            'Patient Count Female': 2,
                            'Day': 'Saturday',
                            'Patient Count': 3, 
                            'Patient Count Male': 1, 
                            'Ordered': datetime.date(year=2011, month=1, day=1),
                            },
                            {'Region': 'Washington (Longview)', 
                            'Patient Count Female': 1,
                            'Day': 'Sunday',
                            'Patient Count': 1, 
                            'Patient Count Male': 0, 
                            'Ordered': datetime.date(year=2011, month=1, day=2),
                            },
                    
                ]
        assertEqual(result, self.csv.gen.load_single_region_data('Washington (Longview)'))
    
if __name__ == '__main__':
    unittest.main()
