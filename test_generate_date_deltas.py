import generate_csv_files
import unittest

class TestGenerateCsvFiles:
    """ """
    def setUp(self):
        self.regions = ('tester',)
        self.test_data_file ='test.csv'
        self.csvgen = CsvGenerator(self.test_data_file, self.regions)

    def check_read_data(self):
        self.csvgen.read_data()
        assertEqual(SOMETHING, self.csv.gen.data)

    def check_write__data(self):
        pass
    
if __name__ == '__main__':
    unittest.main()
