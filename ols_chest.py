import numpy as np
import csv
import math
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


class ols_models():

    def __init__(self, regions):
        self.regions = regions
        self.X = []
        self.X_day = []
        self.error_list = []

        self.error_comparison = \
            {
                    '1day': [],
                    '7day': [],
                    '28day': [],
                    'mean predictor': [],
                    '210slide': [],
                    '420slide': [],
                    #'Monday': [],
                    #'Tuesday': [],
                    #'Wednesday': [],
                    #'Thursday': [],
                    #'Friday': [],
                    #'Saturday': [],
                    #'Sunday': [],

            }
        self.error_comparison_6mo = \
           {
                '1day': [],
                '7day': [],
                '28day': [],
                'mean predictor': [],
                '210slide': [],
                '420slide': [],
                '104_Monday_slide': [],
                '52_Monday_slide': [],
                '104_Tuesday_slide': [],
                '52_Tuesday_slide': [],
                '104_Wednesday_slide': [],
                '52_Wednesday_slide': [],
                '104_Thursday_slide': [],
                '52_Thursday_slide': [],
                '104_Friday_slide': [],
                '52_Friday_slide': [],
                '104_Saturday_slide': [],
                '52_Saturday_slide': [],
                '104_Sunday_slide': [],
                '52_Sunday_slide': [],
                'Monday': [],
                'Tuesday': [],
                'Wednesday': [],
                'Thursday': [],
                'Friday': [],
                'Saturday': [],
                'Sunday': [],
            }

    def build_ols_model(self,regions):
        """ builds all ols_models and outputs them to csvs """
        for region in regions:

            value_to_key = {1: '1day', 7: '7day', 28: '28day', 210: '210slide', 420: '420slide'}
            x_1 = []
            x_2 = []
            y_predict_list = []
            y_actual = []

            input_file = csv.DictReader(open("generated_csv/Data%s.csv" %(region,)))
            for line in input_file:
                self.X.append(int(line['Patient Count']))
                self.X_day.append(line['Day'])
            for cycle in (1, 7, 28): 
                if cycle == 28:  # ugh.. don't make me read this
                    x_1, x_2, y_predict_list, y_actual = self.build_new_model(region=region, cycle=cycle)
                    self.write_confidence_to_csv(region, y_predict_list, y_actual, x_1, x_2)
                    self.error_comparison['mean predictor'] = self.mean_error_list
                    self.error_comparison_6mo['mean predictor'] = self.post_2yr_mean_error
                else:
                    self.build_new_model(region=region, cycle=cycle)
                self.error_comparison[value_to_key[cycle]] = self.error_list
                self.error_comparison_6mo[value_to_key[cycle]] = self.post_2yr_error
            for days_of_data in (104, 52):
                for day in ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'):
                    self.build_new_model(region=region, day=day, cycle=28, days_of_data=days_of_data) #  dynamic 6mo model
                    self.error_comparison_6mo["%s_" %(days_of_data,) + day + "_slide"] = self.post_2yr_error
                    #self.build_new_model(region=region, day=day, cycle=28) #dynamic 6mo model
                    #self.error_comparison[day] = self.error_list

            self.build_models_static(region, 730/28, 28)
            for days_of_data in (210, 420):
                x_1, x_2, y_predict_list, y_actual = self.build_new_model(region=region, cycle=28, days_of_data=days_of_data)
                self.error_comparison[value_to_key[days_of_data]] = self.error_list
                self.error_comparison_6mo[value_to_key[days_of_data]] = self.post_2yr_error
                if days_of_data == 420:
                    self.write_confidence_to_csv(region, y_predict_list, y_actual, x_1, x_2, days_of_data)
            self.write_errors_to_csv(region)

    def write_confidence_to_csv(self,region, y_predict_list, y_actual, x_1, x_2, days_of_data=None):
        """ Writes confidence interval information to CSV """
        if days_of_data is None:
            self.output_file = open("ols_results/output_confidence_%s.csv" % (region,), 'w')
        else:
            self.output_file = open("ols_results/output_confidence_%s_%s.csv" % (region, days_of_data), 'w')
        writer = csv.writer(self.output_file)
        output_columns = ('y_predict', 'y_actual', 'x_1', 'x_2')
        writer.writerow(output_columns)
        for i in range(len(y_predict_list)):
            writer.writerow((float(y_predict_list[i]), int(y_actual[i][0]), float(x_1[i]), float(x_2[i]),))

    def write_errors_to_csv(self,region):
        """ Creates csv file that has all the errors of each model as the model is built """
        output_file = open("ols_results/output_comparison_result%s.csv" %(region,), 'w')
        output_file_6mo = open("ols_results/output_comparison_result%s_6mo.csv" %(region,), 'w')
        writer = csv.writer(output_file)
        output_columns = sorted(self.error_comparison.keys())
        writer.writerow(output_columns)
        for i in range(len(self.error_comparison['28day'])):
            writer.writerow([self.error_comparison[key][i] for key in sorted(self.error_comparison.keys())])

        writer_6mo = csv.writer(output_file_6mo)
        output_columns_6mo = sorted(self.error_comparison_6mo.keys())
        writer_6mo.writerow(output_columns_6mo)
        writer_6mo.writerow([self.error_comparison_6mo[key] for key in sorted(self.error_comparison_6mo.keys())])

    def build_models_static(self, region, up_to_month, cycle):

        model_dict = {
            'Monday': linear_model.LinearRegression(),
            'Tuesday': linear_model.LinearRegression(),
            'Wednesday': linear_model.LinearRegression(),
            'Thursday': linear_model.LinearRegression(),
            'Friday': linear_model.LinearRegression(),
            'Saturday': linear_model.LinearRegression(),
            'Sunday': linear_model.LinearRegression(),

        }
        for day in ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'):
            self.X_train_list = []
            self.Y_train_list = []
            self.X_test_list = []
            self.Y_test_list = []
            self.X_train = None
            self.Y_train = None
            self.X_test = None
            self.Y_test = None
            self.output_file = open("ols_results/output_static_stratified_ols_%sday_%s_%s" %(cycle, region, day,), 'w')
            for i in range(0, (up_to_month-2)*cycle):
                if self.X_day[i+cycle] == day:
                    self.X_train_list.append(self.X[i:i+cycle])
                    self.Y_train_list.append([self.X[i+cycle]])

            self.X_train = np.matrix(self.X_train_list)
            self.Y_train = np.matrix(self.Y_train_list)
            index = up_to_month*cycle
            while ( index+cycle < len(self.X) ): ## Ends when there are less than 2 months left in our data
                if self.X_day[index + cycle] == day:
                    self.X_test_list.append(self.X[index:index+cycle])
                    self.Y_test_list.append([self.X[index+cycle]])
                index+=1
            self.X_test = np.matrix(self.X_test_list)
            self.Y_test = np.matrix(self.Y_test_list)
            model_dict[day].fit(self.X_train, self.Y_train)

            #print debug information
            self.output_file.write("dim X_train: %s\n" %(self.X_train.shape,))
            self.output_file.write("dim Y_train: %s\n" %(self.Y_train.shape,))
            self.output_file.write("dim X_test: %s\n" %(self.X_test.shape,))
            self.output_file.write("dim Y_test: %s\n" %(self.Y_test.shape,))
            self.output_file.write("cycle: %s\n" %(cycle,))
            self.output_file.write("Current coefficients: %s\n" %(self.regr.coef_,))

            y_predict_current_month = model_dict[day].predict(self.X_test)
            mean_train = (1.0 * sum(self.X[0:len(self.X_train)])) / len(self.X_train) ## Finds mean of Training data
            mean_train = [mean_train] * len(self.Y_test)

            working_error = 1.0 * self.calc_relative_error(y_predict_current_month) / len(self.Y_test)
            mean_estimator_error = 1.0 * self.calc_relative_error(mean_train) / len(self.Y_test)

            self.print_errors(cycle, x_1=0, x_2=0, mean_train=mean_train, y_predict_current_month=y_predict_current_month,day=day) # print prediction/error information
            self.output_file.write("error %f\n" %(working_error,))
            self.output_file.write("mean estimator error %f\n" %(mean_estimator_error,))
            self.output_file.write( "\n\n\n")

            self.error_comparison_6mo[day] = float(working_error)

    def update_model(self, month, cycle, days_of_data=None, day=None):
        """ Updates the current linear regression model by updating the training data. Also Updates the test data"""

        for i in range((month-1)*cycle, month*cycle):
            if day is None or self.X_day[i+cycle] == day:
                self.X_train_list.append(self.X[i:i+cycle])
                self.Y_train_list.append([self.X[i+cycle]])

        if days_of_data is not None and len(self.X_train_list) > days_of_data:  ## removes data when it is > sliding rule
            num_remove = len(self.X_train_list) - days_of_data
            self.X_train_list = self.X_train_list[num_remove:]
            self.Y_train_list = self.Y_train_list[num_remove:]
        self.X_train = np.matrix(self.X_train_list)
        self.Y_train = np.matrix(self.Y_train_list)

        for i in range(month*cycle, (month+1)*cycle):
            if day is None or self.X_day[i+cycle] == day:
                self.X_test_list.append(self.X[i:i+cycle])
                self.Y_test_list.append([self.X[i+cycle]])
        self.X_test = np.matrix(self.X_test_list)
        self.Y_test = np.matrix(self.Y_test_list)
        self.regr.fit(self.X_train, self.Y_train)


    def build_new_model(self,region, day=None, cycle=28, days_of_data=None, stratified=False):
        """ Creates a new linear regression model based on paramaters. Returns list of relative errors """
        
        # Initializations
        self.regr = linear_model.LinearRegression()
        self.X_train_list = []   ## training samples
        self.Y_train_list = []   ## Y_real for training
        y_predict_list_past = [] ## refers to predictions up to but not including this month
        Y_test_full_list = [] ## refers to all past values of y_test, used for output
        n = cycle 
        self.num_within = 0  ## stores number within our confidence interval
        self.total = 0  ## total number of predictions 
        total_error = 0  ## keeps running tally of squared error up to but not including this month
        month = 1 # current month (or week or day depending on cycle)
        self.error_list = [] # list of relative errors during building of model used for output
        self.mean_error_list = [] # list of relative errors of mean predictor during building of model used for output
        interval_list_low = [] # list of x_1s used for output
        interval_list_high = [] # list of x_2s used for output
        self.post_2yr_error = []
        self.post_2yr_mean_error = []


        # create file for debug information
        if days_of_data is None:
            self.output_file = open("ols_results/output_ols_%sday_%s" %(cycle, region,), 'w')
        elif day:
            self.output_file = open("ols_results/output_sliding%s_ols_%sday_%s_%s" %(days_of_data, cycle, region, day,), 'w')
        else:
            self.output_file = open("ols_results/output_ols_%sday_slide%s_%s" %(cycle, days_of_data, region,), 'w')

        while ( ((month + 2 ) * cycle) <= len(self.X) ): ## Ends when there are less than 2 months left in our data
            # initializations
            self.X_train = None
            self.X_test = None 
            self.Y_test = None
            self.Y_train = None
            self.X_test_list = []
            self.Y_test_list = []

            if days_of_data is None or days_of_data > n: # days of data referes to size of sliding rule
                n = month * cycle
            else:
                n = days_of_data

            self.update_model(month, cycle, days_of_data, day) ## Updates model for current month
            y_predict_current_month = self.regr.predict(self.X_test)
            
            if len(y_predict_list_past) > 0: # calculate mean of predictions
                mean_ = 1.0 * sum(y_predict_list_past)/len(y_predict_list_past)
            else:
                mean_ = 0

            mean_train = (1.0 * sum(self.X[0:len(self.X_train)])) / len(self.X_train) ## Finds mean of Training data
            mean_train = [mean_train] * len(self.Y_test)
            

            if len(y_predict_list_past) == 0: ## won't have MSE if we haven't predicted anything yet
                MSE = 0
            #else:
            #    to_remove = 0
            #    if days_of_data is not None:
            #        to_remove = len(y_predict_list_past) - days_of_data
            #    MSE = self.calc_mse(self.error_list[to_remove:], len(y_predict_list_past[to_remove:]))  # Calculates MSE of past estimates
            else:
                if day:
                    length = cycle/7
                else:
                    length = cycle
                MSE = mean_squared_error(y_predict_list_past[-len(self.Y_train[length:]):], self.Y_train[length:])
            sigma_ = math.sqrt(MSE) # assumes unbiased estimator


            #print debug information
            self.output_file.write("Training on days: %s-%s by learning days %s-%s\n" \
                                    %(0, month*cycle -1, month*cycle, (month+1)*cycle - 1 ))
            self.output_file.write("dim X_train: %s\n" %(self.X_train.shape,))
            self.output_file.write("dim Y_train: %s\n" %(self.Y_train.shape,))
            self.output_file.write("dim X_test: %s\n" %(self.X_test.shape,))
            self.output_file.write("dim Y_test: %s\n" %(self.Y_test.shape,))
            self.output_file.write("n: %s\n" %(n,))
            self.output_file.write("cycle: %s\n" %(cycle,))
            self.output_file.write("sigma: %s\n" %(sigma_,))
            self.output_file.write("Mean of estimates: %s\n" %(mean_,))
            self.output_file.write("Current coefficients: %s\n" %(self.regr.coef_,))
            self.output_file.write("Total Error up to %s: %s\n" %(n, total_error,))
            self.output_file.write("MSE: %s\n" %(MSE,))
            self.output_file.write("dim y_predict_current_month: %s\n" %(y_predict_current_month.shape,))
            self.output_file.write("dim y_predict_list_past: %s\n" %(len(y_predict_list_past),))
            self.output_file.write("dim X: %s\n" %(self.X_train.shape,))
            self.output_file.write("dim Y: %s\n" %(self.Y_train.shape,))
            self.output_file.write("Testing on days: %s-%s by predicting days %s-%s\n" \
                                    % (month*cycle, (month+1)*cycle - 1, (month+1)*cycle, (month+2)*cycle - 1))

            x_1, x_2 = self.calc_confidence_intervals(mean_, sigma_, cycle, day, y_predict_current_month)  # get confidence interval
            working_error = 1.0 * self.calc_relative_error(y_predict_current_month) / len(self.Y_test)
            mean_estimator_error = 1.0 * self.calc_relative_error(mean_train) / len(self.Y_test)
            
            self.print_errors(cycle, x_1, x_2, mean_train, y_predict_current_month, day) # print prediction/error information
            self.output_file.write("error %f\n" %(working_error,))
            self.output_file.write("mean estimator error %f\n" %(mean_estimator_error,))
            self.output_file.write( "\n\n\n")

            # keep tally of this information
            total_error += self.calc_squared_error(y_predict_current_month)
            y_predict_list_past += list(y_predict_current_month)
            Y_test_full_list += list(self.Y_test_list)
            month += 1
            self.error_list.append(float(working_error))
            self.mean_error_list.append(float(mean_estimator_error))
            interval_list_low += x_1
            interval_list_high += x_2
            if (month+2)*cycle >= 730:
                self.post_2yr_error.append(float(working_error))
                self.post_2yr_mean_error.append(float(mean_estimator_error))
        self.output_file.write("Percentage within range is: %s " %(1.0*self.num_within/self.total,))
        self.post_2yr_error = sum(self.post_2yr_error)/len(self.post_2yr_error)
        self.post_2yr_mean_error = sum(self.post_2yr_mean_error)/len(self.post_2yr_mean_error)

        self.output_file.write("\n%s\n" % (self.post_2yr_error))
        self.output_file.write("%s" % (self.post_2yr_mean_error,))


        if cycle != 28:  # for cycle of 7 it takes average of predicting 28days so can be directly compared to 28day model
            num_needed = 28/cycle
            self.error_list = [1.0*sum(self.error_list[i:i+num_needed])/num_needed for i in range(0, len(self.error_list), num_needed)]
        return interval_list_low, interval_list_high, y_predict_list_past, Y_test_full_list

    def print_errors(self, cycle, x_1, x_2, mean_train, y_predict_current_month, day):
        """ Prints information about predictions of models, their errors, and the confidence interval """
        if day is None: ## silly
                for i in range(0,cycle):
                    self.output_file.write( "  %s    %s <= ||pred:%s, mean_pred: %s, real: %s|| = < %s \n" \
                                    %(self.X_day[i],x_1[i], y_predict_current_month[i], mean_train[i], self.Y_test[i], x_2[i]) )
                    if not (x_1[i] == 0 and x_2[i] == 0):
                        self.total += 1
                        if self.Y_test[i] <= x_2[i] and self.Y_test[i] >= x_1[i]:
                            self.num_within += 1
        else:
            for i in range(0, len(y_predict_current_month)):
                self.output_file.write( "   %s      %s <= ||pred:%s, real: %s|| = < %s \n" \
                                %(day, x_1, y_predict_current_month[i], self.Y_test[i], x_2) )
                if not (x_1 == 0 and x_2 == 0):
                    self.total += 1
                    if self.Y_test[i] <= x_2[i] and self.Y_test[i] >= x_1[i]:
                        self.num_within += 1

    def calc_mse(self,total_error, n):
        """ Calculates MSE of predictions """
        return ( (1.0 * total_error) / n) 
        
    def calc_confidence_intervals(self,mean_, sigma_, cycle, day, y_predict_current_month):
        if sigma_ == 0:
            return [0] * len(y_predict_current_month), [0] * len(y_predict_current_month)
        """ Calculates confidence intervals for predictions """
        x_1 = [ y_predict_current_month[i] - (1.96 * sigma_) for i in range(0, len(y_predict_current_month)) ]
        x_2 = [ y_predict_current_month[i] + (1.96 * sigma_) for i in range(0, len(y_predict_current_month)) ]

        return x_1, x_2

    def calc_squared_error(self, y_predict_current_month):
        """ Calculates squared error of predictions """
        error = 0
        for i in range(len(self.Y_test)):
            error += (math.pow(1.0*self.Y_test[i] - y_predict_current_month[i], 2))
        return error

    def calc_relative_error(self, y_predict_current_month):
        """ Calculates relative error of predictions """
        error = 0
        for i in range(len(self.Y_test)):
            error += abs((1.0*self.Y_test[i] - y_predict_current_month[i])/ self.Y_test[i])
        return error

def main():
    regions = ('OregonEugeneSpringfield', 'WashingtonLongview', 'WashingtonWhatcom')
    #regions = ('WashingtonLongview',)
    #regions = ('WashingtonWhatcom',)
    #regions = ('OregonEugeneSpringfield',)
    mdl = ols_models(regions)
    mdl.build_ols_model(regions)

if __name__ == '__main__':
    main()
