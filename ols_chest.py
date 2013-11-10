import numpy as np
import csv
import sys
import math
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt


def build_ols_model(regions):
    for region in regions:
        X = []
        X_day = []
        error_list = []
        error_lists = []
        input_file = csv.DictReader(open("generated_csv/Data%s.csv" %(region,)))
        for line in input_file:
            X.append(int(line['Patient Count']))
            X_day.append(line['Day'])
        for cycle in (1, 7, 28): ## add back 1 and make sure this works same as with hardcoded months
            error_list = ols_stuffs(region=region, cycle=cycle, X=X)
            error_lists.append(error_list)
            #plt.plot(error_list)
        #plt.legend(['Update 1day', 'Update 7 days', 'Update 28 days'])
        #plt.title("%s" % region)
        #plt.show()
        #plt.plot(error_list) # 28 days model
        for day in ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'):
            #error_lists.append(ols_stuffs(region=region, day=day, cycle=28, X=X, X_day=X_day))
            ols_stuffs(region=region, day=day, cycle=28, X=X, X_day=X_day)
        for days_of_data in (420, 630):
            error_list = ols_stuffs(region=region, cycle=28, X=X, days_of_data=days_of_data)
            error_lists.append(error_list)
            #plt.plot(error_list)
        #plt.legend(['Full data', '420 sliding rule', '630 sliding rule'])
        #plt.title("%s" % region)
        #plt.show()
        write_errors_to_csv(region, error_lists)


def write_errors_to_csv(region, error_lists):
    output_file = open("ols_results/output_comparison_%s.csv" %(region,), 'w')
    writer = csv.writer(output_file)
    output_columns = ('1day', '7day', '28day', '420slide', '630slide',)
    #output_columns = ('1day', '7day', '28day', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
     #                   '420slide', '630slide',)
    writer.writerow(output_columns)
    for i in range(len(error_lists[2])):
        writer.writerow([error_lists[j][i] for j in range(len(error_lists))])
    
    

def ols_stuffs(region, day=None, cycle=28, X=None, X_day=None, days_of_data=None):
    regr = linear_model.LinearRegression()
    
    if day:
        output_file = open("ols_results/output_ols_%sday_%s_%s" %(cycle, region, day,), 'w')
    else:
        if days_of_data is None:
            output_file = open("ols_results/output_ols_%sday_%s" %(cycle, region,), 'w')
        else:
            output_file = open("ols_results/output_ols_%sday_slide%s_%s" %(cycle, days_of_data, region,), 'w')
    
    X_train_list = []   ## training samples
    Y_train_list = []   ## Y_real for training
    y_predict_list_past = [] ## refers to predictions up to but not including this month
    Y_test_full_list = []
    n = cycle
    num_within = 0  ## stores number within our confidence interval
    total = 0   
    total_error = 0  ## keeps running tally of squared error up to but not including this month
    month = 1
    error_list = []
    while ( ((month + 2 ) * cycle) <= len(X) ):
        if days_of_data is None or days_of_data > n:
            n = month * cycle
        else:
        #elif days_of_data <= n:
            n = days_of_data

        X_test_list = []
        Y_test_list = []
        output_file.write("Training on days: %s-%s by learning days %s-%s\n" \
                                %(0, month*cycle -1, month*cycle, (month+1)*cycle - 1 )) ## todo handle sliding rule
        
        for i in range((month-1)*cycle, month*cycle):
            X_train_list.append(X[i:i+cycle])
            Y_train_list.append([X[i+cycle]])
        if days_of_data is not None and len(X_train_list) > days_of_data:  ## removes data when it is larger than sliding rule
            num_remove = len(X_train_list) - days_of_data
            X_train_list = X_train_list[num_remove:]
            Y_train_list = Y_train_list[num_remove:]
        X_train = np.matrix(X_train_list)
        Y_train = np.matrix(Y_train_list)

        output_file.write( "dim X: %s\n" %(X_train.shape,))
        output_file.write("dim Y: %s\n" %(Y_train.shape,))
        output_file.write("Testing on days: %s-%s by predicting days %s-%s\n" \
                                %(month*cycle, (month+1)*cycle - 1, (month+1)*cycle, (month+2)*cycle - 1))
        for i in range(month*cycle, (month+1)*cycle):
            try:
                if day is None or X_day[i+cycle] == day:
                    X_test_list.append(X[i:i+cycle])
                    Y_test_list.append([X[i+cycle]])
            except:
                import pdb; pdb.set_trace() 
            
        X_test = np.matrix(X_test_list)
        Y_test = np.matrix(Y_test_list)
        
        if len(y_predict_list_past) >0:
            mean_ = 1.0* sum(y_predict_list_past)/len(y_predict_list_past)
        else:
            mean_ = 0
        mean_train = [1.0*sum(i)/cycle for i in X_train_list]
        
        if len(y_predict_list_past) == 0: 
            MSE = 0
        elif day is None:
            MSE = calc_mse(total_error, len(y_predict_list_past)) # changed from n for monday predictor
        else:
            MSE = calc_mse(total_error, len(y_predict_list_past)) # changed from n for monday predictor


        sigma_ = math.sqrt(MSE) # assumes unbiased estimator
        
        regr.fit(X_train, Y_train)
        y_predict_current_month = regr.predict(X_test)
        
        # Mostly printed to understand where calculations come from/ debugging
        output_file.write("dim X_test: %s\n" %(X_test.shape,))
        output_file.write("dim Y_test: %s\n" %(Y_test.shape,))
        output_file.write("n: %s\n" %(n,))
        output_file.write("cycle: %s\n" %(cycle,))
        output_file.write("sigma: %s\n" %(sigma_,))
        output_file.write("Mean of estimates: %s\n" %(mean_,))
        output_file.write("Total Error up to %s: %s\n" %(n, total_error,))
        output_file.write("MSE: %s\n" %(MSE,))
        output_file.write("dim y_predict_current_month: %s\n" %(y_predict_current_month.shape,))
        output_file.write("dim y_predict_list_past: %s\n" %(len(y_predict_list_past),))

        x_1, x_2 = calc_confidence_intervals(mean_, sigma_, cycle, day)

        if day is None:
            for i in range(0,cycle):
                output_file.write( " %s <= ||pred:%s, mean_pred: %s, real: %s|| = < %s \n" \
                                %(x_1[i], y_predict_current_month[i], mean_train[i], Y_test[i], x_2[i]) )
                total += 1
                if Y_test[i] <= x_2[i] and Y_test[i] >= x_1[i]:
                    num_within += 1
        else:
            for i in range(0,cycle/7):
                output_file.write( " %s <= ||pred:%s, real: %s|| = < %s \n" \
                                %(x_1[i], y_predict_current_month[i], Y_test[i], x_2[i]) )
                total += 1
                if Y_test[i] <= x_2[i] and Y_test[i] >= x_1[i]:
                    num_within += 1
        if day is None:
            working_error = 1.0 * calc_relative_error(Y_test, y_predict_current_month) / cycle
            mean_estimator_error = 1.0 * calc_relative_error(Y_test, mean_train) / cycle
        else:
            working_error = 1.0 * calc_relative_error(Y_test, y_predict_current_month) / 4
            mean_estimator_error = 1.0 * calc_relative_error(Y_test, mean_train) / 4

        total_error += calc_squared_error(Y_test, y_predict_current_month) 
        
        output_file.write("error %f\n" %(working_error,))
        output_file.write("mean estimator error %f\n" %(mean_estimator_error,))
        output_file.write( "\n\n\n")
        
        y_predict_list_past += list(y_predict_current_month)
        Y_test_full_list += list(Y_test_list)
        month += 1
        error_list.append(float(working_error))
    output_file.write("Percentage within range is: %s " %(1.0*num_within/total,))
    if cycle != 28:
        num_needed = 28/cycle
        error_list = [1.0*sum(error_list[i:i+num_needed])/num_needed for i in range(0, len(error_list), num_needed)]
    return error_list
##average error for 28 days based on 1/7 day rebuild?

def calc_mse(total_error, n):
    return ( (1.0 * total_error) / n) 
    
def calc_confidence_intervals(mean_, sigma_, cycle, day):
# mean_ +- (1.96 * sigma) where sigma is std of estimator and mean is mean of estimator
    if day is None:
        x_1 = [ mean_ - (1.96 * sigma_) for i in range(0, cycle) ]
        x_2 = [ mean_ + (1.96 * sigma_) for i in range(0, cycle) ]
    else:
        x_1 = [ mean_ - (1.96 * sigma_) for i in range(0, cycle/7) ]
        x_2 = [ mean_ + (1.96 * sigma_) for i in range(0, cycle/7) ]

    return x_1, x_2

def calc_squared_error(Y_test, y_predict_current_month):
    error = 0
    for i in range(len(Y_test)):
        error += (math.pow(1.0*Y_test[i] - y_predict_current_month[i], 2))
    return error

def calc_relative_error(Y_test, y_predict_current_month):
    error = 0
    for i in range(len(Y_test)):
        try:
            error += abs((1.0*Y_test[i] - y_predict_current_month[i])/ Y_test[i])
        except:
            import pdb; pdb.set_trace() 
    return error

def main():
    regions = ('OregonEugeneSpringfield', 'WashingtonLongview', 'WashingtonWhatcom')
    #regions = ('WashingtonLongview',)
    #regions = ('WashingtonWhatcom',)
    #regions = ('OregonEugeneSpringfield',)
    build_ols_model(regions)

if __name__ == '__main__':
    main()
