import numpy as np
import csv
import sys
import math
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt

def ols_stuffs(region, model_type=None, num_months=29, cycle=28):
    regr = linear_model.LinearRegression()
    if cycle < 28:
        return
    if model_type:
        input_file = csv.DictReader(open("generated_csv/Data%s_%s.csv" %(region,model_type,)))
        output_file = open("ols_results/output_ols_%sday_%s_%s" %(cycle, region, model_type,), 'w')
    else:
        input_file = csv.DictReader(open("generated_csv/Data%s.csv" %(region,)))
        output_file = open("ols_results/output_ols_%sday_%s" %(cycle, region,), 'w')
    X = []
    X_train_list = []
    Y_train_list = []
    y_predict_list = []
    y_list = []
    Y_test_full_list = []
    n = cycle
    num_within = 0
    total = 0
    total_error = 0
    n = cycle
    for line in input_file:
        X.append(int(line['Patient Count']))
    month = 1
    while ( ((month + 2 ) * cycle) <= len(X) ):
        #for month in range(0,num_months): ## don't require this, use something like while month or n <= something
        X_test_list = []
        Y_test_list = []
        output_file.write("Training on days: %s-%s by learning days %s-%s\n" %(0, month*cycle -1, month*cycle, (month+1)*cycle - 1 ))
        for i in range((month-1)*cycle, month*cycle):
            X_train_list.append(X[i:i+cycle])
            Y_train_list.append([X[i+cycle]])
        X_train = np.matrix(X_train_list)
        Y_train = np.matrix(Y_train_list)

        output_file.write( "dim X: %s\n" %(X_train.shape,))
        output_file.write("dim Y: %s\n" %(Y_train.shape,))
        output_file.write("Testing on days: %s-%s by predicting days %s-%s\n" \
                                %(month*cycle, (month+1)*cycle - 1, (month+1)*cycle, (month+2)*cycle - 1))
        for i in range(month*cycle, (month+1)*cycle):
            try:
                X_test_list.append(X[i:i+cycle])
                Y_test_list.append([X[i+cycle]])
            except:
                import pdb; pdb.set_trace() 
            
        X_test = np.matrix(X_test_list)
        Y_test = np.matrix(Y_test_list)
        output_file.write("dim X_test: %s\n" %(X_test.shape,))
        output_file.write("dim Y_test: %s\n" %(Y_test.shape,))
        output_file.write("n: %s\n" %(cycle*month,))
        output_file.write("cycle: %s\n" %(cycle,))
        #mean_ = 1.0* sum(y_predict_list)/n
        MSE = calc_mse(total_error, cycle*month)
        sigma_ = math.sqrt(MSE) # assumes unbiased estimator
        output_file.write("sigma: %s\n" %(sigma_,))
        #output_file.write("mean estimator: %f\n" %(mean_,))
        
        regr.fit(X_train, Y_train)
        y_predict = regr.predict(X_test)
        output_file.write("dim y_predict: %s\n" %(y_predict.shape,))

        x_1, x_2 = calc_confidence_intervals(sigma_, y_predict, cycle*month, cycle)
        for i in range(0,cycle):
            output_file.write( " %s <= ||pred:%s, real: %s|| = < %s \n" %(x_1[i], y_predict[i], Y_test[i], x_2[i]) )
            total += 1
            if Y_test[i] <= x_2[i] and Y_test[i] >= x_1[i]:
                num_within += 1
        working_error = 1.0 * calc_relative_error(Y_test, y_predict) / cycle
        total_error += calc_squared_error(Y_test, y_predict) 
        output_file.write("error %f\n" %(working_error,))
        output_file.write( "\n\n\n")
        y_predict_list += list(y_predict)
        Y_test_full_list += list(Y_test_list)
        month += 1 
    output_file.write("Percentage within range is: %s " %(1.0*num_within/total,))

def calc_mse(total_error, n):
    return ( (1.0 * total_error) / n) 
    
    
def calc_confidence_intervals(sigma_, y_predict, n, cycle):
    x_1 = [ y_predict[i] - (1.96 * sigma_) for i in range(0, cycle) ]
    x_2 = [ y_predict[i] + (1.96 * sigma_) for i in range(0, cycle) ]

    return x_1, x_2

def calc_squared_error(Y_test, y_predict):
    error = 0
    for i in range(len(Y_test)):
        error += (math.pow(1.0*Y_test[i] - y_predict[i], 2))
    return error

def calc_relative_error(Y_test, y_predict):
    error = 0
    for i in range(len(Y_test)):
        error += abs((1.0*Y_test[i] - y_predict[i])/ Y_test[i])
    return error

def main():
    #regions = ('OregonEugeneSpringfield', 'WashingtonLongview', 'WashingtonWhatcom')
    #regions = ('WashingtonLongview',)
    #regions = ('WashingtonWhatcom',)
    regions = ('OregonEugeneSpringfield',)
    for cycle in (28, 7): ## add back 1 and make sure this works same as with hardcoded months
        for region in regions:
            ols_stuffs(region, num_months =100, cycle=cycle)
    
    #for model_type in ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'weekday', 'weekend', ):
    #    for region in regions:
    #        ols_stuffs(region, model_type, 29, 4 )


if __name__ == '__main__':
    main()
