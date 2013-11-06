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
        input_file = csv.DictReader(open("generated_csv/Data%s.csv" %(region,)))
        for line in input_file:
            X.append(int(line['Patient Count']))
            X_day.append(line['Day'])
        for cycle in (28, 7, 1): ## add back 1 and make sure this works same as with hardcoded months
            ols_stuffs(region=region, cycle=cycle, X=X)
        for day in ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'):
            ols_stuffs(region=region, day=day, cycle=28, X=X, X_day=X_day)


def ols_stuffs(region, day=None, cycle=28, X=None, X_day=None):
    regr = linear_model.LinearRegression()
    
    if day:
        output_file = open("ols_results/output_ols_%sday_%s_%s" %(cycle, region, day,), 'w')
    else:
        output_file = open("ols_results/output_ols_%sday_%s" %(cycle, region,), 'w')
    
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
    month = 1
    while ( ((month + 2 ) * cycle) <= len(X) ):
        X_test_list = []
        Y_test_list = []
        output_file.write("Training on days: %s-%s by learning days %s-%s\n" \
                                %(0, month*cycle -1, month*cycle, (month+1)*cycle - 1 ))
        
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
                if not day or X_day[i+cycle] == day:
                    X_test_list.append(X[i:i+cycle])
                    Y_test_list.append([X[i+cycle]])
            except:
                import pdb; pdb.set_trace() 
            
        X_test = np.matrix(X_test_list)
        Y_test = np.matrix(Y_test_list)

        mean_predict = 1.0* sum(y_predict_list)/n
        mean_train = [1.0*sum(i)/cycle for i in X_train_list]
        MSE = calc_mse(total_error, cycle*month)
        sigma_ = math.sqrt(MSE) # assumes unbiased estimator
        regr.fit(X_train, Y_train)
        y_predict = regr.predict(X_test)

        output_file.write("dim X_test: %s\n" %(X_test.shape,))
        output_file.write("dim Y_test: %s\n" %(Y_test.shape,))
        output_file.write("n: %s\n" %(cycle*month,))
        output_file.write("cycle: %s\n" %(cycle,))
        output_file.write("sigma: %s\n" %(sigma_,))
        #output_file.write("mean estimator: %s\n" %(mean_train,))
        output_file.write("dim y_predict: %s\n" %(y_predict.shape,))

        x_1, x_2 = calc_confidence_intervals(sigma_, y_predict, cycle*month, cycle, day)

        if day is None:
            for i in range(0,cycle):
                output_file.write( " %s <= ||pred:%s, mean_pred: %s, real: %s|| = < %s \n" \
                                %(x_1[i], y_predict[i], mean_train[i], Y_test[i], x_2[i]) )
                total += 1
                if Y_test[i] <= x_2[i] and Y_test[i] >= x_1[i]:
                    num_within += 1
        else:
            for i in range(0,cycle/7):
                output_file.write( " %s <= ||pred:%s, real: %s|| = < %s \n" \
                                %(x_1[i], y_predict[i], Y_test[i], x_2[i]) )
                total += 1
                if Y_test[i] <= x_2[i] and Y_test[i] >= x_1[i]:
                    num_within += 1

        working_error = 1.0 * calc_relative_error(Y_test, y_predict) / cycle
        mean_estimator_error = 1.0 * calc_relative_error(Y_test, mean_train) / cycle
        total_error += calc_squared_error(Y_test, y_predict) 
        
        output_file.write("error %f\n" %(working_error,))
        output_file.write("mean estimator error %f\n" %(mean_estimator_error,))
        output_file.write( "\n\n\n")
        
        y_predict_list += list(y_predict)
        Y_test_full_list += list(Y_test_list)
        month += 1 
    output_file.write("Percentage within range is: %s " %(1.0*num_within/total,))
##average error for 28 days based on 1/7 day rebuild?

def calc_mse(total_error, n):
    return ( (1.0 * total_error) / n) 
    
def calc_confidence_intervals(sigma_, y_predict, n, cycle, day):
# mean_ +- (1.96 * sigma) where sigma is std of estimator and mean is mean of estimator
    if day is None:
        x_1 = [ y_predict[i] - (1.96 * sigma_) for i in range(0, cycle) ]
        x_2 = [ y_predict[i] + (1.96 * sigma_) for i in range(0, cycle) ]
    else:
        x_1 = [ y_predict[i] - (1.96 * sigma_) for i in range(0, cycle/7) ]
        x_2 = [ y_predict[i] + (1.96 * sigma_) for i in range(0, cycle/7) ]

    return x_1, x_2

def calc_squared_error(Y_test, y_predict):
    error = 0
    for i in range(len(Y_test)):
        error += (math.pow(1.0*Y_test[i] - y_predict[i], 2))
    return error

def calc_relative_error(Y_test, y_predict):
    error = 0
    for i in range(len(Y_test)):
        try:
            error += abs((1.0*Y_test[i] - y_predict[i])/ Y_test[i])
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
