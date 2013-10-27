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
    if model_type:
        input_file = csv.DictReader(open("generated_csv/Data%s_%s.csv" %(region,model_type,)))
        output_file = open("ols_results/output_ols_%s_%s" %(region, model_type,), 'w')
    else:
        input_file = csv.DictReader(open("generated_csv/Data%s.csv" %(region,)))
        output_file = open("ols_results/output_ols_%s" %(region,), 'w')
    X = []
    X_train_list = []
    Y_train_list = []
    y_predict_list = []
    y_list = []
    Y_test_full_list = []
    n = cycle
    total_error = 0
    for line in input_file:
        X.append(int(line['Patient Count']))
    for month in range(0,num_months):
        X_test_list = []
        Y_test_list = []
        output_file.write("Training on days: %s-%s by learning days %s-%s\n" %(0, month*n + n-1, month*n + n, month*n + n-1 + n))
        for i in range(month*n, month*n + n):
            X_train_list.append(X[i:i+n])
            Y_train_list.append([X[i+n]])
        X_train = np.matrix(X_train_list)
        Y_train = np.matrix(Y_train_list)
        output_file.write( "dim X: %s\n" %(X_train.shape,))
        output_file.write("dim Y: %s\n" %(Y_train.shape,))
        output_file.write("Testing on days: %s-%s by predicting days %s-%s\n" \
                                %((month+1)*n, (month+1)*n + n-1, (month+1)*n + n, (month+1)*n + n + n ))
        for i in range((month+1)*n, (month+1)*n + n):
            X_test_list.append(X[i:i+n])
            Y_test_list.append([X[i+n]])
        X_test = np.matrix(X_test_list)
        Y_test = np.matrix(Y_test_list)
        output_file.write("dim X_test: %s\n" %(X_test.shape,))
        output_file.write("dim Y_test: %s\n" %(Y_test.shape,))
        regr.fit(X_train, Y_train)
        y_predict = regr.predict(X_test)
        output_file.write("dim y_predict: %s\n" %(y_predict.shape,))
        MSE = mean_squared_error(Y_test, y_predict)
        x_1, x_2 = calc_confidence_intervals(MSE, Y_test, y_predict)
        for i in range(0,n):
            output_file.write( " %s ||%s, %s|| %s \n" %(x_1[i], y_predict[i], Y_test[i], x_2[i]) )

        working_error = 1.0 * calc_relative_error(Y_test, y_predict) / n
        output_file.write("error %f\n" %(working_error,))
        #total_error += working_error
        #n+=28
        output_file.write( "\n\n\n")
        y_predict_list += list(y_predict)
        Y_test_full_list += list(Y_test_list)
    #total_error = 1.0*total_error / n
    #plt.scatter(range(28, len(y_predict_list)+28), y_predict_list, color='black')
    #plt.scatter(range(28, len(y_predict_list)+28), Y_test_full_list, color='red')
    #plt.show()
    #output_file.write( "Total Error for %s: %s" % (region, total_error,))

def calc_confidence_intervals(MSE, Y_test, y_predict):
    sigma = math.sqrt(MSE)
    x_1 = [ y_predict[i] - (1.96 * (sigma/math.sqrt(28))) for i in range(0, len(y_predict)) ]
    x_2 = [ y_predict[i] + (1.96 * (sigma/math.sqrt(28))) for i in range(0, len(y_predict)) ]

    return x_1, x_2

def calc_relative_error(Y_test, y_predict):
    error = 0
    for i in range(len(Y_test)):
        error += abs((1.0*Y_test[i] - y_predict[i])/ Y_test[i])
    return error

def main():
    regions = ('OregonEugeneSpringfield', 'WashingtonLongview', 'WashingtonWhatcom')
    #regions = ('WashingtonLongview',)
    #regions = ('WashingtonWhatcom',)
    #regions = ('OregonEugeneSpringfield',)
    for region in regions:
        ols_stuffs(region)
    
    for model_type in ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'weekday', 'weekend', ):
        for region in regions:
            ols_stuffs(region, model_type, 29, 4 )


if __name__ == '__main__':
    main()
