import numpy as np
import csv
import sys
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def ols_stuffs(region):
    regr = linear_model.LinearRegression()
    input_file = csv.DictReader(open("generated_csv/Data%s.csv" %(region,)))
    X = []
    X_train_list = []
    Y_train_list = []
    n = 28
    total_error = 0
    for line in input_file:
        X.append(int(line['Patient Count']))
    for month in range(0,29):
    #for month in range(0,2):
        X_test_list = []
        Y_test_list = []
        print "Training on days: %s-%s by learning days %s-%s" %(0, month*28 + 27, month*28 + 28, month*28 + 27 + 28)
        for i in range(month*28, month*28 + 28):
            X_train_list.append(X[i:i+28])
            Y_train_list.append([X[i+28]])
        X_train = np.matrix(X_train_list)
        Y_train = np.matrix(Y_train_list)
        print "dim X: %s" %(X_train.shape,)
        print "dim Y: %s" %(Y_train.shape,)
        print "Testing on days: %s-%s by predicting days %s-%s" %((month+1)*28, (month+1)*28 + 27, (month+1)*28 + 28, (month+1)*28 + 28 + 28 )
        for i in range((month+1)*28, (month+1)*28 + 28):
            X_test_list.append(X[i:i+28])
            Y_test_list.append([X[i+28]])
        X_test = np.matrix(X_test_list)
        Y_test = np.matrix(Y_test_list)
        print "dim X_test: %s" %(X_test.shape,)
        print "dim Y_test: %s" %(Y_test.shape,)
        regr.fit(X_train, Y_train)
        y_predict = regr.predict(X_test)
        #print y_predict
        #print Y_test
        for i in range(0,28):
            print y_predict[i], Y_test[i]
        working_error = calc_relative_error(Y_test, y_predict)
        print working_error
        total_error += working_error
        n+=28
        print "\n\n\n"
    total_error = 1.0*total_error / n
    print "Total Error for %s: %s" % (region, total_error,)

def calc_relative_error(Y_test, y_predict):
    error = 0
    for i in range(len(Y_test)):
        error += abs((1.0*Y_test[i] - y_predict[i])/ Y_test[i])
    return error

def main():
    #regions = ('OregonEugeneSpringfield', 'WashingtonLongview', 'WashingtonWhatcom')
    #regions = ('WashingtonLongview',)
    regions = ('WashingtonWhatcom',)
    #regions = ('OregonEugeneSpringfield',)
    for region in regions:
        ols_stuffs(region)

if __name__ == '__main__':
    main()
