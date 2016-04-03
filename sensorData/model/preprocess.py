import csv
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pandas.tools.plotting import lag_plot
import pandas
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import acf
from dim_reduction import *
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import zero_one_loss
from sklearn.multiclass import OneVsOneClassifier

from os import listdir
from os.path import isfile, join

# This function is used to load data
def loadData(filename):
    f = None
    data = []
    try:
        f = open(filename, 'rb')
        data = csv.reader(f)
        data = np.array(list(data)).astype(np.float)
    finally:
        if f != None:
            f.close()
        return data

#This function is used to caculate the autocorrelation.
def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    assert np.allclose(r, np.array([(x[:n - k] * x[-(n - k):]).sum() for k in range(n)]))
    result = r / (variance * (np.arange(n, 0, -1)))
    return result

#extract features. Applying the Autoregressive algorithme to each window(600 samples = 100s). Use the coefficients of the autoregressive to respresnt the features.
def features(data, order, window, slip):
    x = list(data[:, 0])
    x_trans = np.array([x[i:i + window] for i in range(0, len(x) - window, slip)])
    mod_paras_x = [AR(series).fit(order).params for series in x_trans]

    y = list(data[:, 1])
    y_trans = np.array([y[i:i + window] for i in range(0, len(y) - window, slip)])
    mod_paras_y = [AR(series).fit(order).params for series in y_trans]

    z = list(data[:, 2])
    z_trans = np.array([z[i:i + window] for i in range(0, len(z) - window, slip)])
    mod_paras_z = [AR(series).fit(order).params for series in z_trans]

    mod_paras = np.concatenate((mod_paras_x, mod_paras_y, mod_paras_z), 1)
    return mod_paras

# generate features for all samples.
def allSamples(path, category, label_dict, order=2, window=400, slip=200):
    walkingPath = path + category
    onlyfiles = [join(walkingPath, f) for f in listdir(walkingPath) if isfile(join(walkingPath, f))]
    print onlyfiles
    mod_paras = []
    for filename in onlyfiles:
        data = loadData(filename)
        temp = features(data, order, window, slip)
        if mod_paras == []:
            mod_paras = temp
        else:
            mod_paras = np.concatenate((mod_paras, temp), 0)
    Y = np.ones(shape=(len(mod_paras))) * label_dict[category]
    return mod_paras, Y




order = 6
window = 600
slip = 150

label_dict = {"walking": 0, "sitting": 1, "subway": 2}

# This part is used to compare the score of different parameters, suche as window size,slip size, autoregressive orders.
path = './'
orders = range(4, 5)
slips = [150]
score = []
for slip in slips:
    train = []
    label = []
    for category in label_dict.keys():
        X, Y = allSamples(path, category, label_dict, order, window, slip)
        if train == []:
            train = X
            label = Y
        else:
            train = np.concatenate((train, X), 0)
            label = np.concatenate((label, Y), 0)
    #train = dim_reduction_PCA(train,0.999)
    X_train, X_test, Y_train, Y_test = train_test_split(train, label, test_size=0.4, random_state=42)
    C = 1.0
    multiclassifier = OneVsOneClassifier(svm.SVC(kernel="rbf",gamma=0.7,C=C)).fit(X_train, Y_train)
    score.append(multiclassifier.score(X_test, Y_test))
# svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
# rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, Y_train)
# poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, Y_train)

# print score
# plt.figure("Score-order")
# plt.plot(slips,score)
# plt.show()

#Use validate data to test the model.
path = "./"
label_dict = {"test":0}
X_test,Y_test = allSamples(path,"test",label_dict,order,window,slip)
#print X_test,Y_test
print multiclassifier.predict(X_test)
print multiclassifier.score(X_test,Y_test)
#X_test = dim_reduction_PCA(X_test,0.99)
#plot_data(X_test,Y_test,"PLOT",mirror=1)

# y = pandas.Series(x)
# plt.figure()
# lag_plot(y,marker='+',color='gray')
# plt.show()

# autor = estimated_autocorrelation(x)
# autor = autor[autor.size/2:]
#
# timestamp = 1order5order397880668
# date = datetime.datetime.fromtimestamp(timestamp/1e3)
# datetime.datetime(2016, 2, 2, 8, 2order, order0, 668000)
# autor, ci, Q, pvalue = acf(x, nlags=order,
# confint=95, qstat=True,
# unbiased=True)
#
# k = np.array([i for i in range(len(autor))])
# plt.figure()
# plt.plot(k,autor)
# plt.xlabel('k (s)')
# plt.ylabel('autocorrelation')
# plt.show()
