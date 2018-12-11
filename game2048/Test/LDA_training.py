import  pandas as pd
import numpy as np
import csv
from sklearn.externals import joblib


from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 0: left, 1: down, 2: right, 3: up

csv_data = pd.read_csv('Train.csv')
csv_data = csv_data.values
board_data = csv_data[:,0:16]
print(board_data[0:4,:])
direction_data = csv_data[:,16]

# print(type(board_data))
X = np.int32(board_data)/11.0
Y = np.int32(direction_data)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=20000)
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, Y_train)
train_accuracy = clf.score(X_train, Y_train)
test_accuracy = clf.score(X_test, Y_test)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))
joblib.dump(clf, 'LDA_model.pkl')