import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def dt():
    print('Decision Tree')
    clf = tree.DecisionTreeClassifier()
    clf.fit((data_batch_1['data'])[:500], (data_batch_1['labels'])[:500])
    y_prediction = clf.predict((test_batch['data'])[:500])

    score = accuracy_score(y_prediction, (test_batch['labels'])[:500])

    print('{}% of samples were correctly classified'.format(str(score * 100)))
    return

def rf():
    print('Random Forest')
    clf = RandomForestClassifier()
    clf.fit((data_batch_1['data'])[:500], (data_batch_1['labels'])[:500])
    y_prediction = clf.predict((test_batch['data'])[:500])

    score = accuracy_score(y_prediction, (test_batch['labels'])[:500])

    print('{}% of samples were correctly classified'.format(str(score * 100)))
    return

def gbt():
    print('Gradient Boosting Tree')
    clf = GradientBoostingClassifier()
    clf.fit((data_batch_1['data'])[:500], (data_batch_1['labels'])[:500])
    y_prediction = clf.predict((test_batch['data'])[:500])

    score = accuracy_score(y_prediction, (test_batch['labels'])[:500])

    print('{}% of samples were correctly classified'.format(str(score * 100)))
    return

# prepare data
input_dir = '/home/johannes/Dokumente/GitHub/Bachelorarbeit/cifar-10-batches-py/'

data_batch_1=unpickle(input_dir + 'data_batch_1')
data_batch_3=unpickle(input_dir + 'data_batch_3')
data_batch_2=unpickle(input_dir + 'data_batch_2')
data_batch_4=unpickle(input_dir + 'data_batch_4')
data_batch_5=unpickle(input_dir + 'data_batch_5')
test_batch=unpickle(input_dir + 'test_batch')

data = []
labels = []
data = np.asarray((data_batch_1['data'])[:500])
labels = np.asarray((data_batch_1['labels'])[:500])

print(len(data))
dt()
rf()
gbt()
#pickle.dump(best_estimator, open('./model.p', 'wb'))