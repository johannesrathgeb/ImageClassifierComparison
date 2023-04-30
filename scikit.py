import os
import pickle

#from skimage.io import imread
#from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def unpickle(file):
    import pickle
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


def svc():
    # train classifier
    classifier = SVC()

    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

    grid_search = GridSearchCV(classifier, parameters)

    grid_search.fit((data_batch_1['data'])[:500], (data_batch_1['labels'])[:500])

    # test performance
    best_estimator = grid_search.best_estimator_

    y_prediction = best_estimator.predict((test_batch['data'])[:500])

    score = accuracy_score(y_prediction, (test_batch['labels'])[:500])

    print('{}% of samples were correctly classified'.format(str(score * 100)))
    return

# prepare data
input_dir = '/home/johannes/Dokumente/GitHub/Bachelorarbeit/cifar-10-batches-py/'
#categories = ['empty', 'not_empty']


data_batch_1=unpickle(input_dir + 'data_batch_1')
data_batch_3=unpickle(input_dir + 'data_batch_3')
data_batch_2=unpickle(input_dir + 'data_batch_2')
data_batch_4=unpickle(input_dir + 'data_batch_4')
data_batch_5=unpickle(input_dir + 'data_batch_5')
test_batch=unpickle(input_dir + 'test_batch')

#print(len(data_batch_1['data']))
#print(len(data_batch_2['data']))
#print(len(data_batch_3['data']))
#print(len(data_batch_4['data']))
#print(len(data_batch_5['data']))
#print(len(test_batch['data']))


data = []
labels = []
"""for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)"""

data = np.asarray((data_batch_1['data'])[:500])
labels = np.asarray((data_batch_1['labels'])[:500])

print(len(data))
#svc()
dt()
rf()
gbt()
# train / test split
#x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)




#pickle.dump(best_estimator, open('./model.p', 'wb'))