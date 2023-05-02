import os
import pickle
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def run(name ,batch_length, test_length, clf):
    print(name)
    start_time = time.time()
    clf.fit((data_batch_1['data'])[:batch_length], (data_batch_1['labels'])[:batch_length])
    y_prediction = clf.predict((test_batch['data'])[:test_length])
    print("--- %s seconds ---" % (time.time() - start_time))
    score = accuracy_score(y_prediction, (test_batch['labels'])[:test_length])

    print('{}% of samples were correctly classified'.format(str(score * 100)))
    return

# prepare data
input_dir = 'C:/Users/jrath/Documents/PickYourSquad/ImageClassifierComparison/cifar-10-batches-py/'

data_batch_1=unpickle(input_dir + 'data_batch_1')
data_batch_3=unpickle(input_dir + 'data_batch_3')
data_batch_2=unpickle(input_dir + 'data_batch_2')
data_batch_4=unpickle(input_dir + 'data_batch_4')
data_batch_5=unpickle(input_dir + 'data_batch_5')
test_batch=unpickle(input_dir + 'test_batch')


batch_length = 10000
test_length = 10000

#run("Decision Tree", batch_length, test_length, tree.DecisionTreeClassifier(max_depth=None) )

#run("Random Forest", batch_length, test_length, RandomForestClassifier(max_depth=None))

#run("Gradiant Boosting Tree", batch_length, test_length, GradientBoostingClassifier(max_features='sqrt', max_depth=None))

#pickle.dump(best_estimator, open('./model.p', 'wb'))



search_space_dt = {
    "max_depth": [3, 5, 10, 25, 50],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": [0.1, 0.2, 0.5, 1.0]
}

GS_DT = GridSearchCV(estimator= tree.DecisionTreeClassifier(),
                  param_grid= search_space_dt,
                  scoring = ["r2", "neg_root_mean_squared_error"],
                  refit = "r2",
                  cv = 5,
                  verbose = 4)

GS_DT.fit((data_batch_1['data']), (data_batch_1['labels']))

#print (GS_DT.best_estimator_)
print (GS_DT.best_params_)
print(GS_DT.best_score_)