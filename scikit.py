import os
import pickle
import numpy as np
import time
import pandas as pd
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
    clf.fit((train_data)[:batch_length], (train_labels)[:batch_length])
    global training 
    training = training + (time.time() - start_time)
    print("Training Time: --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    y_prediction = clf.predict((test_batch['data'])[:test_length])
    global testing
    testing = testing + (time.time() - start_time)
    print("Testing Time: --- %s seconds ---" % (time.time() - start_time))
    score = accuracy_score(y_prediction, (test_batch['labels'])[:test_length])
    global scores
    scores = scores + (score *100)

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

data_batches = []
data_batches.append(data_batch_1)
data_batches.append(data_batch_2)
data_batches.append(data_batch_3)
data_batches.append(data_batch_4)
data_batches.append(data_batch_5)

train_data = np.concatenate([batch['data'] for batch in data_batches])
train_labels = np.concatenate([batch['labels'] for batch in data_batches])

#print(len(train_data))
#print(len(train_labels))

batch_length = 25000
test_length = 10000

scores = 0
training = 0
testing = 0

for i in range(1, 6):
    #run("Decision Tree " + str(i), batch_length, test_length, tree.DecisionTreeClassifier(max_depth=3, max_features=0.1, min_samples_leaf=10, min_samples_split=10) )

    #run("Random Forest", batch_length, test_length, RandomForestClassifier(max_depth=25, max_features="sqrt", min_samples_leaf=1, min_samples_split=10))

    run("Gradiant Boosting Tree", batch_length, test_length, GradientBoostingClassifier(max_depth=5, max_features="sqrt", min_samples_leaf=10, min_samples_split=2))


print()
print("RESULTS")
print(scores/5)
print(training/5)
print(testing/5)

#run("Random Forest", batch_length, test_length, RandomForestClassifier(max_depth=25, max_features=0.1, min_samples_leaf=1, min_samples_split=10)) #230s 42,2%
#
#run("Random Forest", batch_length, test_length, RandomForestClassifier(max_depth=25, max_features="sqrt", min_samples_leaf=1, min_samples_split=10)) #45s 41,9%

#run("Gradiant Boosting Tree", batch_length, test_length, GradientBoostingClassifier(max_depth=10, max_features="sqrt", min_samples_leaf=10, min_samples_split=2)) #347s 46,9%

#run("Gradiant Boosting Tree", batch_length, test_length, GradientBoostingClassifier(max_depth=5, max_features="sqrt", min_samples_leaf=10, min_samples_split=2)) #159s 46,9%

#run("Gradiant Boosting Tree", batch_length, test_length, GradientBoostingClassifier(max_depth=5, max_features="sqrt", min_samples_leaf=10, min_samples_split=10)) #169s 46.75%

#run("Gradiant Boosting Tree", batch_length, test_length, GradientBoostingClassifier(max_depth=3, max_features="sqrt", min_samples_leaf=10, min_samples_split=10)) #91s 44%



#pickle.dump(best_estimator, open('./model.p', 'wb'))

search_space_dt = {
    "max_depth": [3, 5, 10, 25],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "max_features": [0.1, 0.5, 1.0, None, "sqrt"]
}

search_space_rf = {
    "max_depth": [3, 10, 25],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "max_features": [0.1, 0.5, "sqrt"]
}

search_space_gb = {
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf": [1, 5 ,10],
    "max_features": ["sqrt"]
}

GS_DT = GridSearchCV(estimator= tree.DecisionTreeClassifier(),
                  param_grid= search_space_dt,
                  scoring = ["r2", "neg_root_mean_squared_error"],
                  refit = "r2",
                  cv = 5,
                  verbose = 4)
GS_RF = GridSearchCV(estimator=RandomForestClassifier(),
                    param_grid= search_space_rf,
                    scoring = ["r2", "neg_root_mean_squared_error"],
                    refit = "r2",
                    cv = 5,
                    verbose = 4)

GS_GB = GridSearchCV(estimator=GradientBoostingClassifier(),
                    param_grid= search_space_gb,
                    scoring = ["r2", "neg_root_mean_squared_error"],
                    refit = "r2",
                    cv = 5,
                    verbose = 4)

#print("DT")
#GS_DT.fit((data_batch_1['data']), (data_batch_1['labels']))
#df = pd.DataFrame(GS_DT.cv_results_)
#df = df.sort_values("rank_test_r2")
#df.to_csv("cv_result_DT.csv")

#print("RF")
#GS_RF.fit((data_batch_1['data'])[:batch_length], (data_batch_1['labels'])[:batch_length])
#df = pd.DataFrame(GS_RF.cv_results_)
#df = df.sort_values("rank_test_r2")
#df.to_csv("cv_result_RF.csv")

#print("GB")
#GS_GB.fit((data_batch_1['data'])[:batch_length], (data_batch_1['labels'])[:batch_length])
#df = pd.DataFrame(GS_GB.cv_results_)
#df = df.sort_values("rank_test_r2")
#df.to_csv("cv_result_GB2.csv")

#print("DT")
#print (GS_DT.best_params_)
#print(GS_DT.best_score_)

#print("RF")
#print (GS_RF.best_params_)
#print(GS_RF.best_score_)

#print("GB")
#print (GS_GB.best_params_)
#print(GS_GB.best_score_)