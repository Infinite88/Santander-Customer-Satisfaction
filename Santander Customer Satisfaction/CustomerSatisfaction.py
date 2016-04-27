
# Load Libraries
from __future__ import absolute_import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn.feature_selection import SelectFromModel
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model

from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from itertools import izip

# Load Data Set to train
def loadData(data):
    dataDf = pd.read_csv(data)
    numCustomers = dataDf.shape[0]
    numfeatures = dataDf.shape[1]
    
    print u'Total number of customers: {}'.format(numCustomers)
    print u'Total number of features: {}'.format(numfeatures)
    
    return dataDf

train = loadData(u'data/train.csv')
test = loadData(u'data/test.csv')

# Removes constan columns with a std of zero
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

# Removes columns that are duplicates 
cols = train.columns
for i in xrange(len(cols)-1):
    v = train[cols[i]].values
    for j in xrange(i+1,len(cols)):
        if np.array_equal(v,train[cols[j]].values):
            remove.append(cols[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True) 

X_train = train.drop([u'TARGET', u'ID'], axis=1)
y_train = train.TARGET.values

test_id = test.ID
test = test.drop([u"ID"],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20)

featureSelect_clf = ExtraTreesClassifier(random_state=1729)
selector = featureSelect_clf.fit(X_train, y_train)

# plot most important features
feat_imp = pd.Series(featureSelect_clf.feature_importances_, index = X_train.columns.values).sort_values(ascending=False)
feat_imp[:50].plot(kind=u'bar', title=u'Feature Importances according to ExtraTreesClassifier', figsize=(12, 8))
plt.ylabel(u'Feature Importance Score')
plt.subplots_adjust(bottom=0.3)
plt.savefig(u'feature_imporatnce.png')
plt.show()

# Transforms data to the most important features 
featImports = SelectFromModel(selector, prefit=True)

X_train = featImports.transform(X_train)
X_test = featImports.transform(X_test)
test = featImports.transform(test)

print X_train.shape, test.shape

def trainModel(clf, X_train, y_train):
    print u"------------------------------------------"
    print u"\nClassifier: {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    scores = cross_validation.cross_val_score(clf, X_train, y_train, scoring=u'roc_auc', cv=10) 
    end = time.time()
    print u"time (secs): {:.3f}".format(end - start)
    return scores.mean()

gnb = GaussianNB()
tree = RandomForestClassifier()
dTree = DecisionTreeClassifier()
logReg = linear_model.LogisticRegression()
gboost = ensemble.GradientBoostingClassifier()
xgb = xgb.XGBClassifier()
ada = AdaBoostClassifier()

clfNames = [u'GNB', u'RFC', u'DTC', u'LR', u'GBM', u'XGB', u'ADA']
clfList = [clf, tree, dTree,logReg, gboost, xgb, ada]

# Plots a ROC Curve of the list of models
plt.figure(figsize=(12,8))
for name,clf in izip(clfNames,clfList):

    clf.fit(X_train,y_train)
    y_proba = clf.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=name)

plt.xlabel(u'False positive rate')
plt.ylabel(u'True positive rate')
plt.title(u'ROC curve')
plt.legend(loc=u'best')
plt.savefig(u'ROC_Curve.png')
plt.show() 

# training models to see which has th highest AUROC score
print u"ROC score for training set: {}".format(trainModel(gnb, X_train, y_train))
print u"ROC score for training set: {}".format(trainModel(tree, X_train, y_train))
print u"ROC score for training set: {}".format(trainModel(dTree, X_train, y_train))
print u"ROC score for training set: {}".format(trainModel(logReg, X_train, y_train))
print u"ROC score for training set: {}".format(trainModel(gboost, X_train, y_train))
print u"ROC score for training set: {}".format(trainModel(xgb, X_train, y_train))
print u"ROC score for training set: {}".format(trainModel(ada, X_train, y_train))

print u"ROC score for test set: {}".format(trainModel(gnb, X_test, y_test))
print u"ROC score for test set: {}".format(trainModel(tree, X_test, y_test))
print u"ROC score for test set: {}".format(trainModel(dTree, X_test, y_test))
print u"ROC score for test set: {}".format(trainModel(logReg, X_test, y_test))
print u"ROC score for test set: {}".format(trainModel(gboost, X_test, y_test))
print u"ROC score for test set: {}".format(trainModel(xgb, X_test, y_test))
print u"ROC score for test set: {}".format(trainModel(ada, X_test, y_test))

# Fine Tuned XGB Model
parameters = {u'max_depth':[4],u'min_child_weight':[9], u'gamma':[0.5], u'subsample':[0.8],
              u'colsample_bytree':[0.6], u'reg_alpha':[1e-05], u'learning_rate':[0.1],
              u'nthread':[0], u'n_estimators':[88],
              u'seed':[93], u'objective':[u'binary:logistic']}

bestXGB = GridSearchCV(XGBClassifier(), parameters, scoring=u'roc_auc', cv=10 )

bestXGB.fit(X_train, y_train)

proba = bestXGB.predict_proba(test)

print proba

# Submission dataframe for Kaggle Contest
submission = pd.DataFrame({u"ID":test_id, u"TARGET": proba[:,1]})
submission.head()
submission.to_csv(u"submission.csv", index=False)
