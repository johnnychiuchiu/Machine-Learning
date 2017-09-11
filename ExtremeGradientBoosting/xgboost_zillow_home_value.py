# reference: https://www.kaggle.com/aharless/xgb-w-o-outliers-lgb-with-outliers-combined/code
# in this version, I am going to optimize the parameters of xgboost, referencing:
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import gc
import random

##### READ IN RAW DATA
PATH = '/Users/johnnychiu/Desktop/MyFiles/learning/kaggle/9.Zillow-Home-Value-Prediction'

properties = pd.read_csv(PATH + '/input/properties_2016.csv')
train = pd.read_csv(PATH + '/input/train_2016_v2.csv')

##### PROCESS DATA FOR XGBOOST

print("\nProcessing data for XGBoost ...")
for c in properties.columns:
    properties[c] = properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
x_test = properties.drop(['parcelid'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop out ouliers & downsize the train data by filter extreme cases
train_df = train_df[train_df.logerror > -0.4]
train_df = train_df[train_df.logerror < 0.418]

# # drop out ouliers & downsize the train data by random sampling
# random.seed(33)
# sel_parcelid = random.sample(train_df.parcelid.unique(), 200)
# train_df = train_df[train_df['parcelid'].isin(sel_parcelid)].reset_index(drop=True)


x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

##### XGBOOST MODEL TUNING
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics  # Additional scklearn functions   #cross_validation
# from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import GridSearchCV

##### Step 1: Fix learning rate and number of estimators for tuning tree-based parameters
### description:
# Choose a relatively high learning rate. Generally a learning rate of 0.1 works but somewhere between 0.05 to 0.3
# should work for different problems. Determine the optimum number of trees for this learning rate.
# XGBoost has a very useful function called as "cv" which performs cross-validation at each boosting iteration and thus
# returns the optimum number of trees required.
### note:
# to determine the optimum number of trees for this learning rate, the parameter for the "optimum number of trees" is
# called n_estimators in XGBClassifier; num_boost_round in xgboost

def modelfit(alg, x_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=10):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='mae', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(x_train, y_train, eval_metric='mae')

    # Predict training set:
    # dtrain_predictions = alg.predict(x_train)
    dtrain_predprob = alg.predict_proba(x_train)[:, 1]

    # Print model report:
    print "\nModel Report"
    # print "Accuracy : %.4g" % metrics.accuracy_score(y_train, dtrain_predictions)
    # print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
    # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    print "MAE (Train): %f" % metrics.mean_squared_error(y_train, dtrain_predprob)

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    print feat_imp
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')


# predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier(
    learning_rate=0.2,
    n_estimators=100,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:linear',
    n_jobs=4,
    scale_pos_weight=1,
    random_state=27)

# modelfit(xgb1, x_train, y_train)


##### Step 2: Tune max_depth and min_child_weight
### description
# We tune these first as they will have the highest impact on model outcome. To start with, let's set wider ranges
# and then we will perform another iteration for smaller ranges.
### note
# GridSearchCV documentation -> http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# scoring parameters -> http://scikit-learn.org/stable/modules/model_evaluation.html

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.2, n_estimators=26, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', n_jobs=4, scale_pos_weight=1, random_state=27),
 param_grid = param_test1, scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch1.fit(x_train, y_train)
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_

# Lets go one step deeper and look for optimum values. We'll search for values 1 above and below the optimum values
# because we took an interval of two.
param_test2 = {
 'max_depth':[3,4,5,6],
 'min_child_weight':[1,2,3]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.2, n_estimators=26, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', n_jobs=4, scale_pos_weight=1, random_state=27),
 param_grid = param_test2, scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch2.fit(x_train, y_train)
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_ # {'max_depth': 3, 'min_child_weight': 2}, -0.0062414673157036304


param_test2b = {
 'max_depth':[1,2,3]
}
gsearch2b = GridSearchCV(estimator = XGBClassifier(learning_rate =0.2, n_estimators=26, max_depth=3,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', n_jobs=4, scale_pos_weight=1, random_state=27),
 param_grid = param_test2b, scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch2b.fit(x_train, y_train)
gsearch2b.cv_results_, gsearch2b.best_params_, gsearch2b.best_score_ # {'max_depth': 2}, -0.0062414673157036304


##### Step 3: Tune gamma
### description
# Now lets tune gamma value using the parameters already tuned above. Gamma can take various values but I'll check
# for 5 values here. You can go into more precise values as.
### note
# gamma [default=0] A node is split only when the resulting split gives a positive reduction in the loss function.
# Gamma specifies the minimum loss reduction required to make a split. Higher gamma makes the algorithm conservative.
# The values can vary depending on the loss function and should be tuned.

param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.2, n_estimators=26, max_depth=3,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', n_jobs=4, scale_pos_weight=1, random_state=27),
 param_grid = param_test3, scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch3.fit(x_train, y_train)
gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_ # {'gamma': 0.1}, -0.0062309452332556248






##### Step 4: Tune subsample and colsample_bytree
### description
# The next step would be try different subsample and colsample_bytree values. Lets do this in 2 stages as well and
# take values 0.6,0.7,0.8,0.9 for both to start with.
### note

param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.2, n_estimators=26, max_depth=3,
 min_child_weight=2, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', n_jobs=4, scale_pos_weight=1, random_state=27),
 param_grid = param_test4, scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch4.fit(x_train, y_train)
gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_ # {'subsample': 0.8, 'colsample_bytree': 0.6}

# Now we should try values in 0.05 interval around the optimum value we just got.
param_test5 = {
    'colsample_bytree':[i/10.0 for i in range(3,7)]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.2, n_estimators=26, max_depth=3,
 min_child_weight=2, gamma=0.1, subsample=0.8, colsample_bytree=0.6,
 objective= 'reg:linear', n_jobs=4, scale_pos_weight=1, random_state=27),
 param_grid = param_test5, scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch5.fit(x_train, y_train)
gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_ #

##### Step 5: Tuning Regularization Parameters
### description
# Next step is to apply regularization to reduce overfitting. Though many people don't use this parameters much as
# gamma provides a substantial way of controlling complexity. But we should always try it. I'll tune 'reg_alpha'
# value here and leave it up to you to try different values of 'reg_lambda'.
# Tune regularization parameters (lambda, alpha) for xgboost which can help reduce model complexity and enhance performance.
### note

param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.2, n_estimators=26, max_depth=3,
 min_child_weight=2, gamma=0.1, subsample=0.8, colsample_bytree=0.6,
 objective= 'reg:linear', n_jobs=4, scale_pos_weight=1, random_state=27),
 param_grid = param_test6, scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch6.fit(x_train, y_train)
gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_ # {'reg_alpha': 0.01}, -0.0059092676267027853


param_test7 = {
 'reg_alpha':[0, 0.005, 0.01, 0.03, 0.05]
}
gsearch7 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.2, n_estimators=26, max_depth=3,
 min_child_weight=2, gamma=0.1, subsample=0.8, colsample_bytree=0.6,
 objective= 'reg:linear', n_jobs=4, scale_pos_weight=1, random_state=27, reg_alpha=0.01),
 param_grid = param_test7, scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch7.fit(x_train, y_train)
gsearch7.cv_results_, gsearch7.best_params_, gsearch7.best_score_ # {'reg_alpha': 0.01}, -0.0059092676267027853



##### Step 6: Reducing Learning Rate
### description
# Lastly, we should lower the learning rate and add more trees. Lets use the cv function of XGBoost to do the job again.
# Lower the learning rate and decide the optimal parameters .
### note

# xgb2 = XGBClassifier(
#     learning_rate=0.05,
#     n_estimators=105,
#     max_depth=3,
#     min_child_weight=2,
#     gamma=0.1,
#     subsample=0.8,
#     colsample_bytree=0.6,
#     objective='reg:linear',
#     n_jobs=4,
#     scale_pos_weight=1,
#     random_state=27,
#     reg_alpha=0.01)
#
# modelfit(xgb2, x_train, y_train)
# learning_rate = 0.1,  n_estimators=66 (max 100), 0.005946
# learning_rate = 0.05,  n_estimators=105 (max 200), 0.005946
# learning_rate = 0.03,  n_estimators=188 (max 300), 0.005947



##### RUN XGBOOST

print("\nSetting up data for XGBoost ...")
# xgboost params

new_xgb_params = {
    'eta': 0.05,
    'num_boost_round': 105,
    'max_depth': 3,
    'min_child_weight': 2,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'objective': 'reg:linear',
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27,
    'alpha': 0.01,
    'eval_metric': 'mae',
    'silent': 0
}
# xgb_params = {
#     'eta': 0.037,
#     'max_depth': 5,
#     'subsample': 0.80,
#     'objective': 'reg:linear',
#     'eval_metric': 'mae',
#     'lambda': 0.8,
#     'alpha': 0.4,
#     'base_score': y_mean,
#     'silent': 0
# }

# Enough with the ridiculously overfit parameters.
# I'm going back to my version 20 instead of copying Jayaraman.
# I want a num_boost_rounds that's chosen by my CV,
# not one that's chosen by overfitting the public leaderboard.
# (There may be underlying differences between the train and test data
#  that will affect some parameters, but they shouldn't affect that.)

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

# cross-validation
# print( "Running XGBoost CV ..." )
# cv_result = xgb.cv(xgb_params,
#                   dtrain,
#                   nfold=5,
#                   num_boost_round=350,
#                   early_stopping_rounds=50,
#                   verbose_eval=10,
#                   show_stdv=False
#                  )
# num_boost_rounds = len(cv_result)

# num_boost_rounds = 150
# num_boost_rounds = 242
# print("\nXGBoost tuned with CV in:")
# print("   https://www.kaggle.com/aharless/xgboost-without-outliers-tweak ")
# print("num_boost_rounds=" + str(num_boost_rounds))

# train model
print("\nTraining XGBoost ...")
model = xgb.train(dict(new_xgb_params), dtrain, num_boost_round=105)

print("\nPredicting with XGBoost ...")
xgb_pred = model.predict(dtest)

print("\nXGBoost predictions:")
print(pd.DataFrame(xgb_pred).head())

##### WRITE THE RESULTS

print("\nPreparing results for write ...")
y_pred = []

for i, predict in enumerate(xgb_pred):
    y_pred.append(str(round(predict, 4)))
y_pred = np.array(y_pred)

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
                       '201610': y_pred, '201611': y_pred, '201612': y_pred,
                       '201710': y_pred, '201711': y_pred, '201712': y_pred})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
from datetime import datetime

print("\nWriting results to disk ...")
output.to_csv(
    PATH + '/20170726_XGB-LGB-combined/_submission/sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')),
    index=False)

print("\nFinished ...")
