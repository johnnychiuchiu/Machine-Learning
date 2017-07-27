##reference:
# https://www.kaggle.com/abriosi/predicting-red-hat-business-value/raddar-0-98-xgboost-sparse-matrix-python/output

##key feature:
# better do oneHotEncoding for categorical variables before training the data

import numpy as np
import pandas as pd
import xgboost_red_hat_business as xgb
from sklearn.preprocessing import OneHotEncoder

#######################################################################################
######################################################## Load Dataset #################
#######################################################################################
PATH = '/Users/johnnychiu/Desktop/MyFiles/learning/kaggle/5.Predicting-Red-Hat-Business-Value/_data/'


act_train_data = pd.read_csv(PATH+"act_train.csv",
                             dtype={'people_id': np.str, 'activity_id': np.str, 'outcome': np.int8},
                             parse_dates=['date'])
act_test_data = pd.read_csv(PATH+"act_test.csv", dtype={'people_id': np.str, 'activity_id': np.str},
                            parse_dates=['date'])
people_data = pd.read_csv(PATH+"people.csv",
                          dtype={'people_id': np.str, 'activity_id': np.str, 'char_38': np.int32}, parse_dates=['date'])



######################################################################################
################################################ Data Pre-processing #################
######################################################################################
######################################################################################
################################################ Data Spliting #######################
######################################################################################

#data pre-processing

def data_treatment(dsname):
    # turn object type column into int32
    # turn bool type column into int8
    # add year, month, day, isweekend columns using date column, and drop date column
    dataset = dsname

    for col in list(dataset.columns):
        if col not in ['people_id', 'activity_id', 'date', 'char_38', 'outcome']:
            if dataset[col].dtype == 'object':
                dataset[col].fillna('type 0', inplace=True)
                dataset[col] = dataset[col].apply(lambda x: x.split(' ')[1]).astype(np.int32)
            elif dataset[col].dtype == 'bool':
                dataset[col] = dataset[col].astype(np.int8)

    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    dataset['day'] = dataset['date'].dt.day
    dataset['isweekend'] = (dataset['date'].dt.weekday >= 5).astype(int)
    dataset = dataset.drop('date', axis=1)

    return dataset

# drop unneeded column
# act_train_data = act_train_data.drop('char_10', axis=1)
# act_test_data = act_test_data.drop('char_10', axis=1)

# transform act and people data
act_train_data = data_treatment(act_train_data)
act_test_data = data_treatment(act_test_data)
people_data = data_treatment(people_data)

# merge act data with people data
train = act_train_data.merge(people_data, on='people_id', how='left', left_index=True)
test = act_test_data.merge(people_data, on='people_id', how='left', left_index=True)

# drop analysis unrelated columns, such as people_id...etc
train=train.drop(['people_id', 'activity_id'], axis=1)
test=test.drop(['people_id', 'activity_id'], axis=1)

y = train.outcome
train = train.drop('outcome', axis=1)

# find categorical and non-categorical columns
categorical = ['group_1', 'activity_category', 'char_1_x', 'char_2_x', 'char_3_x', 'char_4_x', 'char_5_x', 'char_6_x',
               'char_7_x', 'char_8_x', 'char_9_x', 'char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y',
               'char_7_y', 'char_8_y', 'char_9_y']
not_categorical = []
for category in train.columns:
    if category not in categorical:
        not_categorical.append(category)

# turn categorical variable into separate columns using OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc = enc.fit(pd.concat([train[categorical], test[categorical]]))
train_cat_sparse = enc.transform(train[categorical])
test_cat_sparse = enc.transform(test[categorical])


# combine the encoded sparse matrix with non-categorical(boolean) using hstack
from scipy.sparse import hstack
train_sparse = hstack((train[not_categorical], train_cat_sparse))
test_sparse = hstack((test[not_categorical], test_cat_sparse))


# turn the combined sparse matrix into xgboost usable data
dtrain = xgb.DMatrix(train_sparse, label=y)
dtest = xgb.DMatrix(test_sparse)





######################################################################################
################################################ Data Training and tuning#############
######################################################################################

param = {'max_depth': 10, 'eta': 0.02, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['min_child_weight'] = 0
param['booster'] = "gblinear"

watchlist = [(dtrain, 'train')]
num_round = 300
early_stopping_rounds = 10
bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=early_stopping_rounds)


###############################################################################
################################################ Model Selection ##############
###############################################################################

################################################################################
################################################ Model Prediction ##############
################################################################################

ypred = bst.predict(dtest)
output = pd.DataFrame({'activity_id': test['activity_id'], 'outcome': ypred})
output.head()
