##reference:
#- https://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/
#- http://xgboost.readthedocs.io/en/latest/model.html
#- http://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html
#- https://www.r-bloggers.com/an-introduction-to-xgboost-r-package/

library(xgboost)

##key feature:
#- Extreme Gradient Boosting (xgboost) is similar to gradient boosting framework but more efficient. 
#  It is an extension of the classic gbm algorithm.
#- It supports various objective functions, including regression, classification and ranking.
#- XGBoost is able to utilize more computational power and get more accurate prediction.

#######################################################################################
################################################ Load Dataset  ########################
################################################ Data Spliting ########################
#######################################################################################

data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test

dim(train$data)
dim(test$data)


######################################################################################
################################################ Data Training and tuning#############
######################################################################################
#objective = "binary:logistic": we will train a binary classification model ; 
#max.deph = 2: the trees won’t be deep, because our case is very simple ; 
#nthread = 2: the number of cpu threads we are going to use; 
#nround = 2: there will be two passes on the data, the second one will enhance the model by further reducing the difference between ground truth and prediction.

###########################################
################# create model using sparse matrix
bstSparse <- xgboost(data = train$data, 
                     label = train$label, 
                     max.depth = 2, 
                     eta = 1, 
                     nthread = 2, 
                     nround = 2, 
                     objective = "binary:logistic")

######## setting version options
# verbose = 0, no message
bst <- xgboost(data = dtrain, 
               max.depth = 2, 
               eta = 1, 
               nthread = 2, 
               nround = 2, 
               objective = "binary:logistic", 
               verbose = 0)

# verbose = 1, print evaluation metric
bst <- xgboost(data = dtrain, 
               max.depth = 2, 
               eta = 1, 
               nthread = 2, 
               nround = 2, 
               objective = "binary:logistic", 
               verbose = 1)

# verbose = 2, also print information about tree
bst <- xgboost(data = dtrain, 
               max.depth = 2, 
               eta = 1, 
               nthread = 2, 
               nround = 2, 
               objective = "binary:logistic", 
               verbose = 2)


###########################################
################# create model using dense matrix
bstDense <- xgboost(data = as.matrix(train$data), 
                    label = train$label, 
                    max.depth = 2, 
                    eta = 1, 
                    nthread = 2, 
                    nround = 2, 
                    objective = "binary:logistic")

###########################################
################# create model using its own xgb.DMatrix
#XGBoost offers a way to group them in a xgb.DMatrix. You can even add other meta data in it. It will be useful for the most advanced features we will discover later.
dtrain <- xgb.DMatrix(data = train$data, label = train$label)
bstDMatrix <- xgboost(data = dtrain, 
                      max.depth = 2, 
                      eta = 1, 
                      nthread = 2, 
                      nround = 2, 
                      objective = "binary:logistic")

###########################################
################# create model and measure learning progress with xgb.train
#One of the special feature of xgb.train is the capacity to follow the progress of the learning after each round. Because of the way boosting works, there is a time when having too many rounds lead to an overfitting. You can see this feature as a cousin of cross-validation method. The following techniques will help you to avoid overfitting or optimizing the learning time in stopping it as soon as possible.
#One way to measure progress in learning of a model is to provide to XGBoost a second dataset already classified. Therefore it can learn on the first dataset and test its model on the second one. Some metrics are measured after each round during the learning.
#For the purpose of this example, we use watchlist parameter. It is a list of xgb.DMatrix, each of them tagged with a name.
dtrain <- xgb.DMatrix(data = train$data, label=train$label)
dtest <- xgb.DMatrix(data = test$data, label=test$label)

watchlist <- list(train=dtrain, test=dtest)
bstCustomTrain <- xgb.train(data=dtrain, 
                 max.depth=2, 
                 eta=1, 
                 nthread = 2, 
                 nround=2, 
                 watchlist=watchlist, 
                 #we can add more evaluation metrics using eval.metric parameter
                 #eval.metric = "error", eval.metric = "logloss", 
                 objective = "binary:logistic")

###########################################
################# create model and measure learning progress using linear boosting
bstCustomLinear <- xgb.train(data=dtrain, 
                 booster = "gblinear", 
                 max.depth=2, 
                 nthread = 2, 
                 nround=2, 
                 watchlist=watchlist, 
                 eval.metric = "error", eval.metric = "logloss", 
                 objective = "binary:logistic")





###############################################################################
################################################ Model Selection ##############
###############################################################################


###########################################
################# create model using sparse matrix
#bstSparse
pred_Sparse <- predict(bstSparse, test$data)
err <- mean(as.numeric(pred_Sparse > 0.5) != test$label) #count True / the number of test dataset 
print(paste("test-error=", err)) #test-error= 0.0217256362507759

###########################################
################# create model using dense matrix
#bstDense
pred_Dense <- predict(bstDense, test$data)
err <- mean(as.numeric(pred_Dense > 0.5) != test$label) #count True / the number of test dataset 
print(paste("test-error=", err)) #test-error= 0.0217256362507759

###########################################
################# create model using its own xgb.DMatrix
#bstDMatrix
pred_DMatrix <- predict(bstDMatrix, test$data)
err <- mean(as.numeric(pred_DMatrix > 0.5) != test$label) #count True / the number of test dataset 
print(paste("test-error=", err)) #test-error= 0.0217256362507759

###########################################
################# create model and measure learning progress with xgb.train
#bstCustomTrain
pred_CustomTrain <- predict(bstCustomTrain, test$data)
err <- mean(as.numeric(pred_CustomTrain > 0.5) != test$label) #count True / the number of test dataset 
print(paste("test-error=", err)) #test-error= 0.0217256362507759

###########################################
################# create model and measure learning progress using linear boosting
#bstCustomLinear
pred_CustomLinear <- predict(bstCustomLinear, test$data)
err <- mean(as.numeric(pred_CustomLinear > 0.5) != test$label) #count True / the number of test dataset 
print(paste("test-error=", err)) #test-error= 0  (0!?!? winner!)


################################################################################
################################################ Model Prediction ##############
################################################################################
pred <- predict(pred_CustomLinear, testing_dataset_which_I_do_not_have)
prediction <- as.numeric(pred > 0.5) #convert the prediction into binary
print(head(prediction))



################################################################################################################################################################
###### Others


################################################################################
################################################ feature importance ############
################################################################################
importance_matrix <- xgb.importance(model = bst)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)



