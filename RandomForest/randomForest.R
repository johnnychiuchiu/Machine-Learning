#reference:http://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/
library(randomForest)
library(caret)
library(mlbench)

#######################################################################################
######################################################## Load Dataset #################
#######################################################################################

data(Sonar)
dataset <- Sonar


######################################################################################
################################################ Data Spliting #######################
######################################################################################
#simple bootstraping 
inTrain <- createDataPartition(y=dataset$Class, p=0.7, list=FALSE)
training <- dataset[inTrain,]
validating <- dataset[-inTrain,]


#k-fold cross validation
control <- trainControl(method="repeatedcv", number=10, repeats=3)

seed <- 7
set.seed(seed)
######################################################################################
################################################ Data Training and tuning#############
######################################################################################

###########################################
################# Create model with default paramters
rf_default <- train(Class~ .,data=training,method="rf",prox=TRUE)
print(rf_default)
rf_default$finalModel #OOB estimate of  error rate: 18.49% (accuracy=100-18.49)
plot(rf_default)

###########################################
#################Custom setting: create model with custom paramters, such as mtry, metric
metric <- "Accuracy"
mtry <- sqrt(ncol(x))
tunegrid <- expand.grid(.mtry=mtry)

rf_custom <- train(Class~., data=training, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_custom)
rf_custom$finalModel #OOB estimate of  error rate: 16.44% (accuracy=100-16.44)
plot(rf_custom)

###########################################
#################Random Search (for the best mtry)
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(seed)
mtry <- sqrt(ncol(x))
rf_random <- train(Class~., data=training, method="rf", metric=metric, tuneLength=15, trControl=control)
#a data frame with possible tuning values. The columns are named the same as the tuning parameters. Use getModelInfo to get a list of tuning parameters for each model or see http://topepo.github.io/caret/modelList.html. (NOTE: If given, this argument must be named.)
print(rf_random)
rf_random$finalModel #OOB estimate of  error rate: 19.86%  (accuracy=100-19.86)
plot(rf_random)

###########################################
#################Grid Search (for the best mtry)
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:15))
rf_gridsearch <- train(Class~., data=training, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
rf_gridsearch$finalModel #OOB estimate of  error rate: 17.12%  (accuracy=100-17.12)
plot(rf_gridsearch)

###########################################
#################Algorithm Tune (tuneRF) (for the best mtry)

bestmtry <- tuneRF(training[,1:60], training[,61], stepFactor=1.5, improve=1e-5, ntree=500)
print(bestmtry)
#result: mtry=7, with the lowest OOBerror 0.1917
#OOB estimate of  error rate: 19.18%  (accuracy=100-19.18)


mtry <- 7
tunegrid <- expand.grid(.mtry=mtry)

rf_tune <- train(Class~., data=training, method="rf", tuneGrid=tunegrid, trControl=control)
print(rf_tune)
rf_tune$finalModel #OOB estimate of  error rate: 17.12% (accuracy=100-17.12)
plot(rf_tune)



###############################################################################
################################################ Model Selection ##############
###############################################################################
my_accuracy<-function(confusion_table){
  return ((confusion_table[1,1]+confusion_table[2,2])/(confusion_table[1,1]+confusion_table[2,2]+confusion_table[1,2]+confusion_table[2,1]))
}

###########################################
################# Create model with default paramters!!!!!winner 
# mtry=2
pred_default <- predict(rf_default, newdata = validating)
my_accuracy(table(observed = validating$Class, predicted = pred_default))



###########################################
#################Custom setting: create model with custom paramters, such as mtry, metric
# mtry=8
pred_custom <- predict(rf_custom, newdata = validating)
my_accuracy(table(observed = validating$Class, predicted = pred_custom))


###########################################
#################Random Search
# mtry=5
pred_random <- predict(rf_random, newdata = validating)
my_accuracy(table(observed = validating$Class, predicted = pred_random))


###########################################
#################Grid Search
# mtry=1
pred_gridsearch <- predict(rf_gridsearch, newdata = validating)
my_accuracy(table(observed = validating$Class, predicted = pred_gridsearch))


###########################################
#################Algorithm Tune (tuneRF)
# mtry=7
pred_gridsearch <- predict(rf_gridsearch, newdata = validating)
my_accuracy(table(observed = validating$Class, predicted = pred_gridsearch))



################################################################################
################################################ Model Prediction ##############
################################################################################

final_prediction <- predict(rf_default, newdata = testing_dataset_which_I_do_not_have)


