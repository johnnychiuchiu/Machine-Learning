##reference:
# https://www.analyticsvidhya.com/blog/2015/09/complete-guide-boosting-methods/
# https://www.quora.com/What-is-the-difference-between-gradient-descent-and-gradient-boosting
# https://en.wikipedia.org/wiki/Gradient_boosting
  #note: Gradient boosting is typically used with decision trees (especially CART trees) of a fixed size as base learners.
# http://amunategui.github.io/binary-outcome-modeling/
# https://www.r-bloggers.com/predicting-titanic-deaths-on-kaggle-ii-gbm/
# https://www.quora.com/What-is-the-difference-between-gradient-descent-and-gradient-boosting
  #In another word: gradient descent update the parameters of a function step by step to reach a local minimal of loss function; 
  #gradient boosting adds new function to existing one in each step to reach a local minimal of loss function. 
  #In the end, the result of gradient descent is still the same function as at the beginning, just with a better parameters. 
  #But gradient boosting will end with a totally different functions (additions of multiple functions). 
  #e.g. Adaboost is a special case of gradient boosting.


##key feature:
#- Gradient boosting is a machine learning technique for regression and classification problems, 
#  which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.
#- There are diffierent kinds of machine learning techniques, such as bagging, boosting, stacking. 
#  It's one kind of boosting method.

library(caret)
#######################################################################################
######################################################## Load Dataset #################
#######################################################################################
titanicDF <- read.csv('http://math.ucdenver.edu/RTutorial/titanic.txt',sep='\t')

#We need to clean up a few things as is customary with any data science project. The Name variable is mostly unique so we’re going to extract the title and throw the rest away.
titanicDF$Title <- ifelse(grepl('Mr ',titanicDF$Name),'Mr',ifelse(grepl('Mrs ',titanicDF$Name),'Mrs',ifelse(grepl('Miss',titanicDF$Name),'Miss','Nothing'))) 

#The Age variable has missing data (i.e. NA’s) so we’re going to impute it with the mean value of all the available ages. 
titanicDF$Age[is.na(titanicDF$Age)] <- median(titanicDF$Age, na.rm=T)

titanicDF <- titanicDF[c('PClass', 'Age',    'Sex',   'Title', 'Survived')]
print(str(titanicDF))

#Our data is starting to look good but we have to fix the factor variables as most models only accept numeric data. 
#Again, gbm can deal with factor variables as it will dummify them internally, but glmnet won’t. 
#In a nutshell, dummifying factors breaks all the unique values into separate columns. 
#This is a caret function:
titanicDF$Title <- as.factor(titanicDF$Title)
titanicDummy <- dummyVars("~.",data=titanicDF, fullRank=F)
titanicDF <- as.data.frame(predict(titanicDummy,titanicDF))


#I like generalizing my variables so that I can easily recycle the code for subsequent needs:
outcomeName <- 'Survived'
predictorsNames <- names(titanicDF)[names(titanicDF) != outcomeName]

#It is important to know what type of modeling a particular model supports. 
#This can be done using the caret function getModelInfo:
getModelInfo()$gbm$type

# This tells us that gbm supports both regression and classification. As this is a binary classification, 
# we need to force gbm into using the classification mode. 
# We do this by changing the outcome variable to a factor 
# (we use a copy of the outcome as we’ll need the original one for our next model):
titanicDF$Survived <- ifelse(titanicDF$Survived==1,'yes','nope')
titanicDF$Survived <- as.factor(titanicDF$Survived)



######################################################################################
################################################ Data Spliting #######################
######################################################################################
set.seed(1234)
inTrain <- createDataPartition(y=titanicDF[,outcomeName], p = .75, list = FALSE, times = 1)
training <- titanicDF[inTrain,]
validating <- titanicDF[-inTrain,]

######################################################################################
################################################ Data Training and tuning#############
######################################################################################
objControl <- trainControl(method='cv', number=3, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)

objModel <- train(training[,predictorsNames], training[,outcomeName], 
                  method='gbm', 
                  trControl=objControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"))

summary(objModel)
print(objModel)


###############################################################################
################################################ Model Selection ##############
###############################################################################
# my_accuracy<-function(confusion_table){
#   return ((confusion_table[1,1]+confusion_table[2,2])/(confusion_table[1,1]+confusion_table[2,2]+confusion_table[1,2]+confusion_table[2,1]))
# }


###########################################
################# 
#There are two types of evaluation we can do here, raw or prob. 
#Raw gives you a class prediction, in our case yes and nope, while prob gives you the probability on how sure the model is about it’s choice. 
#I always use prob, as I like to be in control of the threshold and also like to use AUC score which requires probabilities, not classes. 
#There are situations where having class values can come in handy, such as with multinomial models where you’re predicting more than two values.
#We now call the predict function and pass it our trained model and our testing data. 

# get accuracy score
predictions <- predict(object=objModel, validating[,predictorsNames], type='raw') # default type is 'row'
head(predictions)
print(postResample(pred=predictions, obs=as.factor(validating[,outcomeName])))

# I don't need my_accuracy function anymore. postResample function from caret can replace it. 
# pred_default <- predict(objModel, newdata = validating[,predictorsNames])
# head(pred_default)
# my_accuracy(table(observed = validating$Survived, predicted = pred_default))


# probabilites 
library(pROC)
predictions <- predict(object=objModel, validating[,predictorsNames], type='prob')
head(predictions)

#To get the AUC score, you need to pass the yes column to the roc function 
#(each row adds up to 1 but we’re interested in the yes, the survivors):
auc <- roc(ifelse(validating[,outcomeName]=="yes",1,0), predictions[[2]])
print(auc$auc)
  


################################################################################
################################################ Model Prediction ##############
################################################################################
pred <- predict(object=objModel, testing_dataset_which_I_do_not_have)
print(head(prediction))

