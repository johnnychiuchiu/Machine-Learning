library(caret)

#######################################################################################
######################################################## Load Dataset #################
#######################################################################################

dataset<-data.frame(mtcars)
dataset

######################################################################################
################################################ Data Spliting #######################
######################################################################################
inTrain <- createDataPartition(y=dataset$mpg, p=0.7, list=FALSE)
training <- dataset[inTrain,]
validating <- dataset[-inTrain,]

######################################################################################
################################################ Data Training and tuning#############
######################################################################################

###########################################
################# In the first model, I use all the variables to predict mpg. 
#The result Adjusted R- squared is 0.8017, but we can see easily that the P value of many of our coefficients are large. It suggest us it's not a good model to use.
model_1<-lm( mpg~. , data=training)
summary(model_1) 

###########################################
################# In the second model, I choose 5 of the coefficients that have lower P value.
#Among the variable that we have, "disp", "hp", "wt", "qsec", and "am" have lower P value. In the summary report, we see that the Adjusted R squared have rise to 0.8116.
model_2<-lm( mpg~disp+hp+wt+qsec+am , data=training)
summary(model_2) 

###########################################
################# In our third model, I use the "step" function from the base {stats} package. 
#This step function help us Choose a model by AIC in a stepwise algorithm. AIC represents Akaike information criterion. The lower AIC, the better model selection suggested.
#We can see below that this step function in the end choose "wt", "qsec", and "am" as our final coefficients. In the summary report, we see that the Adjusted R squared equals 0.8409.
model_3 <- step(model_1)
summary(model_3) 


###############################################################################
################################################ Model Selection ##############
###############################################################################

###########################################
################# Method 1: adjusted R-square

summary(model_1) #0.8017 
summary(model_2) #0.8116 
summary(model_3) #0.8409 winner

###########################################
################# Method 2: Pearson Correlation using validation data. The higher, the better.
#model_1
mpgPred <- predict(model_1, validating)
actuals_preds <- data.frame(cbind(actuals=validating$mpg, predicteds=mpgPred))  # make actuals_predicteds dataframe.
correlation_accuracy <- cor(actuals_preds)  # 77.7

#model_2
mpgPred <- predict(model_2, validating)
actuals_preds <- data.frame(cbind(actuals=validating$mpg, predicteds=mpgPred))  # make actuals_predicteds dataframe.
correlation_accuracy <- cor(actuals_preds)  # 93.4 winner!

#model_3
mpgPred <- predict(model_3, validating)
actuals_preds <- data.frame(cbind(actuals=validating$mpg, predicteds=mpgPred))  # make actuals_predicteds dataframe.
correlation_accuracy <- cor(actuals_preds)  # 76.2

###########################################
################# Method 2: k-fold cross validation 
# Here, I use 3 fold cross-validation
library(DAAG)
cv.lm(data=dataset, model_1, m=3) #overall ms 34.9
cv.lm(data=dataset, model_2, m=3) #overall ms 7.76 winner!
cv.lm(data=dataset, model_3, m=3) #overall ms 8.63

######
# model_2 is the winner.
######

################################################################################
################################################ Model Prediction ##############
################################################################################

final_prediction <- predict(model_2, testing_dataset_which_I_do_not_have) 


#some reference: 
#http://r-statistics.co/Linear-Regression.html

