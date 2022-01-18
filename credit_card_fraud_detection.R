## loading the data
data = read.csv("creditcard.csv")

##load the required libraries
library(Hmisc)
library(tidyr)
library(dplyr)
library(caTools)   ## for dividing data into test and train
library(rpart)     ## decision tree algorithm
library(rpart.plot) 
library(caret)     ## for making confusion matrix
library(randomForest)  ## Random Forest algorithm
library(e1071)         ## Support Vector Machine Algorithm
library(ROSE)      ## library required to do oversampling and under sampling
library(xgboost)     ## xgboost algorithm 
 

describe(data)
## There is no null value in any of the explanatory variable in the sample data

## there are 0.17% of fraudulent transaction which
perc_fraud = round(100*sum(data$Class)/nrow(data),2)

sprintf("total number of transaction %d",nrow(data))
sprintf("Number of Genuine Transaction %d",nrow(data[data$Class==0,]))
sprintf("Number of Fradulent Transaction %d",nrow(data[data$Class==1,]))
sprintf("percentage of fraud transaction in the data %g",perc_fraud)

## removing the duplicate data in my data set
data = distinct(data)
data = subset(data,select = -c(Time))
data$Class = as.factor(data$Class)

##shuffling the data before dividing it into train and test
data= data[sample(1:nrow(data)), ]

## dividing the data set in test and train data
split = sample.split(data$Class,SplitRatio = 0.8)
train = subset(data, split == TRUE)
test = subset(data, split == FALSE)

## comparing the division of fraudulent transactions in training and testing data
nrow(test[test$Class==1,])/nrow(test)       ## 0.167% fraudulent transactions in test data
nrow(train[train$Class==1,])/nrow(train)    ## 0.166% fraudulent transactions in train data


## using Decision Tree machine learning algorithm to do classification
dt = rpart(Class~., train, method = "class")
rpart.plot(dt,extra = 100)
prediction = predict(dt,test,type = "class")
confusionMatrix(prediction,test$Class)

precision = posPredValue(prediction, test$Class, positive="1")
recall = sensitivity(prediction, test$Class, positive="1")
F1= (2*precision*recall)/(precision+recall)
sprintf("F1 score for Decision Tree Algorithm is %f",F1)     ## F1 -score of 0.80


## using Logistic Regression Learning algorithm to do classification
lr = glm(Class~.,train,family = "binomial")
summary(lr)

prediction = predict(lr,test,type = "response")
prediction = ifelse(prediction>=0.5,1,0)


confusion = table(prediction = prediction,Actual = test$Class)
confusion
prediction = as.factor(prediction)
precision = posPredValue(prediction, test$Class, positive="1")
recall = sensitivity(prediction, test$Class, positive="1")
F1= (2*precision*recall)/(precision+recall)
sprintf("F1 score for Logistic Regression Algorithm is %f",F1)   ## F1-score of 0.76

## first we are required to do under sampling of data
under = ovun.sample(Class~., train, method = "under", N = 45*length(which(train$Class==1)))
under = under$data

## using Support Vector Machine Algorithm to do Classification
svm = svm(Class~., under, type = "C-classification", kernel = "linear")
print(svm)
prediction = predict(svm, test)
confusion = table(prediction = prediction,Actual = test$Class)
confusion

precision = posPredValue(prediction, test$Class, positive="1")
recall = sensitivity(prediction, test$Class, positive="1")
F1= (2*precision*recall)/(precision+recall)
sprintf("F1 score for Support Vector Machine Algorithm is %f",F1)    ## F1 score from SVM is converging to 78.5%  upon increasing the size of "under" training data


## Using Random Forest machine learning algorithm to do classification
rf = randomForest(Class~.,under)
print(rf)

prediction = predict(rf,test)
confusion = table(prediction = prediction,actual = test$Class)
confusion

precision = posPredValue(prediction,test$Class,positive = "1")
recall = sensitivity(prediction,test$Class,positive = "1")
F1 = (2*precision*recall)/(precision+recall)
sprintf("F1 score for Random Forest Machine Algorithm is %f",F1)   ## F1 score from ranom Forest is converging to 78.3% upon increasing the size of "under" training data


## using xgboost algorithm to do classification
under = ovun.sample(Class~.,train , method = "under", N = 60*length(which(train$Class==1)) )
under = under$data
under1 = subset(under, select = -c(Class))
under$Class = as.factor(under$Class)


grid_tune = expand.grid(
  nrounds = 1500,
  gamma = 0,
  max_depth =6,
  eta = 0.3,
  subsample =1,
  min_child_weight =1,
  colsample_bytree = 1
)


xgb = train(x = as.matrix(under1),
                  y = under$Class,
                  tuneGrid = grid_tune,
                  method= "xgbTree"
            )

prediction = predict(xgb, newtest)

table(prediction = prediction,actual = test$Class)

precision = posPredValue(prediction,test$Class,positive = "1")
recall = sensitivity(prediction,test$Class,positive = "1")
F1 = (2*precision*recall)/(precision+recall)
sprintf("F1 score for Xgboost Machine Algorithm is %f",F1)   ## F1-score of 83.7 for Xgboost

