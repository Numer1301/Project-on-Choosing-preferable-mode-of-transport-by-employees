setwd("/Users/numerp/Documents/PGP-BABI/Module 6   Machine Learning/Project 5")
getwd()
library(readr)
library(dplyr)
library(psych)
library(car)
library(carData)
library(ggplot2)
library(mice)
library(lattice)
library(nFactors)
library(scatterplot3d)
library(data.table)
library(tidyverse)
library(broom)
library(GGally)
transport=read_csv("Cars_edited.csv",col_names = TRUE)
names(transport)
names(transport)[5]="WorkExp"
names(transport)
transport
str(transport)
summary(transport)
dim(transport)
any(is.na(transport))
transport=na.omit(transport)
dim(transport)
summary(transport)
transport$Gender=as.factor(transport$Gender)
transport$Engineer=as.factor(transport$Engineer)
transport$MBA=as.factor(transport$MBA)
transport$license=as.factor(transport$license)
transport$Transport=as.factor(transport$Transport)
str(transport)
summary(transport)
transport$Transport=as.character(transport$Transport)
transport$Transport[transport$Transport %in% "2Wheeler"]="0"
transport$Transport[transport$Transport %in% "Car"]="1"
transport$Transport[transport$Transport %in% "Public Transport"]="0"
str(transport$Transport)
transport$Gender=as.character(transport$Gender)
transport$Gender[transport$Gender %in% "Female"]="0"
transport$Gender[transport$Gender %in% "Male"]="1"
str(transport$Gender)
transport$Transport=as.factor(transport$Transport)
transport$Gender=as.factor(transport$Gender)
str(transport)
summary(transport)
summary(transport$Transport)
summary(transport$Gender)
head(transport,3)
tail(transport,4)
ct.data=subset(transport,select = c(Gender,Engineer,MBA,license))
num.data=subset(transport,select = -c(Gender,Engineer,MBA,license,Transport))
names(ct.data)
names(num.data)
by(transport,INDICES = transport$Transport,FUN = summary)
ggpairs(transport[,c("Age","WorkExp","Salary","Distance")],
        ggplot2::aes(colour=as.factor(transport$Transport)))
outliers=boxplot(num.data,plot = FALSE)$out
outliers
plot(outliers)
correlation=cor(num.data)
correlation
corrplot::corrplot(correlation,method = "circle",type = "upper")
par(mfrow=c(2,2))
hist(transport$Age,main = "Age of the employees",col = "grey")
hist(transport$WorkExp,main = "Work Experience",col = "red")
hist(transport$Salary,main = "Salary of the employees",col = "yellow")
hist(transport$Distance,main = "Distance Travelling",col = "green")
par(mfrow=c(2,2))
boxplot(transport$Age,horizontal = T,main = "Age of the employees",col = "grey")
boxplot(transport$WorkExp,horizontal = T,main = "Work Experience",col = "red")
boxplot(transport$Salary,horizontal = T,main = "Salary of the employees",col = "yellow")
boxplot(transport$Distance,horizontal = T,main = "Distance Travelling",col = "green")
par(mfrow=c(2,2))
for(i in names(ct.data)){
  print(i)
  print(round(prop.table(table(transport$Transport,ct.data[[i]])),digits = 3)*100)
  barplot(table(transport$Transport,ct.data[[i]]),
          col = c("yellow","blue"),
          main = names(ct.data[i]))
}
attach(transport)
log.reg=glm(Transport~Age,data = transport,family = binomial(link = logit))
summary(log.reg)
log.reg=glm(Transport~Gender,data = transport,family = binomial(link = logit))
summary(log.reg)
log.reg=glm(Transport~Engineer,data = transport,family = binomial(link = logit))
summary(log.reg)
log.reg=glm(Transport~MBA,data = transport,family = binomial(link = logit))
summary(log.reg)
log.reg=glm(Transport~WorkExp,data = transport,family = binomial(link = logit))
summary(log.reg)
log.reg=glm(Transport~Salary,data = transport,family = binomial(link = logit))
summary(log.reg)
log.reg=glm(Transport~Distance,data = transport,family = binomial(link = logit))
summary(log.reg)
log.reg=glm(Transport~license,data = transport,family = binomial(link = logit))
summary(log.reg)
model=glm(Transport~.,data = transport,family = binomial(link = "logit"))
model
summary(model)
car::vif(model)
par(mfrow=c(1,4))
plot(model)
#splitting data
library(caTools)
set.seed(50)
splits=sample.split(transport$Transport,SplitRatio = 0.80)
train=subset(transport,splits==TRUE)
test=subset(transport,splits==FALSE)
prop.table(table(transport$Transport))
prop.table(table(test$Transport))
prop.table(table(train$Transport))           
trainmodel=glm(Transport~.,data = train,family = binomial(link = logit))
summary(trainmodel)
car::vif(trainmodel)
library(lmtest)
lrtest(trainmodel)
library(pscl)
pR2(trainmodel)["McFadden"]
logLik(trainmodel)
trainmodel1=glm(Transport~1,data = train,family = binomial(link = logit))
1-(logLik(trainmodel)/logLik(trainmodel1))
logLik(trainmodel1)
testpredict=predict(trainmodel,newdata = test,type = "response")
testpredict
table(test$Transport,testpredict>0.5)
#**************************************************************************************************
#logistics regression
library(caret)
vif(glm(Transport~Age+WorkExp+Salary+Distance,data = train,family = binomial(link = logit)))
summary(glm(Transport~Age+WorkExp+Salary+Distance,data = train,family = binomial(link = logit)))
logistics.car=glm(Transport~Age+WorkExp+Salary+Distance,data = train,family = binomial(link = logit))
logistics.car
exp(coef(logistics.car))
exp(coef(logistics.car))/(1+exp(coef(logistics.car)))
nrow(train[train$Transport==1,])/nrow(train)
lrtest(logistics.car)
pR2(logistics.car)["McFadden"]
logLik(logistics.car)
logistics.pred=predict(logistics.car,data=train,type = "response")
logistics.pred
pred.num=ifelse(logistics.pred>0.5,1,0)
pred.num
pred=factor(pred.num,levels = c(0,1),labels = c(0,1))
pred
pred.actual=train$Transport
pred.actual
cm_log_reg=confusionMatrix(pred,pred.actual,positive="1")
cm_log_reg
#***************************************************************************************************
#knn
library(class)
scale = preProcess(train, method = "range")
train.norm = predict(scale, train)
test.norm = predict(scale, test)
knn = train(Transport ~., data = train.norm, method = "knn",
                trControl = trainControl(method = "cv", number = 3),
                tuneLength = 10)
knn
knn$bestTune$k
class::knn(train.norm[,-c(2,3,4,8,9)],test.norm[,-c(2,3,4,8,9)],train.norm$Transport,k=15)
ktransport=knn(train.norm[,-c(2,3,4,8,9)],test.norm[,-c(2,3,4,8,9)],train.norm$Transport,k=15)
table(test.norm$Transport,ktransport)
knnpred.train = predict(knn, data = train.norm[-9], type = "raw")
confusionMatrix(knnpred.train,train.norm$Transport,positive="1")
knnpred.test = predict(knn, newdata = test.norm[-9], type = "raw")
knnpred.test
cm_knn=confusionMatrix(knnpred.test,test.norm$Transport,positive="1")
cm_knn
#****************************************************************************************************
#Naive Bayes
library(e1071)
NB = naiveBayes(x=train.norm[-c(2,3,4,8,9)], y=train.norm$Transport)
NB
NB.pred=predict(NB,type = "raw",newdata = train.norm)
NB.pred
par(mfrow=c(1,1))
plot(train.norm$Transport,NB.pred[,2])
NBpred.train = predict(NB, newdata = train.norm[-9])
confusionMatrix(NBpred.train, train.norm$Transport,positive="1")
NBpred.test = predict(NB, newdata = test.norm[-9])
cm_NB=confusionMatrix(NBpred.test,test.norm$Transport,positive="1")
cm_NB
#*****************************************************************************************************
#Modelling - Bagging, Boosting, SMOTE
#Bagging
library(xgboost)
library(ipred)
library(rpart)
library(caret)
bagtrain=train
bagtest=test
Transport.bagging=bagging(Transport~.,data = bagtrain,
                          control=rpart.control(maxdepth = 5,minsplit = 4))
bagtest$pred.class.bag=predict(Transport.bagging,bagtest)
bagtest$pred.class.bag
cm_bagging=confusionMatrix(data = factor(bagtest$pred.class.bag),
                reference = factor(bagtest$Transport),
                positive = "1")
cm_bagging
table.bagging=table(bagtest$Transport,bagtest$pred.class.bag==1)
table.bagging
#***************************************************************************************************
#Gradient Boosting using CARET Package. Since GBM package is not working properly.
library(caret)
sp=createDataPartition(transport$Transport,p=0.80,list = FALSE)
gb.train=transport[sp,]
gb.test=transport[-sp,]
gbmfit=caret::train(Transport~.,
                    data = gb.train,
                    method = "gbm",
                    trControl = trainControl(method = "repeatedcv",
                                             number = 5,
                                             repeats = 3,
                                             verboseIter = FALSE),
                    verbose = 0)
gbmfit
cm_gb=caret::confusionMatrix(data = predict(gbmfit,gb.test),
                             reference = gb.test$Transport)
cm_gb
#***************************************************************************************************
#Extreme Gradient Boosting
library(xgboost)
xgbtrain=train
xgbtest=test
xgbftrain=as.matrix(train[,-c(2,3,4,8,9)])
xgbltrain=as.matrix(train[,9])
xgbftest=as.matrix(test[,-c(2,3,4,8,9)])
xgbfit=xgboost::xgboost(
  data = xgbftrain,
  label = xgbltrain,
  eta = 0.001,
  max_depth = 3,
  min_child_weight = 3,
  nrounds = 100,
  nfold = 5,
  objective = "binary:logistic",
  verbose = 0,
  early_stopping_rounds = 10
)
xgbfit
xgbtest$pred.class.xgb=predict(xgbfit,xgbftest)
table.xgb=table(xgbtest$Transport,xgbtest$pred.class.xgb>0.5)
table.xgb
xgbtest$pred.class.xgb=ifelse(xgbtest$pred.class.xgb<0.5,0,1)
cm_xgb=caret::confusionMatrix(data = factor(xgbtest$pred.class.xgb),
                              reference = factor(xgbtest$Transport),
                              positive = "1")
cm_xgb
sum(xgbtest$Transport==1 & xgbtest$pred.class.xgb>=0.5)
#Extreme Gradient Boosting Tuning
t.xgb=vector()
l=c(0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1)
m=c(1,3,5,7,9,15)
n=c(2, 50, 100,1000,10000)
for (i in l) {
  xgbfit=xgboost::xgboost(
    data = xgbftrain,
    label = xgbltrain,
    eta = i,
    max_depth = 5,
    nrounds = 10,
    nfold = 5,
    objective = "binary:logistic",
    verbose = 0,
    early_stopping_rounds = 10
  )
  xgbtest$pred.class.xgb=predict(xgbfit,xgbftest)
  t.xgb=cbind(t.xgb,sum(xgbtest$Transport==1 & xgbtest$pred.class.xgb>=0.5))
}
t.xgb
#Best Fit
xgbfit=xgboost(
  data = xgbftrain,
  label = xgbltrain,
  eta = 0.7,
  max_depth = 5,
  nrounds = 20,
  nfold = 5,
  objective = "binary:logistic",
  verbose = 1,
  early_stopping_rounds = 10
)
xgbtest$pred.class.xgb=predict(xgbfit,xgbftest)
sum(xgbtest$Transport==1 & xgbtest$pred.class.xgb>=0.5)
table.xgb=table(xgbtest$Transport,xgbtest$pred.class.xgb>=0.5)
table.xgb
#**************************************************************************************************
#Adaptive Boosting
library(fastAdaboost)
adatrain=train
adatest=test
str(adatrain$Transport)
adaboost.fit=fastAdaboost::adaboost(Transport~.,data = as.data.frame(adatrain),nIter = 10)
adaboost.fit
ada.pred=predict(adaboost.fit,newdata = adatest)
cm_adaboost=caret::confusionMatrix(data = factor(ada.pred$class),
                                   reference = factor(adatest$Transport),
                                   positive = "1")
cm_adaboost
#***************************************************************************************************
#SMOTE
library(DMwR)
table(transport$Transport)
strain=subset(transport,splits==TRUE)
stest=subset(transport,splits==FALSE)
table(strain$Transport)
strain$Transport=as.factor(strain$Transport)
balanced.transport=DMwR::SMOTE(Transport~.,as.data.frame(strain),perc.over = 200,k=5,perc.under = 200)
table(balanced.transport$Transport)
sftrain=as.matrix(balanced.transport[,-c(2,3,4,8,9)])
sltrain=as.matrix(balanced.transport$Transport)
smote.xgb=xgboost::xgboost(
  data = sftrain,
  label = sltrain,
  eta = 0.7,
  max_depth = 5,
  nrounds = 50,
  nfold = 5,
  objective = "binary:logistic",
  verbose = 0,
  early_stopping_rounds = 10
)
smote.xgb
sftest=as.matrix(stest[,-c(2,3,4,8,9)])
stest$pred.class.smote=predict(smote.xgb,sftest)
stest$pred.class.smote=ifelse(stest$pred.class.smote<0.5,0,1)
cm_smote=caret::confusionMatrix(data = factor(stest$pred.class.smote),
                                reference = factor(stest$Transport),
                                positive = "1")
cm_smote
table.smote=table(stest$Transport,stest$pred.class.smote>=0.5)
table.smote
sum(stest$Transport==1 & stest$pred.class.smote>=0.5)
#***************************************************************************************************
#Model Comparison for Ensemble Methods
modelcomparison=c("cm_bagging","cm_gb","cm_xgb","cm_smote","cm_adaboost")
modelcomparison
table_ensemble=data.frame(Sensitivity = NA,
                 Specificity = NA,
                 Precision = NA,
                 Recall = NA,
                 F1 = NA)
for (i in seq_along(modelcomparison)) {
  model=get(modelcomparison[i])
  a=data.frame(Sensitivity = model$byClass["Sensitivity"],
               Specificity = model$byClass["Specificity"],
               Precision = model$byClass["Precision"],
               Recall = model$byClass["Recall"],
               F1 = model$byClass["F1"])
  rownames(a)=NULL
  table_ensemble=rbind(table_ensemble,a)
}
table_ensemble=table_ensemble[-1,]
row.names(table_ensemble)=c("BAGGING","GBM","XGB","SMOTE","ADABOOST")
table_ensemble
#**************************************************************************************************
#Model Comparison for Model Performance Matrices
modelcomp=c("cm_knn","cm_NB")
modelcomp
table_modelcomp=data.frame(Sensitivity = NA,
                           Specificity = NA,
                           Precision = NA,
                           Recall = NA,
                           F1 = NA)
table_modelcomp
for (i in seq_along(modelcomp)) {
  model1=get(modelcomp[i])
  b=data.frame(Sensitivity = model1$byClass["Sensitivity"],
               Specificity = model1$byClass["Specificity"],
               Precision = model1$byClass["Precision"],
               Recall = model1$byClass["Recall"],
               F1 = model1$byClass["F1"])
  rownames(b)=NULL
  table_modelcomp=rbind(table_modelcomp,b)
}
table_modelcomp=table_modelcomp[-1,]
row.names(table_modelcomp)=c("KNN","NAIVE BAYES")
table_modelcomp
#**************************************************************************************************