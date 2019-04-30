## Reading and structuring data for a neural network model ##

hr<-read.csv("hr_att.csv",header = T)

hr$Education<-as.factor(hr$Education)
hr$EnvironmentSatisfaction<-as.factor(hr$EnvironmentSatisfaction)
hr$JobInvolvement<-as.factor(hr$JobInvolvement)
hr$JobLevel<-as.factor(hr$JobLevel)
hr$JobSatisfaction<-as.factor(hr$JobSatisfaction)
hr$PerformanceRating<-as.factor(hr$PerformanceRating)
hr$RelationshipSatisfaction<-as.factor(hr$RelationshipSatisfaction)
hr$StockOptionLevel<-as.factor(hr$StockOptionLevel)
hr$WorkLifeBalance<-as.factor(hr$WorkLifeBalance)
hr$TrainingTimesLastYear<-as.factor(hr$TrainingTimesLastYear)

hr$EmployeeCount<-NULL
hr$EmployeeNumber<-NULL
hr$Over18<-NULL
hr$StandardHours<-NULL

str(hr)

#TAKING ONLY VARIABLES WHICH HAVE BEEN USED TO CREATE A CLASSIFICATION TREE
##Variables actually used in tree construction:
#[1] Age                      DailyRate                DistanceFromHome         Education                EducationField          
#[6] EnvironmentSatisfaction  HourlyRate               JobInvolvement           JobLevel                 JobRole                 
#[11] JobSatisfaction          MaritalStatus            MonthlyIncome            MonthlyRate              NumCompaniesWorked      
#[16] OverTime                 PercentSalaryHike        RelationshipSatisfaction StockOptionLevel         TotalWorkingYears       
#[21] TrainingTimesLastYear    WorkLifeBalance          YearsWithCurrManager 

names(hr)
hr[,c(1,2,4,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,31)]->d       #choosing variables from CART
str(d)
dim(d)
library(fastDummies)
dummy_cols(d[,-2])->dums
dums[,24:84]->dums1
names(dums1)
dums1<-dums1[,-c(2,6,15,19,21,25,37,40,42,43,47,51,58)]         #n-1 dummy variables
str(d)
hr1<-d[,c(1,2,3,4,8,14,15,16,18,21,24)]



cbind(hr1,dums1)->df           #Dataset with selected variables from a tree(their dummy variables) 

str(df)

df$Attrition<-as.integer(ifelse(df$Attrition=="Yes",1,0))
str(df)

#converting all variables in 0-1 range

rng01<-function(x){(x-min(x))/(max(x)-min(x))}
as.data.frame(apply(df,2,rng01))->rngdf        
names(rngdf)         #names have spaces and '-' in them converting all names to Syntactically valid names
names(rngdf)<-make.names(names(rngdf),unique = T)
names(rngdf)

str(rngdf)             # #final dataframe for neural network

#dividing in train and test set
set.seed(1234)
sample(1:2940,2040)->id
train_nn<-rngdf[id,]
test_nn<-rngdf[-id,]

paste(names(rngdf)[-2],collapse = '+')->s
s
paste("Attrition ~ ",s)->s
s


####### Making a neural network ######
library(neuralnet)

neuralnet(as.formula(s),data = train_nn,hidden =c(50,40),err.fct = 'ce',act.fct = 'logistic',linear.output = F)->nn

nn$result.matrix

compute(nn,test_nn[-2])->rslt           #predicting with the network
rslt$net.result->p
p
quantile(rslt$net.result,c(.1,.2,.6,.9))         #fairly distributed range of predictions

library(ROCR)

ROCR::prediction(p,test_nn$Attrition)->predobj

perf1<-performance(predobj,'auc')            #auc of 0.89
perf1

perf2<-performance(predobj,'tpr','fpr')
plot(perf2,print.cutoffs.at=seq(0,1,0.5))         

perf3<-performance(predobj,'acc')       #cut off between 0.2 to 0.5 gives the same accuracy
plot(perf3)


ifelse(p>0.5,1,0)->nnpredict
nnpredict

table(actual=test_nn$Attrition,predicted=nnpredict)->t1    # Mis classification of 3.7 % only

library(caret)
confusionMatrix(t1)




########################### Creating a Random Forest Model ######################
#since variable after dummies is a nice big number random forest might be a nice idea
library(randomForest)

randomForest(Attrition~.,data = train_ct,ntree =20,mtry=8,method='class')->rf
rf
plot(rf)
importance(rf)          #importance of variables

tuneRF(x=train_ct[,-2],y=train_ct$Attrition,mtryStart = 4 ,ntreeTry = 20,
       stepFactor = 2,improve = 0.0000001,doBest = T, trace = T, plot = T)->rftune

predict(rf,test_ct)->prf
table(actual=test_ct$Attrition,predicted=prf)     #4.7 % Error

predict(rf,test_ct,'prob')->probrf
probrf[,2]->probabrf


#### ensemble (avg) ######  

p->p_nn                    #probab from Neural Net
pprob[,2]->p_ct               #prob from decision tree
probabrf->p_rf                  #prob from random forest

p_ens=(p_nn+p_ct+p_rf)/3

ifelse(p_ens>0.5,1,0)->p_avg

table(actual=test_ct$Attrition,predicted=p_avg)->t_ens      # error of 3.7 % in average ensemble method
t_ens
p_avg 

##### ENSEMBLE MAJORITY###

class(pclass)
class(prf)
nn_c<-as.factor(nnpredict)
ct_c<-as.factor(pclass)
rf_c<-as.factor(prf)

cbind(nn_c,ct_c)->ens_c
cbind(ens_c,rf_c)->ens_c

as.data.frame(ens_c)->ens_c
ens_c$nn_c<-as.factor(ens_c$nn_c)
ens_c$ct_c<-as.factor(ens_c$ct_c)
ens_c$rf_c<-as.factor(ens_c$rf_c)

ifelse(ens_c$nn_c=='2',3,1)->ens_c$nn_c
ifelse(ens_c$nn_c=='1',0,3)->ens_c$nn_c
ifelse(ens_c$nn_c=='3',1,0)->ens_c$nn_c

ifelse(ens_c$ct_c=='2',3,1)->ens_c$ct_c
ifelse(ens_c$ct_c=='1',0,3)->ens_c$ct_c
ifelse(ens_c$ct_c=='3',1,0)->ens_c$ct_c

ifelse(ens_c$rf_c='2',3,1)->ens_c$rf_c
ifelse(ens_c$rf_c=='1',0,3)->ens_c$rf_c
ifelse(ens_c$rf_c=='3',1,0)->ens_c$rf_c

str(ens_c)

ifelse(nn_c=='0' & ct_c=='0',0,ifelse(nn_c=='0' & rf_c=='0',0,ifelse(ct_c=='0' & rf_c=='0',0,1)))->maj_ens

maj_ens




## ensemble of weighted average is same as average
ens_wtd_avg<-(p_nn*0.30)+(p_ct*0.20)+(p_rf*0.50)

ifelse(ens_wtd_avg>0.5,1,0)->ens_wavg

table(actual=test_ct$Attrition,predicted=ens_wavg)


