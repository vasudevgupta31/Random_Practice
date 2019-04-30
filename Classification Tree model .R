### Importing and cleaning/structuring data ###

hr<-read.csv("hr_att.csv",header = T)
dim(hr)

str(hr)
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

dim(hr)
str(hr)

library(rpart)
library(rpart.plot)

#EDA
library(DataExplorer)


plot_str(hr)         #basic str
plot_missing(hr)       #check for missing values

plot_histogram(hr)     #histograms for all continuous variables
plot_density(hr)       #density plots for all continuous variables    #try combining this with bar
plot_bar(hr)           #frequency plots for all categorical variables

str(hr)

                                             ###### Classification Tree ####

library(rpart)
library(rpart.plot)

#creating test and training set

set.seed(1234)
sample(1:2940,2040)->id
train_ct<-hr[id,]
test_ct<-hr[-id,]

#overfit tree to check CP value
rpart(Attrition~. , data = train_ct , method = 'class', control = rpart.control(cp=0.001) )->ct1  
printcp(ct1)
rpart.plot(ct1)

#Variables actually used in tree construction:
#[1] Age                      DailyRate                DistanceFromHome         Education                EducationField          
#[6] EnvironmentSatisfaction  HourlyRate               JobInvolvement           JobLevel                 JobRole                 
#[11] JobSatisfaction          MaritalStatus            MonthlyIncome            MonthlyRate              NumCompaniesWorked      
#[16] OverTime                 PercentSalaryHike        RelationshipSatisfaction StockOptionLevel         TotalWorkingYears       
#[21] TrainingTimesLastYear    WorkLifeBalance          YearsWithCurrManager    


prune(ct1,cp=0.0169)->pt    
rpart.plot(pt,type = 2)
printcp(pt)
library(rattle)
fancyRpartPlot(pt)
#Variables actually used in tree construction:
#[1] HourlyRate            JobRole               JobSatisfaction       MaritalStatus         MonthlyIncome         OverTime             
#[7] TotalWorkingYears     TrainingTimesLastYear

#Root node error of 17.4%

predict(pt,test_ct,'class')->pclass
table(actual=test_ct$Attrition,predicted=pclass)      #prediction on test data has missclassification of 12.2%

library(ROCR)

predict(pt,test_ct,'prob')->pprob
pprob[,2]->p
prediction(p,test_ct$Attrition)->predobjct

perf1<-performance(predobjct,'tpr','fpr')            #checking cut off value for probabilities according TPR and FPR distribution
plot(perf1,print.cutoffs.at=seq(0,1,0.2))

perf2<-performance(predobjct,'acc')            #accuracy remains constant after cut off of 0.4 till 0.8, after which drops a bit
plot(perf2)

perf3<-performance(predobjct,'auc')
perf3                                     #AUC of 0.6848
library(caret)
table(actual=test_ct$Attrition,predicted=pclass)->t_ct
confusionMatrix(t_ct)

perf4<-performance(predobj,'lift')
plot(perf4)
