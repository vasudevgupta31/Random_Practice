library(ISLR)
attach(Smarket)
train=(Year<2005)
Smarket.2005=Smarket[!train,]
dim(Smarket.2005)
Direction.2005=Direction[!train]
library(e1071)
snb=naiveBayes(Direction~Lag1+Lag2,Smarket,subset=train)
snb.pred=predict(snb,Smarket.2005)
table(snb.pred,Direction.2005)
mean(snb.pred==Direction.2005)

#QDA
library(MASS)
qda.fit=qda(Direction~Lag1+Lag2,data=Smarket,subset=train)
qda.class=predict(qda.fit,Smarket.2005)$class
table(qda.class,Direction.2005)
mean(qda.class==Direction.2005)
table(qda.class,snb.pred)
