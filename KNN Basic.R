library(ISLR)
attach(Smarket)
train=(Year<2005)
Smarket.2005=Smarket[!train,]
dim(Smarket.2005)
Direction.2005=Direction[!train]

# K-Nearest Neighbors

library(class)
train.X=cbind(Lag1,Lag2)[train,]
test.X=cbind(Lag1,Lag2)[!train,]
train.Direction=Direction[train]
set.seed(1)
knn.pred=knn(train.X,test.X,train.Direction,k=1)
table(knn.pred,Direction.2005)
(83+43)/252
knn.pred=knn(train.X,test.X,train.Direction,k=3)
table(knn.pred,Direction.2005)
mean(knn.pred==Direction.2005)




For Iris:
  
  library(caret)
data(iris)
# fit model
fit <- knn3(Species~., data=iris, k=5)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, iris[,1:4], type="class")
# summarize accuracy
table(predictions, iris$Species)