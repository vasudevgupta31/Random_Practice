# Ridge Regression

# load the package
library(glmnet)
# load data
data(longley)
str(longley)
x <- as.matrix(longley[,1:6])
y <- as.matrix(longley[,7])

pairs(longley, main = "longley data")
summary(fm1 <- lm(Employed ~ ., data = longley))

opar <- par(mfrow = c(2, 2), oma = c(0, 0, 1.1, 0),
            mar = c(4.1, 4.1, 2.1, 1.1))
plot(fm1)
par(opar)


# fit model
fit.ridge <- glmnet(x, y, family="gaussian", alpha=0, lambda=0.001)       #lambda intially given , alpha for RIDGE = 0 , for LASSO 
# summarize the fit
print(fit.ridge)


# make predictions
predictions <- predict(fit.ridge, x, type="link")
# summarize accuracy
mse.ridge <- mean((y - predictions)^2)
print(mse.ridge)

# 3 components in GLM, random(The random component states the probability distribution
# of the response variable.), systematic(It specifies the linear combination of the explanatory
#variables, it consist in a vector of predictors) and link(It connects the random and the systematic component. It
#shows how the expected value of the response variable is connected to the linear predictor of explanatory variables)


# Least Absolute Shrinkage and Selection Operator    LASSO !!

# load the package
library(lars)
# load data
data(longley)
x <- as.matrix(longley[,1:6])
y <- as.matrix(longley[,7])
# fit model
fit.las <- lars(x, y, type="lasso")
#summarize the fit
print(fit.las)
fit.las$beta
#Step size is how lambda changes between each calculation of the model.
#Best step is just labeling the chosen model from amongst the family of models. The example uses minimum RSS as the metric for best.
#select a step with a minimum error
best_step <- fit.las$df[which.min(fit$RSS)]
# make predictions
predictions <- predict(fit.las, x, s=best_step, type="fit")$fit
# summarize accuracy
mse.lasso <- mean((y - predictions)^2)
print(mse.lasso)


pred.lm <- predict(fm1,as.data.frame(x))
mse.lm <- mean((y - pred.lm)^2)
mselm


