library(keras)
library(tensorflow)

#mnist <- dataset_mnist()
#class(mnist)


# Use Iris with Keras
# Keras use matrix 
i <- iris

rng <- function(x){(x-min(x))/(max(x)-min(x))}       
ir <- as.data.frame(lapply(i[1:4], rng))    # apply rng function on all cols 1:4 are the x's
ir <- as.matrix(ir)                         # keras use matrices
dimnames(ir) <- NULL                       # dimension names in matrices are NULL

# use normalise function in keras
imatrix <- as.matrix(i[,1:4])
dimnames(imatrix)<-NULL
i_nrm <- normalize(imatrix)

## Let's keep the data without normalising

id <- sample(2,nrow(iris),replace = T,prob = c(0.67,0.33))

iris.training <- as.matrix(iris[id==1,1:4])
iris.test <- as.matrix(iris[id==2,1:4])

# Take the target in a seperate vector

training_target_label <- iris[id==1,5]
test_target_label <- iris[id==2,5]

# In case of multinomial classification this is done to convert the levels of target to dummy variables
iris_train_label <- dummies::dummy(training_target_label)
iris_test_label <- dummies::dummy(test_target_label)

### Start with modelling ###

model <- keras_model_sequential()

model <- model %>% layer_dense(units = 8,activation = "relu",input_shape = c(4)) %>%
                layer_dense(units = 3,activation = "softmax")

# two different activation functions in the hidden layer - relu and at output layer -softmax to get in range of 0-1 as probabilities

# 1st layer has 8 hidden nodes and input shape = 4 as coloumns are 4 = no of predictors
# output layer has 3 nodes as 3 classes to predict


# Inspect model with 
summary(model)
get_config(model)
get_layer(model,index = 1)
model$layers
model$inputs
model$outputs



# Compile and fit the model 

model <- model %>% compile(
                          loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics='accuracy')
# Fit

model <- model %>% fit(
                        iris.training,
                        iris_train_label,
                        epochs=200,
                        batch_size=5,
                        validation_split=0.2)
plot(model)

dimnames(iris.test)<- NULL
dimnames(iris.training) <-  NULL
predictions <- model %>%  predict_classes(iris.test,batch_size = 128)

predictions <- predict_classes(model,iris.test,batch_size = 128)


