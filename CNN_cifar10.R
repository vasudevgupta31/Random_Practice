# Simple deep CNN

require(dplyr)
require(keras)
require(tensorflow)
use_backend("tensor")


cifar <- dataset_cifar10()

str(cifar)
class_names=c('airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks')

# Feature Scale RGB values in test and train 
train.x <- cifar$train$x/255
train.y <- to_categorical(cifar$train$y,num_classes = 10)

test.x <- cifar$test$x/255
test.y <- to_categorical(cifar$test$y,num_classes = 10)

# Understand the structure of data
# Train set
dim(train.x) # 50,000 images with 32x32 pixels for 3 colours 
dim(train.y) # 50,000  labels

# Test Set
dim(test.x)     # 10,000 images with 32x32 pixels for 3 colours
dim(test.y)


# Model Building #
#1. Wrapper
model <- keras_model_sequential()

#2. Layers
model %>% 
  # Start with hidden 2D convolution layer being fed 32x32 pixel images
  layer_conv_2d(filter= 32, kernel_size = c(3,3),padding="same",input_shape = c(32,32,3)) %>% layer_activation("relu") %>%
  # Second hidden Layer
  layer_conv_2d(filter=32,kernel_size = c(3,3)) %>% layer_activation("relu") %>%
  # max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>% layer_dropout(0.25) %>%
  # 2 additional hidden 2D convolution layers
  layer_conv_2d(filter=32,kernel_size = c(3,3),padding = "same") %>% layer_activation("relu") %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),padding = "same") %>% layer_activation("relu") %>%
  #max pooling once more
  layer_max_pooling_2d(pool_size = c(2,2))%>% layer_dropout(0.25)%>%
  # Flatten this max pooled layer for classification - deep network
  layer_flatten()%>%
  layer_dense(units = 512,activation = "relu") %>% layer_dropout("relu")%>%
  # output layer for 10 classes
  layer_dense(units=10,activation = "softmax")

#3. Compile

model %>% compile(loss='categorical_crossentropy',
                  optimizer=optimizer_rmsprop(lr=0.0001,decay = 1e-6),
                  metrics=c('accuracy'))
summary(model)
                  
#4. Fit

history <- model %>% fit(train.x,train.y,
                         batch_size=32,
                         epochs=100,
                         validation_split=0.2,
                         shuffle=T)
