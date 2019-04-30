require(keras)
require(tensorflow)
require(dplyr)
require(tibble)
require(tidyr)
require(ggplot2)
require(purrr)
require(tensorflow)

fashion_mnist <- dataset_fashion_mnist()
str(fashion_mnist)

c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test



class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

#### Explore the data ####


# Train Set
dim(train_images)  #60,000 images in the training set
# with each image represented as 28 x 28 pixels

dim(train_labels) # These are the labels  for the training images
range(train_labels)

# Test set
dim(test_images) # 10,000 images for test dataset with the same 28x28 pixel values
dim(test_labels)  # labels for the test images



## Pre-processing ##

train_images <- train_images/255
test_images <- test_images/255


# Model Building #
# Create a baseline model

baseline_model <- keras_model_sequential() %>% layer_flatten(input_shape = c(28,28)) %>%
                                        layer_dense(units = 50,activation = 'relu') %>%
                                        layer_dense(units = 10,activation = 'softmax')
baseline_model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = list("accuracy"))

baseline_model %>% summary()


baseline_history <- baseline_model %>% fit(
                                           train_images,
                                            train_labels,
                                          epochs = 10,
                                           batch_size = 5,
                                          validation_data = list(test_images,test_labels),
                                               verbose = 2)



# Create a smaller model

smaller_model <- keras_model_sequential() %>% layer_flatten(input_shape = c(28,28)) %>%
                                        layer_dense(units = 7,activation = 'relu') %>%
                                      layer_dense(units = 10,activation = 'softmax')

smaller_model %>% compile(
                          optimizer = "adam",
                          loss = "sparse_categorical_crossentropy",
                          metrics = list("accuracy"))

smaller_model %>% summary()


smaller_history <- smaller_model %>% fit(
  train_images,
  train_labels,
  epochs = 10,
  batch_size = 5,
  validation_data = list(test_images,test_labels),
  verbose = 2)


# Create a bigger model

bigger_model <- keras_model_sequential() %>% layer_flatten(input_shape = c(28,28)) %>%
  layer_dense(units = 400,activation = 'relu') %>%
  layer_dense(units = 10,activation = 'softmax')

bigger_model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = list("accuracy"))

bigger_model %>% summary()


bigger_history <- bigger_model %>% fit(
  train_images,
  train_labels,
  epochs = 10,
  batch_size = 5,
  validation_data = list(test_images,test_labels),
  verbose = 2)


# Comparing the 3 models 

compare_cx <- data.frame(
  baseline_train = baseline_history$metrics$loss,
  baseline_val = baseline_history$metrics$val_loss,
  smaller_train = smaller_history$metrics$loss,
  smaller_val = smaller_history$metrics$val_loss,
  bigger_train = bigger_history$metrics$loss,
  bigger_val = bigger_history$metrics$val_loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)

ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")
#

## Regularisation ##

# 1. Weight Regularisation

l2_model <- keras_model_sequential() %>% layer_flatten(input_shape = c(28,28)) %>% 
                                         layer_dense(units = 20,activation = "relu",kernel_regularizer = regularizer_l2(l=0.001)) %>%
                                        layer_dense(units = 10,activation = 'softmax')

l2_model %>% compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics='accuracy')                                

l2_history <- l2_model %>% fit(
  train_images,
  train_labels,
  epochs = 20,
  batch_size = 5,
  validation_data = list(test_images, test_labels),
  verbose = 2
)           

# Compare Weight Regularisation

compare_cx <- data.frame(
  bigger_train = bigger_history$metrics$loss,
  bigger_val = bigger_history$metrics$val_loss,
  l2_train = l2_history$metrics$loss,
  l2_val = l2_history$metrics$val_loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)

ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")



# 2. Dropout Regularisation

dropout_model <- keras_model_sequential() %>% layer_flatten(input_shape = c(28,28)) %>%
                                              layer_dense(units = 400,activation = "relu") %>% layer_dropout(0.4) %>%
                                              layer_dense(units=10,activation = 'softmax')

dropout_model %>% compile(optimizer='adam',
                                             metrics='accuracy',
                                             loss='sparse_categorical_crossentropy')

dropout_history <- dropout_model %>% fit(train_images,
                                         train_labels,
                                         epochs=10,
                                         batch_size=5,
                                         validation_data=list(test_images,test_labels),
                                         verbose=2)

# Compare Dropout Regularisation
compare_cx <- data.frame(
  bigger_train = bigger_history$metrics$loss,
  bigger_val = bigger_history$metrics$val_loss,
  dropout_train = dropout_history$metrics$loss,
  dropout_val = dropout_history$metrics$val_loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%s
  gather(key = "type", value = "value", -rowname)

ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")
