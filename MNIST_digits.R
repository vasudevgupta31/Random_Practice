require(dplyr)
require(keras)
require(tensorflow)
use_backend("tensorflow")


# CNN for CIFAR IMAGES

mnist <- dataset_mnist()

train.x <- mnist$train$x
train.y <- to_categorical(mnist$train$y,10)
range(train.y)

test.x <- mnist$test$x
test.y <- to_categorical(mnist$test$y,10)

dim(train.x)
dim(train.y)
dim(test.x)
dim(test.y)

# Convert the image data which is a 3d array to a matrix by reshaping it and converting these 28x28 values to coloumns

dim(train.x) <- c(nrow(train.x),28^2)
dim(test.x) <- c(nrow(test.x),28^2)

# Scale the data. Since the values are all between 0-255 we can scale it by simply dividing by 255

train.x <- train.x/255
test.x <- test.x/255
# Defining the model

model <- keras_model_sequential()

model %>% layer_dense(units = 256,activation = 'relu',input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128,activation = 'relu') %>%
  layer_dropout(rate=0.3)%>%
  layer_dense(units = 10,activation = 'softmax')

summary(model)

model %>% compile(
  loss='categorical_crossentropy',
  optimizer=optimizer_rmsprop(),
  metrics=c('accuracy')
  )

history <- model %>% fit(train.x,train.y,
                         epochs=30,
                         batch_size=128,
                         validation_split=0.2,
                         callback=list(callback_reduce_lr_on_plateau(monitor = 'val_loss',factor = 0.1,patience = 2,verbose = 1,mode = 'min',cooldown = 2),
                                       callback_early_stopping(monitor = 'val_loss',patience = 10,mode = 'min')))

model %>% evaluate(test.x,test.y,verbose=2)

preds <- model %>% predict_classes(test.x)
actuals <- mnist$test$y

caret::confusionMatrix(table(preds,actuals))


