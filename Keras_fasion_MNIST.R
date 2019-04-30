require(keras)
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



#### Pre-processing ####

require(tidyr)
require(ggplot2)

image_1 <- as.data.frame(train_images[1,,])     # that's how each image is stored as numbers 
colnames(image_1) <- seq_len(ncol(image_1))     # change col names to numbers 1-28
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)


ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")

# if the above plot gives error use dev.off()

# the first image in the training set, pixel values fall in the range of 0 to 255

# Let's scale the data to 0-1 for both train and test by simply dividing by 255

train_images <- train_images/255
test_images <- test_images/255

# Check first 25 images in the train and test sets

par(mfcol=c(5,5))
par(mar=c(0,0,1.5,0),xaxes='i',yaxes='i')
for (i in 1:25) { 
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
       main = paste(class_names[train_labels[i] + 1]))
}


## Model Building ##

# 1. Wrapper
model <- keras_model_sequential()

# 2. Layers

model <- model %>% layer_flatten(input_shape = c(28,28)) %>%
                   layer_dense(units = 150,activation = 'relu') %>%
                  layer_dense(units = 10,activation = 'softmax')

#The first layer in this network, layer_flatten, transforms the format of the images 
#from a 2d-array (of 28 by 28 pixels),to a 1d-array of 28 * 28 = 784 pixels
# This layer has no parameters to learn; it only reformats the data.

#After the pixels are flattened, the network consists of a sequence of two dense layers. 
#These are densely-connected, or fully-connected, neural layers. 
#The first dense layer has 128 nodes (or neurons). 
#The second (and last) layer is a 10-node softmax layer —this returns an array of 10 probability scores that sum to 1. 
#Each node contains a score that indicates the probability that the current image belongs to one of the 10 digit classes.


# 3. Compile
# Loss function — This measures how accurate the model is during training. We want to minimize this function to “steer” the model in the right direction.
# Optimizer — This is how the model is updated based on the data it sees and its loss function.
# Metrics —Used to monitor the training and testing steps. 

model %>% compile(loss="sparse_categorical_crossentropy",
                  metrics=c('accuracy'),
                  optimizer= 'adam')

# 4. Train the model

model %>% fit(train_images,train_labels,epochs=10,validation_split=0.2) 



# Evaluate model on test dataset

score <- model %>% evaluate(test_images,test_labels)
score$loss
score$acc

cat('Test loss:',score$loss,"\n")
cat('Test Acc:',score$acc,"\n")    # Shows overfitting

# Make predictions

predictions <- model %>% predict(test_images)   # predicting probabilities
predictions[1,]
which.max(predictions[1,])

classes <- model %>% predict_classes(test_images,batch_size = 7)
classes[1:20]
test_labels[1:20]

confusionMatrix(table(classes,test_labels))

# Check with plots
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800' 
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}
