
library(purrr)
library(readr)
library(dplyr)
library(EBImage)
library(stringr)
library(pbapply)
library(keras)

# Data Processing ---------------------------------------------------------

set.seed(1)
train_index <- sample(1:1000, 900)
file_names_train <- list.files("data/cats_and_dogs/train", full.names = TRUE)[train_index]
file_names_test <- list.files("data/cats_and_dogs/train", full.names = TRUE)[-train_index]

# Take all images (train)

pb <- progress_estimated(length(file_names_train))
image_train.list <- map(
  file_names_train,
  function(file_name) {
    pb$tick()$print()
    image.img <- readImage(file_name)
    image_resized.img <- resize(image.img, w = 128, h = 128)
    image_resized.mat <- array_reshape(image_resized.img, dim = c(128, 128, 3))
    return(image_resized.mat)
  }
)

# Take all images (test)

pb <- progress_estimated(length(file_names_test))
image_test.list <- map(
  file_names_test,
  function(file_name) {
    pb$tick()$print()
    image.img <- readImage(file_name)
    image_resized.img <- resize(image.img, w = 128, h = 128)
    image_resized.mat <- array_reshape(image_resized.img, dim = c(128, 128, 3))
    return(image_resized.mat)
  }
)

# For Viewing

EBImage::combine(
  map(
    image_train.list[1:36],
    ~Image(., colormode = Color)
  )
) %>% 
  display(method = "raster", all = TRUE)

# Combine all into an array (ready for keras consumption)

image_train.array <- image_train.list %>% 
  abind(along = 4) %>% 
  aperm(c(4,1,2,3))

image_test.array <- image_test.list %>% 
  abind(along = 4) %>% 
  aperm(c(4,1,2,3))

# Labels 

labels_train <- ifelse(str_detect(list.files("data/cats_and_dogs/train"), "cat.")[train_index], 1, 0)
labels_test <- ifelse(str_detect(list.files("data/cats_and_dogs/train"), "cat.")[-train_index], 1, 0)


# Parameters --------------------------------------------------------------

input_shape <- c(128, 128, 3) # height, width, channel

# try reshaping -----------------------------------------------------------

image_train.array.128.384 <- image_train.array %>% array_reshape(dim = c(800, 128, 128 * 3))
image_train.array.128.384 %>% 
  array_reshape(dim = c(800, 128, 128, 3)) %>% 
  {.[1,,,]} %>% Image(colormode = Color) %>% display()

# Neural Net --------------------------------------------------------------

# Define model
nn.model <- keras_model_sequential() %>%
  layer_dense(units = 900, activation = "relu", input_shape = 128 * 128 * 3) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

# Compile
nn.model %>%
  compile(loss = 'binary_crossentropy',
          optimizer = optimizer_adam(),
          metrics = "accuracy")

# Train Model
nn.model %>%
  fit(image_train.array %>% array_reshape(dim = c(900, 128 * 128 * 3)),
      labels_train,
      epochs = 4,
      batch_size = 32,
      validation_split = 0.2) # could be set to 0

# Scores
nn.model %>% evaluate(
  image_test.array %>% array_reshape(dim = c(100, 128 * 128 * 3)),
  labels_test,
  verbose = 0
)

# Probabilities
nn_model.pred <- nn.model %>% predict_proba(image_test.array %>% array_reshape(dim = c(100, 128 * 128 * 3)))

# Plot
by <- 4
EBImage::combine(
  map(
    image_test.list[1:by^2],
    ~Image(., colormode = Color)
  )
) %>% 
  display(method = "raster", all = TRUE)
row_indeces <- seq(0, 128*(by-1), 128)
column_indeces <- seq(0, 128*(by-1), 128)
counter <- 0
for (i in row_indeces) {
  for (j in column_indeces) {
    counter <- counter + 1
    text(
      x = i + 5, y = j + 5,
      label = ifelse(nn_model.pred[counter] >= 0.5, "Cat", "Dog"),
      adj = c(0, 1), col = ifelse(nn_model.pred[counter] >= 0.5, "Green", "Red"),
      cex = 2
    )
  }
}

# Convolutional Neural Network --------------------------------------------

# Define model
cnn.model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(4, 4), activation = "relu", input_shape = c(128, 128, 3), stride = 2) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 16, kernel_size = c(2, 2), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 4, kernel_size = c(2, 2), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 196, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

# Compile
cnn.model %>%
  compile(loss = 'binary_crossentropy',
          optimizer = optimizer_adam(),
          metrics = "accuracy")

# Train Model
cnn.model %>%
  fit(image_train.array,
      labels_train,
      epochs = 12,
      batch_size = 128,
      validation_split = 0.2) # could be set to 0

# Scores
cnn.model %>% evaluate(
  image_test.array,
  labels_test,
  verbose = 0
)

# Probabilities
cnn_model.pred <- cnn.model %>% predict_proba(image_test.array)

# Plot
by <- 10
EBImage::combine(
  map(
    image_test.list[1:by^2],
    ~Image(., colormode = Color)
  )
) %>% 
  display(method = "raster", all = TRUE)
row_indeces <- seq(0, 128*(by-1), 128)
column_indeces <- seq(0, 128*(by-1), 128)
counter <- 0
for (i in row_indeces) {
  for (j in column_indeces) {
    counter <- counter + 1
    text(
      x = i + 2.5, y = j + 2.5,
      label = ifelse(cnn_model.pred[counter] >= 0.5, "Cat", "Dog"),
      adj = c(0, 1), col = ifelse(cnn_model.pred[counter] >= 0.5, "Green", "Red"),
      cex = 1
    )
  }
}


# CNN Take 2 --------------------------------------------------------------

# Define model
cnn2.model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(128, 128, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 16, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dense(units = 28624, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

# Compile
cnn2.model %>%
  compile(loss = 'binary_crossentropy',
          optimizer = optimizer_adam(),
          metrics = "accuracy")

# Train Model
cnn2.model %>%
  fit(image_train.array,
      labels_train,
      epochs = 24,
      batch_size = 128,
      validation_split = 0.2) # could be set to 0

# Scores
cnn2.model %>% evaluate(
  image_test.array,
  labels_test,
  verbose = 0
)

# Probabilities
cnn2_model.pred <- cnn2.model %>% predict_proba(image_test.array)

# Plot
by <- 10
EBImage::combine(
  map(
    image_test.list[1:by^2],
    ~Image(., colormode = Color)
  )
) %>% 
  display(method = "raster", all = TRUE)
row_indeces <- seq(0, 128*(by-1), 128)
column_indeces <- seq(0, 128*(by-1), 128)
counter <- 0
for (i in row_indeces) {
  for (j in column_indeces) {
    counter <- counter + 1
    text(
      x = i + 2.5, y = j + 2.5,
      label = ifelse(cnn_model.pred[counter] >= 0.5, "Cat", "Dog"),
      adj = c(0, 1), col = ifelse(cnn_model.pred[counter] >= 0.5, "Green", "Red"),
      cex = 1
    )
  }
}

