library(keras)
library(EBImage)
library(dplyr)
library(purrr)

mnist <- dataset_mnist()

mnist$train$x[7,,] %>% 
  aperm(c(2,1)) %>% 
  Image(colormode = "Grayscale") %>% 
  display(method = "raster")

# Build Train Set ---------------------------------------------------------

pb <- progress_estimated(nrow(mnist$train$x))

train.list <- list(
  train_x =
    abind(
      lapply(
        1:nrow(mnist$train$x),
        function(x) {
          pb$tick()$print()
          # Turn to 0, 1 scale
          (mnist$train$x[x,,]/255) %>% 
            aperm(c(2,1))
        }
      ),
      along = 3
    ) %>% 
    aperm(c(3,1,2)),
  train_y = to_categorical(mnist$train$y, 10) 
)

pb <- progress_estimated(nrow(mnist$test$x))

test.list <- list(
  test_x =
      abind(
        lapply(
          1:nrow(mnist$test$x),
          function(x) {
            pb$tick()$print()
            # Turn to 0, 1 scale
            (mnist$test$x[x,,]/255) %>% 
              aperm(c(2,1))
          }
        ),
        along = 3
      ) %>% 
      aperm(c(3,1,2)),
    test_y = to_categorical(mnist$test$y, 10) 
  )


# Global Vars -------------------------------------------------------------

img_length <- 28
img_width <- 28
img_channels <- 1
batch_size <- 128
num_classes <- 10
epochs <- 12

# Utility Functions -------------------------------------------------------

convert_vector_to_prediction <- function(prob_vector) {
  which(prob_vector == max(prob_vector)) - 1
}

# Model Training ----------------------------------------------------------


# Traditional Neural Network ----------------------------------------------

nn.model <- keras_model_sequential() %>%
  layer_flatten() %>% 
  layer_dense(units = img_width * img_length * img_channels, 
              activation = "relu") %>% 
  layer_dense(units = img_width * img_length * img_channels / 2,
              activation = "relu") %>% 
  layer_dense(units = img_width * img_length * img_channels / 4,
              activation = "relu") %>% 
  layer_dense(units = num_classes, activation = "softmax")

# Compile model
nn.model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Train model
nn.model %>% fit(
  train.list$train_x, 
  train.list$train_y,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
)

# Scores
nn.model %>% evaluate(
  test.list$test_x,
  test.list$test_y,
  verbose = 0
)

# Probabilities
nn_model.pred <- nn.model %>% 
  predict_proba(test.list$test_x)

# Plot
by <- 10
EBImage::combine(
  map(
    1:by^2,
    function(x) {
      test.list$test_x[x,,] %>% 
        Image(colormode = "GrayScale")
    }
  )
) %>% 
  display(method = "raster", all = TRUE)
row_indeces <- seq(0, img_width*(by-1), img_width)
column_indeces <- seq(0, img_length*(by-1), img_length)
counter <- 0
for (i in row_indeces) {
  for (j in column_indeces) {
    counter <- counter + 1
    text(
      x = j + 1, y = i + 1,
      label = convert_vector_to_prediction(nn_model.pred[counter, ]),
      adj = c(0, 1), 
      col = ifelse(
        convert_vector_to_prediction(nn_model.pred[counter, ]) == 
          convert_vector_to_prediction(test.list$test_y[counter, ]),
        "Green",
        "Red"
      ),
      cex = 2
    )
  }
}

# Traditional Neural Network w/ Drop Out ----------------------------

nn_do.model <- keras_model_sequential() %>%
  layer_flatten() %>% 
  layer_dense(units = img_width * img_length * img_channels, 
              activation = "relu") %>% 
  layer_dense(units = img_width * img_length * img_channels / 2,
              activation = "relu") %>% 
  layer_dense(units = img_width * img_length * img_channels / 4,
              activation = "relu") %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = num_classes, activation = "softmax")

# Compile model
nn_do.model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Train model
nn_do.model %>% fit(
  train.list$train_x, 
  train.list$train_y,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
)

# Scores
nn_do.model %>% evaluate(
  test.list$test_x,
  test.list$test_y,
  verbose = 0
)

# Probabilities
nn_do_model.pred <- nn_do.model %>% 
  predict_proba(test.list$test_x)

# Plot
nn_do_model.pred %>% {
  by <- 15
  EBImage::combine(
    map(
      1:by^2,
      function(x) {
        test.list$test_x[x,,] %>% 
          Image(colormode = "GrayScale")
      }
    )
  ) %>% 
    display(method = "raster", all = TRUE)
  row_indeces <- seq(0, img_width*(by-1), img_width)
  column_indeces <- seq(0, img_length*(by-1), img_length)
  counter <- 0
  for (i in row_indeces) {
    for (j in column_indeces) {
      counter <- counter + 1
      text(
        x = j + 1, y = i + 1,
        label = convert_vector_to_prediction(.[counter, ]),
        adj = c(0, 1), 
        col = ifelse(
          convert_vector_to_prediction(.[counter, ]) == 
            convert_vector_to_prediction(test.list$test_y[counter, ]),
          "Green",
          "Red"
        ),
        cex = 2
      )
    }
  }
}

# Convolutional Neural Network ------------------------------------

cnn.model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = c(img_width, img_length, img_channels)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')

# Compile model
cnn.model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Train model
cnn.model %>% fit(
  train.list$train_x %>% array_reshape(dim = c(60000, 28, 28, 1)), 
  train.list$train_y,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
)

# Scores
cnn.model %>% evaluate(
  test.list$test_x %>% array_reshape(dim = c(10000, 28, 28, 1)),
  test.list$test_y,
  verbose = 0
)

# Probabilities
cnn_model.pred <- cnn.model %>% 
  predict_proba(test.list$test_x %>% array_reshape(dim = c(10000, 28, 28, 1)))

# Plot
cnn_model.pred %>% {
  by <- 15
  EBImage::combine(
    map(
      1:by^2,
      function(x) {
        test.list$test_x[x,,] %>% 
          Image(colormode = "GrayScale")
      }
    )
  ) %>% 
    display(method = "raster", all = TRUE)
  row_indeces <- seq(0, img_width*(by-1), img_width)
  column_indeces <- seq(0, img_length*(by-1), img_length)
  counter <- 0
  for (i in row_indeces) {
    for (j in column_indeces) {
      counter <- counter + 1
      text(
        x = j + 1, y = i + 1,
        label = convert_vector_to_prediction(.[counter, ]),
        adj = c(0, 1), 
        col = ifelse(
          convert_vector_to_prediction(.[counter, ]) == 
            convert_vector_to_prediction(test.list$test_y[counter, ]),
          "Green",
          "Red"
        ),
        cex = 2
      )
    }
  }
}

# Which did not match?
which(
  sapply(
    1:nrow(cnn_model.pred),
    function(x) {
      convert_vector_to_prediction(cnn_model.pred[x, ]) !=
        convert_vector_to_prediction(test.list$test_y[x, ])
    }
  )
)

4537 %>% {
  test.list$test_x[.,,] %>% 
    Image(colormode = "GrayScale") %>% 
    display(method = "raster", all = TRUE)
  text(
    x = 1, y = 1,
    label = convert_vector_to_prediction(cnn_model.pred[., ]),
    adj = c(0, 1), cex = 2,
    col = ifelse(
      convert_vector_to_prediction(cnn_model.pred[., ]) == 
        convert_vector_to_prediction(test.list$test_y[., ]),
      "Green", "Red"
    )
  )
}




# Data Augmentation -------------------------------------------------------

img_datagen <- keras::image_data_generator(
  rotation_range = 15,
  shear_range = 0.5
)

# Data Generator
nmist_img_datagen <- flow_images_from_data(
  train.list$train_x %>% 
    array_reshape(dim = c(60000, img_width, img_length, 1)),
  train.list$train_y,
  generator = img_datagen,
  batch_size = batch_size
)

nmist_img_datagen_test <- flow_images_from_data(
  test.list$test_x %>% 
    array_reshape(dim = c(10000, img_width, img_length, 1)),
  test.list$test_y,
  generator = img_datagen,
  batch_size = batch_size
)

nmist_img_datagen %>% 
  generator_next() %>% {
    {.[[1]][1,,,]} %>% 
      Image(colormode = "Grayscale") %>% 
      display(method = "raster")
    text(
      x = 1, y = 1,
      label = convert_vector_to_prediction(.[[2]][1,]),
      adj = c(0, 1), cex = 2,
      col = "Grey"
    )
  }



# Augmented Traditional Neural Network ----------------------------------------------

ann.model <- keras_model_sequential() %>%
  layer_flatten() %>% 
  layer_dense(units = img_width * img_length * img_channels, 
              activation = "relu") %>% 
  layer_dense(units = img_width * img_length * img_channels / 2,
              activation = "relu") %>% 
  layer_dense(units = img_width * img_length * img_channels / 4,
              activation = "relu") %>% 
  layer_dense(units = num_classes, activation = "softmax")

# Compile model
ann.model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Train model
ann.model %>% fit_generator(
  generator = nmist_img_datagen,
  epochs = epochs,
  steps_per_epoch = 30
)

# Scores
ann.model %>% evaluate_generator(
  nmist_img_datagen_test,
  steps = 50
)

# Probabilities
ann_model.pred <- ann.model %>% 
  predict_generator(
    nmist_img_datagen_test,
    steps = 50
  )

ann_model.pred %>% {
  by <- 15
  EBImage::combine(
    map(
      1:by^2,
      function(x) {
        test.list$test_x[x,,] %>% 
          Image(colormode = "GrayScale")
      }
    )
  ) %>% 
    display(method = "raster", all = TRUE)
  row_indeces <- seq(0, img_width*(by-1), img_width)
  column_indeces <- seq(0, img_length*(by-1), img_length)
  counter <- 0
  for (i in row_indeces) {
    for (j in column_indeces) {
      counter <- counter + 1
      text(
        x = j + 1, y = i + 1,
        label = convert_vector_to_prediction(.[counter, ]),
        adj = c(0, 1), 
        col = ifelse(
          convert_vector_to_prediction(.[counter, ]) == 
            convert_vector_to_prediction(test.list$test_y[counter, ]),
          "Green",
          "Red"
        ),
        cex = 2
      )
    }
  }
}

