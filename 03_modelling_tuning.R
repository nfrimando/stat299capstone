# Hyperparameter Tuning

suppressPackageStartupMessages({
  library(tidyverse)
  library(keras)
})

data.list <- readRDS("data/modelling_data/data.rds")

# Need to run 03_modelling.R until line for join generators

left_input_tensor      <- layer_input(shape = list(shape_size_width, shape_size_length, 1), name = "left_input_tensor")
right_input_tensor     <- layer_input(shape = list(shape_size_width, shape_size_length, 1), name = "right_input_tensor")

# Model 2 -------------------------------------------------------------------

# Parameters
conv_first_num_filters <- 4
conv_2nd_num_filters <- 8
conv_3rd_num_filters <- 8
dense_1st_nodes <- 32
dense_2nd_nodes <- 64

# Build

conv_base <- keras_model_sequential() %>%
  layer_conv_2d(filters = conv_first_num_filters, kernel_size = c(3, 3), activation = "relu", input_shape = list(shape_size_width, shape_size_length, 1), stride = 1) %>%
  layer_max_pooling_2d(pool_size = c(2, 2), stride = 2) %>%
  layer_conv_2d(filters = conv_2nd_num_filters, kernel_size = c(2, 2), activation = "relu", stride = 1) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = conv_3rd_num_filters, kernel_size = c(2, 2), activation = "relu", stride = 1) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = dense_1st_nodes, activation = "relu", name = 'fc1')  %>%
  layer_dropout(rate = 0.1, name = 'dropout1')                 %>%
  layer_dense(units = dense_2nd_nodes, activation = "sigmoid", name = 'fc2')

left_output_tensor     <- left_input_tensor  %>% conv_base
right_output_tensor    <- right_input_tensor %>%  conv_base

L1_distance <- function(tensors) { # build keras backend's function
  c(x,y) %<-% tensors
  return( k_abs( x - y ) )
}

L1_layer    <- layer_lambda(
  object = list(left_output_tensor,right_output_tensor) , # To build self define layer, you must use layer_lamda
  f = L1_distance
)

prediction  <- L1_layer%>%
  layer_dense( units = 1 , activation = "sigmoid" )

model2       <- keras_model( list(left_input_tensor,right_input_tensor), prediction)

# Model Fit

model2 %>% compile(
  loss      = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0005),
  metrics   = c("accuracy")
)

start_time <- Sys.time()
history2 <- model2 %>% fit_generator(
  generator = train_join_generator,
  steps_per_epoch = 25,
  epochs = 30,
  validation_data = val_join_generator,
  validation_steps = 25
)
end_time <- Sys.time()
model2_time <- end_time - start_time

model2 %>% evaluate_generator(test_join_generator, steps = 1000)

# Model 3 -----------------------------------------------------------------

# Parameters
conv_first_num_filters <- 2
conv_2nd_num_filters <- 4
conv_3rd_num_filters <- 4
dense_1st_nodes <- 16
dense_2nd_nodes <- 32

# Build

conv_base <- keras_model_sequential() %>%
  layer_conv_2d(filters = conv_first_num_filters, kernel_size = c(3, 3), activation = "relu", input_shape = list(shape_size_width, shape_size_length, 1), stride = 1) %>%
  layer_max_pooling_2d(pool_size = c(2, 2), stride = 2) %>%
  layer_conv_2d(filters = conv_2nd_num_filters, kernel_size = c(2, 2), activation = "relu", stride = 1) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = conv_3rd_num_filters, kernel_size = c(2, 2), activation = "relu", stride = 1) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = dense_1st_nodes, activation = "relu", name = 'fc1')  %>%
  layer_dropout(rate = 0.1, name = 'dropout1')                 %>%
  layer_dense(units = dense_2nd_nodes, activation = "sigmoid", name = 'fc2')

left_output_tensor     <- left_input_tensor  %>% conv_base
right_output_tensor    <- right_input_tensor %>%  conv_base

L1_distance <- function(tensors) { # build keras backend's function
  c(x,y) %<-% tensors
  return( k_abs( x - y ) )
}

L1_layer    <- layer_lambda(
  object = list(left_output_tensor,right_output_tensor) , # To build self define layer, you must use layer_lamda
  f = L1_distance
)

prediction  <- L1_layer%>%
  layer_dense( units = 1 , activation = "sigmoid" )

model3       <- keras_model( list(left_input_tensor,right_input_tensor), prediction)

# Model Fit

model3 %>% compile(
  loss      = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0005),
  metrics   = c("accuracy")
)

start_time <- Sys.time()
history3 <- model3 %>% fit_generator(
  generator = train_join_generator,
  steps_per_epoch = 25,
  epochs = 30,
  validation_data = val_join_generator,
  validation_steps = 25
)
end_time <- Sys.time()
model3_time <- end_time - start_time

model3 %>% evaluate_generator(test_join_generator, steps = 1000)
