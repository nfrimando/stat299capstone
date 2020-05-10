# Practice 
# https://www.kaggle.com/shih0430/mnist-siamese-neural-network-for-keras-r-code

# Preparing Data Set ------------------------------------------------------
library(keras)
library(abind)

mnist <- dataset_mnist()

train_images <- mnist$train$x
train_labels <- mnist$train$y

set.seed(9478) 

# Validation Indeces
val_idx      <- sample( 1:nrow(train_images) , size = ceiling(0.2*nrow(train_images)) , replace = F ) 
val_images   <- train_images[val_idx,,] 
val_labels   <- train_labels[val_idx]  
train_images <- train_images[-val_idx,,] 
train_labels <- train_labels[-val_idx]  
test_images  <- mnist$test$x 
test_labels  <- mnist$test$y

# Check out an image
train_images[5,,] %>% 
  aperm(c(2,1)) %>% 
  Image(colormode = "Grayscale") %>% 
  display(method = "raster")


# Parameters --------------------------------------------------------------

num_classes  <- 10 # only number : 0,1,2,3,4,5,6,7,8,9 
shape_size   <- 28 # mnist shape ( ,28,28)
train_batch  <- 20 
val_batch    <- 20 
test_batch   <- 1


# Data Preprocess ---------------------------------------------------------

train_data_list    <- list() 
grp_kind     <- sort( unique( train_labels ) )   
for( grp_idx in 1:length(grp_kind) ) { # grp_idx = 1     
  label                      <- grp_kind[grp_idx]     
  tmp_images                 <- train_images[train_labels==label,,]     
  tmp_images                 <- array( tmp_images , dim = c( dim(tmp_images) , 1) )  # why reshape array? because keras image_data_generator only accept rank = 4  
  train_data_list[[grp_idx]] <- list( data  = tmp_images ,                                         
                                      label = train_labels[train_labels==label]                                        
  )   
}  

val_data_list      <- list() 
grp_kind     <- sort( unique( val_labels ) )   
for( grp_idx in 1:length(grp_kind) ) { # grp_idx = 1     
  label                      <- grp_kind[grp_idx]     
  tmp_images                 <- val_images[val_labels==label,,]     
  tmp_images                 <- array( tmp_images , dim = c( dim(tmp_images) , 1) )     
  val_data_list[[grp_idx]]   <- list( data  = tmp_images ,                                         
                                      label = val_labels[val_labels==label]      
  )   
}

train_data_list[[1]]$data[5,,,] %>% 
  aperm(c(2,1)) %>% 
  Image(colormode = "Grayscale") %>% 
  display(method = "raster")


# Data Augmentation -------------------------------------------------------

train_datagen = image_data_generator(
  rescale = 1/255          ,
  rotation_range = 5       ,
  width_shift_range = 0.1  ,
  height_shift_range = 0.05,
  #shear_range = 0.1,
  zoom_range = 0.1         ,
  horizontal_flip = FALSE  ,
  vertical_flip = FALSE    ,
  fill_mode = "constant"
)

train_augmentation_generator <- function(label_index) {
  flow_images_from_data(
    x = train_data_list[[label_index]]$data  ,   
    y = train_data_list[[label_index]]$label ,   
    train_datagen                  ,   
    shuffle = TRUE                 ,   
    seed = 9487                    ,   
    batch_size = 1 
  )
}

val_augmentation_generator <- function(label_index) {
  flow_images_from_data(
    x = val_data_list[[label_index]]$data  ,   
    y = val_data_list[[label_index]]$label ,   
    train_datagen                  ,   
    shuffle = TRUE                 ,   
    seed = 9487                    ,   
    batch_size = 1 
  )
}


# 0 to 9 label but index is 1 to 10
train_augmentation_generator.list <- map(
  grp_kind,
  function(label_index) {train_augmentation_generator(label_index + 1)}
)

val_augmentation_generator.list <- map(
  grp_kind,
  function(label_index) {val_augmentation_generator(label_index + 1)}
)


# joining images together -------------------------------------------------

join_generator <- function( generator_list , batch ) { 
  function() { 
    batch_left  <- NULL 
    batch_right <- NULL 
    similarity  <- NULL 
    for( i in seq_len(batch) ) { # i = 1 
      # front half 
      if( i <= ceiling(batch/2) ) { # It's suggest to use balance of positive and negative data set, so I divide half is 1(same) and another is 0(differnet).
        grp_same    <- sample( seq_len(num_classes) , 1 ) 
        batch_left  <- abind( batch_left , generator_next(generator_list[[grp_same]])[[1]] , along = 1 ) 
        batch_right <- abind( batch_right , generator_next(generator_list[[grp_same]])[[1]] , along = 1 ) 
        similarity  <- c( similarity , 1 ) # 1 : from the same number
        #par(mar = c(0,0,4,0)) 
        #plot( as.raster(batch_left[21,,,]) ) 
        #title( batch_left[[2]] ) 
      } else { # after half 
        grp_diff    <- sort( sample( seq_len(num_classes) , 2 ) ) 
        batch_left  <- abind( batch_left , generator_next(generator_list[[grp_diff[1]]])[[1]] , along = 1 ) 
        batch_right <- abind( batch_right , generator_next(generator_list[[grp_diff[2]]])[[1]] , along = 1 ) 
        similarity  <- c( similarity , 0 ) # 0 : from the differnet number
      } 
    } 
    return( 
      list( 
        list(batch_left, batch_right), 
        similarity 
      ) 
    ) 
  } 
}

train_join_generator   <- join_generator( train_augmentation_generator.list , train_batch )
val_join_generator     <- join_generator( val_augmentation_generator.list   , val_batch   )

train_join_generator() %>% {
  sample <- .
  sample[[1]][[1]][1,,,] %>% aperm(c(2,1)) %>% 
    abind(sample[[1]][[2]][1,,,] %>% aperm(c(2,1)), along = 1) %>% 
    Image(colormode = "Grayscale") %>% 
    display(method = "raster")  
}



# building the model ------------------------------------------------------

left_input_tensor      <- layer_input(shape = list(shape_size, shape_size, 1), name = "left_input_tensor")
right_input_tensor     <- layer_input(shape = list(shape_size, shape_size, 1), name = "right_input_tensor")

conv_base              <- keras_model_sequential()           %>%
  layer_flatten(input_shape = list(shape_size, shape_size, 1)) %>%
  layer_dense(units = 128, activation = "relu", name='fc1')  %>%
  layer_dropout(rate = 0.1, name='dropout1')                 %>%
  layer_dense(units = 128, activation = "relu", name='fc2')  %>% 
  layer_dropout(rate = 0.1, name='dropout2')                 %>%
  layer_dense(units = 128, activation = "relu", name='fc3')

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

model       <- keras_model( list(left_input_tensor,right_input_tensor), prediction)


# Model Fit ---------------------------------------------------------------


model %>% compile(
  loss      = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-3),
  metrics   = c("accuracy")
)

history <- model %>% fit_generator(
  generator = train_join_generator,
  steps_per_epoch = 100,
  epochs = 50,
  validation_data = val_join_generator,
  validation_steps = 50
)

plot(history)


# Test Model --------------------------------------------------------------

# same number
mnist_number_left  <- 9
filter_idx_left    <- sample( which( test_labels == mnist_number_left  ) , 1 )
img_input_left     <- test_images[filter_idx_left ,,]/255
mnist_number_right <- 0
filter_idx_right   <- sample( which( test_labels == mnist_number_right ) , 1 )
img_input_right    <- test_images[filter_idx_right,,]/255
img_input_left     <- array_reshape(img_input_left , c(1, shape_size, shape_size, 1))
img_input_right    <- array_reshape(img_input_right, c(1, shape_size, shape_size, 1))

similarity         <- model %>% 
  predict(
    list(img_input_left,img_input_right)
  )

abind(img_input_left[1,,,] %>% aperm(c(2,1)),
      img_input_right[1,,,] %>% aperm(c(2,1)),
      along = 1) %>% 
  Image(colormode = "Grayscale") %>% 
  display(method = "raster")  
title( 
  paste0( 
    test_labels[filter_idx_left] , " v.s " , test_labels[filter_idx_right] , " , similarity : " , 
    round(similarity,3) 
  ) 
)  


# Entire Set --------------------------------------------------------------

set.seed(1)

# Similar Numbers
samples <- 1000
test_similar <- map(
  1:samples,
  function(sample_num) {
    number <- sample(0:9, 1)
    mnist_number_left <- number
    mnist_number_right <- number
    filter_idx_left    <- sample( which( test_labels == mnist_number_left  ) , 1 )
    img_input_left     <- test_images[filter_idx_left ,,]/255
    filter_idx_right   <- sample( which( test_labels == mnist_number_right ) , 1 )
    img_input_right    <- test_images[filter_idx_right,,]/255
    img_input_left     <- array_reshape(img_input_left , c(1, shape_size, shape_size, 1))
    img_input_right    <- array_reshape(img_input_right, c(1, shape_size, shape_size, 1))
    similarity         <- model %>% 
      predict(
        list(img_input_left,img_input_right)
      )
    return(
      list(
        number = number,
        images = list(img_input_left, img_input_right),
        prediction = similarity
      )
    )
  }
)

# Different Numbers
test_different <- map(
  1:samples,
  function(sample_num) {
    number_left <- sample(0:9, 1)
    number_right <- sample((0:9)[which(0:9 != number_left)], 1)
    mnist_number_left <- number_left
    mnist_number_right <- number_right
    filter_idx_left    <- sample( which( test_labels == mnist_number_left  ) , 1 )
    img_input_left     <- test_images[filter_idx_left ,,]/255
    filter_idx_right   <- sample( which( test_labels == mnist_number_right ) , 1 )
    img_input_right    <- test_images[filter_idx_right,,]/255
    img_input_left     <- array_reshape(img_input_left , c(1, shape_size, shape_size, 1))
    img_input_right    <- array_reshape(img_input_right, c(1, shape_size, shape_size, 1))
    similarity         <- model %>% 
      predict(
        list(img_input_left,img_input_right)
      )
    return(
      list(
        numbers = list(number_left, number_right),
        images = list(img_input_left, img_input_right),
        prediction = similarity
      )
    )
  }
)

