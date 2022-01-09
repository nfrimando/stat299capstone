# Practice
# https://www.kaggle.com/shih0430/mnist-siamese-neural-network-for-keras-r-code


# Libraries ---------------------------------------------------------------

library(keras)
library(glue)
library(tidyverse)
library(EBImage)

shape_size_length  <- 300
shape_size_width   <- shape_size_length * 0.7

set.seed(9478)

# Preparing Data Set ------------------------------------------------------

documents <- readRDS(glue("data/data_processed.rds"))

# Limit Data set to having equal representation
smallest_sample_size <- data.frame(labels = documents$labels) %>%
  group_by(labels) %>% summarise(count = n()) %>% ungroup() %>%
  {.$count} %>%
  min()

max_sample_size <- 1000

sample_indeces <- map(
  unique(documents$labels),
  function(x) {
    which(documents$labels == x) %>%
      sample(min(max_sample_size, sum(documents$labels == x)))
  }
) %>% unlist

documents_samples <- list(
  images = documents$images[sample_indeces,,],
  labels = documents$labels[sample_indeces]
)

# Split to train, validation, and test
# 64-16-20

# train_indeces <- sample(1:length(documents_samples$labels), ceiling(length(documents_samples$labels) * 0.8))
# test_indeces <- (1:length(documents_samples$labels))[-train_indeces]
# val_indeces <- sample(train_indeces, length(train_indeces) * 0.2)
# train_indeces <- train_indeces[!(train_indeces %in% val_indeces)]
labels <- factor(documents_samples$labels, levels = sort(unique(as.numeric(documents_samples$labels)))) # Turn lables to numeric

b9592_test_indeces <- sample(which(documents_samples$labels == "9592"), 27)
b9763_test_indeces <- sample(which(documents_samples$labels == "9763"), 25)
b9211_test_indeces <- sample(which(documents_samples$labels == "9211"), 19)
b2929_test_indeces <- sample(which(documents_samples$labels == "2929"), 16)
b8607_test_indeces <- sample(which(documents_samples$labels == "8607"), 10)
b12754_test_indeces <- sample(which(documents_samples$labels == "12754"), 9)
test_indeces <- c(b9592_test_indeces, b9763_test_indeces, b9211_test_indeces,
                  b2929_test_indeces, b8607_test_indeces, b12754_test_indeces)

train_indeces <- (1:length(documents_samples$labels))[-test_indeces]
val_indeces <- sample(train_indeces, length(train_indeces) * 0.2)
train_indeces <- train_indeces[!(train_indeces %in% val_indeces)]

# Train
train_images <- 1 - documents_samples$images[train_indeces, ,]
train_labels <- as.numeric(labels[train_indeces]) - 1

# Validation
val_images <- 1 - documents_samples$images[val_indeces, ,]
val_labels <- as.numeric(labels[val_indeces]) - 1

# Test
test_images <- 1 - documents_samples$images[test_indeces, ,]
test_labels <- as.numeric(labels[test_indeces]) - 1

# Check out an image
train_images[32,,] %>%
  Image(colormode = "Grayscale") %>%
  display(method = "raster")

train_labels[32]


# Parameters --------------------------------------------------------------

num_classes  <- 6
train_batch  <- 20
val_batch    <- 10
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

test_data_list      <- list()
grp_kind     <- sort( unique( test_labels ) )
for( grp_idx in 1:length(grp_kind) ) { # grp_idx = 1
  label                      <- grp_kind[grp_idx]
  tmp_images                 <- test_images[test_labels==label,,]
  tmp_images                 <- array( tmp_images , dim = c( dim(tmp_images) , 1) )
  test_data_list[[grp_idx]]   <- list( data  = tmp_images ,
                                      label = test_labels[test_labels==label]
  )
}

train_data_list[[3]]$data[9,,,] %>%
  Image(colormode = "Grayscale") %>%
  display(method = "raster")

# Save for Usage

saveRDS(
  list(
    train_data = train_data_list,
    test_data = test_data_list,
    val_data = val_data_list
  ),
  "data/modelling_data/data.rds"
)

# Data Augmentation -------------------------------------------------------

train_datagen = image_data_generator(
  rescale = 1          ,
  rotation_range = 10       ,
  width_shift_range = 0.01  ,
  height_shift_range = 0.05,
  shear_range = 0.1,
  zoom_range = 0.1         ,
  horizontal_flip = FALSE  ,
  vertical_flip = FALSE    ,
  fill_mode = "constant",
  cval = 0
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

test_augmentation_generator <- function(label_index) {
  flow_images_from_data(
    x = test_data_list[[label_index]]$data  ,
    y = test_data_list[[label_index]]$label ,
    train_datagen                  ,
    shuffle = TRUE                 ,
    seed = 9487                    ,
    batch_size = 1
  )
}


# 0 to 5 label but index is 1 to 6
train_augmentation_generator.list <- map(
  grp_kind,
  function(label_index) {train_augmentation_generator(label_index + 1)}
)

val_augmentation_generator.list <- map(
  grp_kind,
  function(label_index) {val_augmentation_generator(label_index + 1)}
)

test_augmentation_generator.list <- map(
  grp_kind,
  function(label_index) {test_augmentation_generator(label_index + 1)}
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
test_join_generator     <- join_generator( test_augmentation_generator.list   , test_batch   )

train_join_generator() %>% {
  sample <- .
  sample[[1]][[1]][11,,,] %>%
    abind(sample[[1]][[2]][11,,,], along = 1) %>%
    Image(colormode = "Grayscale") %>%
    display(method = "raster")
}

saveRDS(
  list(
    train_join_generator = train_join_generator,
    val_join_generator = val_join_generator,
    test_join_generator = test_join_generator
  ),
  "data/modelling_data/join_generators.rds"
)

# building the model ------------------------------------------------------

left_input_tensor      <- layer_input(shape = list(shape_size_width, shape_size_length, 1), name = "left_input_tensor")
right_input_tensor     <- layer_input(shape = list(shape_size_width, shape_size_length, 1), name = "right_input_tensor")

# conv_base              <- keras_model_sequential()           %>%
#   layer_flatten(input_shape = list(shape_size_width, shape_size_length, 1)) %>%
#   layer_dense(units = 512, activation = "relu", name='fc1')  %>%
#   layer_dropout(rate = 0.1, name='dropout1')                 %>%
#   layer_dense(units = 256, activation = "relu", name='fc2')  %>%
#   layer_dropout(rate = 0.1, name='dropout2')                 %>%
#   layer_dense(units = 128, activation = "relu", name='fc3')

conv_base <- keras_model_sequential() %>%
  layer_conv_2d(filters = 8, kernel_size = c(3, 3),
                activation = "relu",
                input_shape = list(shape_size_width, shape_size_length, 1),
                stride = 1) %>%
  layer_max_pooling_2d(pool_size = c(2, 2), stride = 2) %>%
  layer_conv_2d(filters = 16, kernel_size = c(2, 2),
                activation = "relu", stride = 1) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 16, kernel_size = c(2, 2),
                activation = "relu", stride = 1) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu", name = 'fc1')  %>%
  layer_dropout(rate = 0.1, name = 'dropout1')                 %>%
  layer_dense(units = 128, activation = "sigmoid", name = 'fc2')

left_output_tensor     <- left_input_tensor  %>% conv_base

right_output_tensor    <- right_input_tensor %>%  conv_base

L1_distance <- function(tensors) {
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
  optimizer = optimizer_rmsprop(lr = 0.0005),
  metrics   = c("accuracy")
)

start_time <- Sys.time()
history <- model %>% fit_generator(
  generator = train_join_generator,
  steps_per_epoch = 25,
  epochs = 30,
  validation_data = val_join_generator,
  validation_steps = 25
)
end_time <- Sys.time()
model1_time <- end_time - start_time

plot(history)


# Data testing ------------------------------------------------------------

model %>% evaluate_generator(test_join_generator, steps = 1000)
model %>% predict_generator(test_generator, steps=num_test_images)


# Test Model --------------------------------------------------------------

# same number
mnist_number_left  <- 2
filter_idx_left    <- sample( which( test_labels == mnist_number_left  ) , 1 )
img_input_left     <- test_images[filter_idx_left ,,]
mnist_number_right <- 3
filter_idx_right   <- sample( which( test_labels == mnist_number_right ) , 1 )
img_input_right    <- test_images[filter_idx_right,,]
img_input_left     <- array_reshape(img_input_left , c(1, shape_size_width, shape_size_length, 1))
img_input_right    <- array_reshape(img_input_right, c(1, shape_size_width, shape_size_length, 1))

similarity         <- model %>%
  predict(
    list(img_input_left, img_input_right)
  )

abind(1 - img_input_left[1,,,],
      1 - img_input_right[1,,,],
      along = 1) %>%
  Image(colormode = "Grayscale") %>%
  display(method = "raster")
title(
  paste0(
    test_labels[filter_idx_left] , " v.s " , test_labels[filter_idx_right] , " , similarity : " ,
    round(similarity,5)
  ),
  col.main = "blue"
)

# Test set vs Test set (non-augmented) ------------------------------------------

library(glue)
# Do for all items in test set
non_augmented_similarities_test_only <- map_df(
  1:length(test_labels),
  function(test_index) {
    map_df(
      1:length(test_labels),
      function(test_index_2) {
        message(glue("Comparing test index {test_index} against test index {test_index_2} out of {(test_images %>% dim())[1]}"))

        similarity <- model %>%
          predict(
            list(
              array_reshape(test_images[test_index,,] , c(1, shape_size_width, shape_size_length, 1)),
              array_reshape(test_images[test_index_2,,] , c(1, shape_size_width, shape_size_length, 1))
            )
          )

        data.frame(
          test_index = test_index,
          test_label = test_labels[test_index],
          test_index_2 = test_index_2,
          test_label_2 = test_labels[test_index_2],
          similarity = similarity
        )
      }
    )
  }
)

saveRDS(
  list(
    non_augmented_similarities_test_only
  ),
  "data/modelling_data/non_augmented_similarities_test_only.rds"
)

non_augmented_similarities_test_only %>%
  filter(test_label == 0) %>%
  filter(test_index != test_index_2) %>%
  mutate(is_predicted_similar = similarity >= 0.5) %>%
  mutate(is_correct = (is_predicted_similar & test_label == test_label_2) | (!is_predicted_similar & test_label != test_label_2)) %>%
  # group_by(test_label) %>%
  summarise(
    number_of_pairs = n(),
    number_of_test_samples = length(unique(test_index)),
    number_of_actual_similar_pairs = sum(test_label == test_label_2),
    number_of_predicted_similar_pairs = sum(is_predicted_similar),
    number_of_actual_dissimilar_pairs = sum(test_label != test_label_2),
    number_of_predicted_dissimilar_pairs = sum(!is_predicted_similar),
    number_of_correctly_predicted_pairs = sum(is_correct),
    accuracy = sum(is_correct)/n(),
    sensitivity = sum(is_predicted_similar & test_label == test_label_2)/sum(test_label == test_label_2),
    specificity = sum(!is_predicted_similar & test_label != test_label_2)/sum(test_label != test_label_2)
  ) %>%
  ungroup() %>%
  arrange(desc(number_of_test_samples)) %>% View()

# Entire Set --------------------------------------------------------------

test_labels

index <- 4
(1 - test_images[index,,]) %>% display()

# compare against everything in test set
similarities <- map(
  (1:(test_images %>% dim())[1])[-index],
  function(x){
    similarity <- model %>%
      predict(
        list(
          array_reshape(test_images[index,,] , c(1, shape_size_width, shape_size_length, 1)),
          array_reshape(test_images[x ,,] , c(1, shape_size_width, shape_size_length, 1))
        )
      )

    list(
      image = test_images[x ,,],
      similarity = similarity
    )
  }
)

# Order based on most similar
ordered_similarities <- similarities[order(map_dbl(similarities, c("similarity")), decreasing = TRUE)]

abind(1 - test_images[index,,],
      1 - ordered_similarities[[1]]$image,
      1 - ordered_similarities[[2]]$image,
      1 - ordered_similarities[[3]]$image,
      1 - ordered_similarities[[4]]$image,
      along = 1) %>%
  Image(colormode = "Grayscale") %>%
  display(method = "raster")
title(
  paste(
    "Original",
    round(ordered_similarities[[1]]$similarity,5),
    round(ordered_similarities[[2]]$similarity,5),
    round(ordered_similarities[[3]]$similarity,5),
    round(ordered_similarities[[4]]$similarity,5),
    sep = ", "
  ),
  col.main = "blue",
  line = -3
)



# Performance on non-augmented data -----------------------------

test_labels %>% length()


train_labels %>% table()
# which(test_labels == 5)
index <- 98 # for testing a single document against train set

# compare against everything in train set
similarities <- map(
  (1:(train_images %>% dim())[1]),
  function(x){
    message(glue("Image {x} out of 342"))
    similarity <- model %>%
      predict(
        list(
          array_reshape(test_images[index,,] , c(1, shape_size_width, shape_size_length, 1)),
          array_reshape(train_images[x ,,] , c(1, shape_size_width, shape_size_length, 1))
        )
      )

    list(
      image = train_images[x ,,],
      similarity = similarity
    )
  }
)

# Order based on most similar
ordered_similarities <- similarities[order(map_dbl(similarities, c("similarity")), decreasing = TRUE)]

abind(1 - test_images[index,,],
      1 - ordered_similarities[[4]]$image,
      1 - ordered_similarities[[3]]$image,
      1 - ordered_similarities[[2]]$image,
      1 - ordered_similarities[[1]]$image,
      along = 1) %>%
  Image(colormode = "Grayscale") %>%
  display(method = "raster")
title(
  paste(
    "Original",
    round(ordered_similarities[[1]]$similarity,5),
    round(ordered_similarities[[2]]$similarity,5),
    round(ordered_similarities[[3]]$similarity,5),
    round(ordered_similarities[[4]]$similarity,5),
    sep = ", "
  ),
  col.main = "blue",
  line = -3
)



# measure percent correct for one test image
map_chr(similarities, c('is_correct'))

library(glue)
# Do for all items in test set
non_augmented_similarities <- map_df(
  1:length(test_labels),
  function(test_index) {
    map_df(
      1:length(train_labels),
      function(train_index) {
        message(glue("Comparing test index {test_index} against train index {train_index} out of {(train_images %>% dim())[1]}"))

        similarity <- model %>%
          predict(
            list(
              array_reshape(test_images[test_index,,] , c(1, shape_size_width, shape_size_length, 1)),
              array_reshape(train_images[train_index ,,] , c(1, shape_size_width, shape_size_length, 1))
            )
          )

        data.frame(
          test_index = test_index,
          test_label = test_labels[test_index],
          train_index = train_index,
          train_label = train_labels[train_index],
          similarity = similarity
        )
      }
    )
  }
)

saveRDS(
  list(
    non_augmented_similarities
  ),
  "data/modelling_data/non_augmented_similarities.rds"
)

train_labels %>% table()

non_augmented_similarities %>%
  mutate(is_predicted_similar = similarity >= 0.5) %>%
  mutate(is_correct = (is_predicted_similar & test_label == train_label) | (!is_predicted_similar & test_label != train_label)) %>%
  group_by(test_label) %>%
  summarise(
    number_of_pairs = n(),
    number_of_test_samples = length(unique(test_index)),
    number_of_actual_similar_pairs = sum(test_label == train_label),
    number_of_predicted_similar_pairs = sum(is_predicted_similar),
    number_of_correctly_predicted_similar_pairs = sum(is_predicted_similar & is_correct),
    number_of_actual_dissimilar_pairs = sum(test_label != train_label),
    number_of_predicted_dissimilar_pairs = sum(!is_predicted_similar),
    number_of_correctly_predicted_dissimilar_pairs = sum(!is_predicted_similar & is_correct),
    number_of_correctly_predicted_pairs = sum(is_correct),
    accuracy = sum(is_correct)/n(),
    sensitivity = sum(is_predicted_similar & test_label == train_label)/sum(test_label == train_label),
    specificity = sum(!is_predicted_similar & test_label != train_label)/sum(test_label != train_label)
  ) %>%
  ungroup() %>%
  arrange(desc(number_of_test_samples)) %>% View()

# Threshold sensitivity
auc <- map_df(
  seq(0, 1, 0.05),
  function(cutoff) {
    non_augmented_similarities %>%
      mutate(is_predicted_similar = similarity >= cutoff) %>%
      mutate(is_correct = (is_predicted_similar & test_label == train_label) | (!is_predicted_similar & test_label != train_label)) %>%
      summarise(
        accuracy = sum(is_correct)/n(),
        sensitivity = sum(is_predicted_similar & test_label == train_label)/sum(test_label == train_label),
        specificity = sum(!is_predicted_similar & test_label != train_label)/sum(test_label != train_label)
      ) %>%
      ungroup() %>%
      mutate(threshold = cutoff)
  }
)

library(scales)
auc %>%
  gather(key = variable, value = value, -threshold) %>%
  ggplot() +
  geom_line(mapping = aes(x = threshold, y = value, color = variable)) +
  theme_bw() +
  scale_x_continuous(breaks = seq(0, 1, 0.05), labels = percent) +
  scale_y_continuous(breaks = seq(0, 1, 0.05), labels = percent) +
  labs(
    y = "Value",
    x = "Threshold",
    title = "Threshold vs Metric"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20),
    legend.text = element_text(size = 20),
    legend.title = element_text(size = 15),
    axis.title = element_text(size = 20)
  )

# new document performance ------------------------------------------------

# 2441
b441.imgs <- collect_images("data/documents_processed/b2441", width = shape_size_width, height = shape_size_length)

new_images <- b441.imgs %>%
  aperm(c(3, 1, 2))

new_images[1,,] %>%
  Image(colormode = "Grayscale") %>%
  display(method = "raster")

new_images[2,,] %>%
  Image(colormode = "Grayscale") %>%
  display(method = "raster")

# compare against everything in train set
similarities <- map(
  (1:(train_images %>% dim())[1]),
  function(x){
    message(glue("Image {x} out of 342"))
    similarity <- model %>%
      predict(
        list(
          array_reshape(1 - new_images[1,,] , c(1, shape_size_width, shape_size_length, 1)),
          array_reshape(train_images[x ,,] , c(1, shape_size_width, shape_size_length, 1))
        )
      )

    list(
      image = train_images[x ,,],
      similarity = similarity
    )
  }
)

same_similarity <- model %>%
  predict(
    list(
      array_reshape(1 - new_images[1,,] , c(1, shape_size_width, shape_size_length, 1)),
      array_reshape(1 - new_images[2,,] , c(1, shape_size_width, shape_size_length, 1))
    )
  )

# Order based on most similar
ordered_similarities <- similarities[order(map_dbl(similarities, c("similarity")), decreasing = TRUE)]

abind(new_images[1,,],
      new_images[2,,],
      1 - ordered_similarities[[1]]$image,
      1 - ordered_similarities[[2]]$image,
      1 - ordered_similarities[[3]]$image,
      along = 1) %>%
  Image(colormode = "Grayscale") %>%
  display(method = "raster")
title(
  paste(
    "Original",
    round(same_similarity,5),
    round(ordered_similarities[[1]]$similarity,5),
    round(ordered_similarities[[2]]$similarity,5),
    round(ordered_similarities[[3]]$similarity,5),
    sep = ", "
  ),
  col.main = "blue",
  line = -3
)


# test against shearing ---------------------------------------------------

sample_1.img <- train_data_list[[4]]$data[1,,,]

sample_1.img %>% {1 - .} %>%
  Image(colormode = "Grayscale") %>%
  display(method = "raster")

# sample_2.img <- sample_1.img %>% rotate(-5) %>% resize(w = shape_size_width, h = shape_size_length)
# sample_3.img <- sample_1.img %>% rotate(5) %>% resize(w = shape_size_width, h = shape_size_length)
# sample_4.img <- sample_1.img %>% rotate(10) %>% resize(w = shape_size_width, h = shape_size_length)
# sample_5.img <- sample_1.img %>% rotate(20) %>% resize(w = shape_size_width, h = shape_size_length)
sample_2.img <- sample_1.img %>% {. * 0.9}
sample_3.img <- sample_1.img %>% {. * 0.7}
sample_4.img <- sample_1.img %>% {. * 0.5}
sample_5.img <- sample_1.img %>% {. * 0.3}


# compare against all
similarities <- map(
  list(sample_2.img, sample_3.img, sample_4.img, sample_5.img),
  function(x){
    similarity <- model %>%
      predict(
        list(
          array_reshape(sample_1.img , c(1, shape_size_width, shape_size_length, 1)),
          array_reshape(x , c(1, shape_size_width, shape_size_length, 1))
        )
      )

    list(
      image = x,
      similarity = similarity
    )
  }
)


# Order based on most similar
ordered_similarities <- similarities[order(map_dbl(similarities, c("similarity")), decreasing = TRUE)]

abind(1 - sample_1.img,
      1 - sample_2.img,
      1 - sample_3.img,
      1 - sample_4.img,
      1 - sample_5.img,
      along = 1) %>%
  Image(colormode = "Grayscale") %>%
  display(method = "raster")
title(
  paste(
    "Original",
    round(ordered_similarities[[1]]$similarity,5),
    round(ordered_similarities[[2]]$similarity,5),
    round(ordered_similarities[[3]]$similarity,5),
    round(ordered_similarities[[4]]$similarity,5),
    sep = ", "
  ),
  col.main = "blue",
  line = -3
)

