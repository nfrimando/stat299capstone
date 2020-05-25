library(EBImage)


# Sample From Internet ---------------------------------------------------

## get sample pixel intensity RGB array
x <- readImage(system.file("images", "sample-color.png", package="EBImage"))
a <- imageData(x)

## convert back to Image
img <- Image(a, colormode=Color)

## combine and display the result
img2 <- combine(img, img)

display(img2, method="raster", all=TRUE)

# Try with mnist ----------------------------------------------------------

mnist$train$x[6,,] %>%
  Image(colormode = "grayscale") %>%
  # Image(colormode = Color) %>%
  display(method = "raster", all = TRUE)


# Try with Cats and Dogs --------------------------------------------------

# extract_feature <- function(dir_path, width, height) {
#   img_size <- width * height
#   images <- list.files(dir_path)
#   label <- ifelse(grepl("dog", images) == T, 1, 0)
#   print(paste("Processing", length(images), "images"))
#   feature_list <- pblapply(images, function(imgname) {
#     img <- readImage(file.path(dir_path, imgname))
#     img_resized <- EBImage::resize(img, w = width, h = height)
#     img_matrix <- matrix(reticulate::array_reshape(img_resized, (width *
#                                                                    height * channels)), nrow = width * height * channels)
#     img_vector <- as.vector(t(img_matrix))
#     return(img_vector)
#   })
#   feature_matrix <- do.call(rbind, feature_list)
#   return(list(t(feature_matrix), label))
# }

file_names <- list.files("data/cats_and_dogs/train", full.names = TRUE)

image.img <- readImage(file_names[3]) # Read Image
display(image.img) # Show Image
image_resized.img <- resize(image, w = 64, h = 64)
display(image_resized.img)

image_resized.mat <- array_reshape(
  image_resized.img,
  dim = c(64, 64, 3)
)

image_resized.mat %>%
  Image(colormode = Color) %>%
  display(method = "raster", all = TRUE)


# build array of images ---------------------------------------------------

image.img <- readImage(file_names[3])
image_resized.img <- resize(image.img, w = 64, h = 64)
image2.img <- readImage(file_names[4])
image2_resized.img <- resize(image2.img, w = 64, h = 64)

image_resized.mat <- array_reshape(image_resized.img, dim = c(64, 64, 3))
image_resized.mat2 <- array_reshape(image2_resized.img, dim = c(64, 64, 3))

# image_resized.mat %>% Image(colormode = Color) %>% display(all = TRUE)

input_data <- abind(
  list(
    image_resized.mat,
    image_resized.mat2
  ),
  along = 4
) %>%
  aperm(c(4,1,2,3))

# input_data[1,,,] %>% Image(colormode = Color) %>% display(all = TRUE)
