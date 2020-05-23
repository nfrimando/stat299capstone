library(paws)

s3 <- paws::s3()
s3$list_buckets()

# List available buckets
s3$list_buckets()

# Check access to bucket 
s3$head_bucket(Bucket = "firstcircle") # list() return means bucket available
s3$head_bucket(Bucket = "firstcircle/production/") # Nope

# List objects in bucket
s3$list_objects(
  Bucket = "firstcircle",
  MaxKeys = "4"
)

# Get an object
s3$get_object(
  Bucket = "firstcircle",
  Key = "production/loan_applications/54617/attachments/v2/original/120860/received_1051360558560856.jpeg"
)

library(tidyverse)
library(magick)

# Save Image
x <- .Last.value
x$Body %>% image_read()
x$Body %>% 
  image_read() %>% 
  image_convert("rgba") %>% 
  image_write(path = "data/test.jpeg", format = "jpeg")

# Read as Image (EBiMage)

library(EBImage)
image.img <- readImage("data/test.jpeg") # Read Image
display(image.img)
