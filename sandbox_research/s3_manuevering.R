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



# Trying to create a function to pull -------------------------------------

# https://s3.console.aws.amazon.com/s3/buckets/firstcircle/production/loan_applications/950/purchase_order_attachments/12227/?region=ap-southeast-1&tab=overview

library(glue)

loan_application_id <- 950
purchase_order_id <- 12227

contents <- s3$list_objects(
  Bucket = "firstcircle",
  Prefix = glue(
    "production/loan_applications/{loan_application_id}/purchase_order_attachments/{purchase_order_id}/"
  )
)

length(contents$Contents) == 1

key <- contents$Contents[[1]]$Key

object <- s3$get_object(
  Bucket = "firstcircle",
  Key = key
)

object$Body %>% image_read() %>% image_write(path = "data/test.pdf", format = "pdf")
