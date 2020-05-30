
# Read JPEGS ---------------------------------------------------------------

library(stat299capstone)
library(tidyverse)

# 12754 -- 32 images -- SM
b12754.imgs <- collect_images("data/documents_processed/b12754", width = 350, height = 500)

# 2929 -- 58 images -- National Energy
# Remove b2929_la285_a9312_page2.png
b2929.imgs <- collect_images("data/documents_processed/b2929", width = 350, height = 500)

# 8607 -- 55 images -- JLL
b8607.imgs <- collect_images("data/documents_processed/b8607", width = 350, height = 500)

# 9211 -- 98 images - MDC
b9211.imgs <- collect_images("data/documents_processed/b9211", width = 350, height = 500)

# 9592 -- 151 images - Makati Med
b9592.imgs <- collect_images("data/documents_processed/b9592", width = 350, height = 500)

# 9763 -- 139 images -- Metro Gaisano
b9763.imgs <- collect_images("data/documents_processed/b9763", width = 350, height = 500)


# Compiling ---------------------------------------------------------------

compiled.imgs <- b12754.imgs %>%
  abind(b2929.imgs, along = 3) %>%
  abind(b8607.imgs, along = 3) %>%
  abind(b9211.imgs, along = 3) %>%
  abind(b9592.imgs, along = 3) %>%
  abind(b9763.imgs, along = 3) %>%
  aperm(c(3, 1, 2))

labels <- c(
  rep("12754", dim(b12754.imgs)[3]),
  rep("2929", dim(b2929.imgs)[3]),
  rep("8607", dim(b8607.imgs)[3]),
  rep("9211", dim(b9211.imgs)[3]),
  rep("9592", dim(b9592.imgs)[3]),
  rep("9763", dim(b9763.imgs)[3])
)

# Save RDS ----------------------------------------------------------------

saveRDS(
  list(
    images = compiled.imgs,
    labels = labels
  ),
  "data/data_processed.rds"
)

