library(stat299capstone)
library(glue)
library(magick)
library(stringr)

folder_path <- "data/documents/"
path_folder <- "data/documents_processed/"

# Data Processing ---------------------------------------------------------

# 8607

buyer_folder <- "b8607"

for (i in c(28964, 28982, 28983)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la16125_a{i}.pdf"),
    retained_pages = 1, path_folder = path_folder
  )
}

for (i in c(12914, 14611, 14633:14635)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la22579_a{i}.pdf"),
    retained_pages = 1, path_folder = path_folder
  )
}

# Skipped
# 14607 -- looks like a receipt lang

for (i in c(26011:26013)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la23259_a{i}.pdf"),
    retained_pages = 1, path_folder = path_folder
  )
}

for (i in c(967:968)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la8378_a{i}.pdf"),
    retained_pages = 1, path_folder = path_folder
  )
}

files <- list.files("data/documents/b8607")[
  str_detect(list.files("data/documents/b8607", full.names = TRUE), "la23259")
  ]

for (i in files) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{i}"),
    retained_pages = c(1), path_folder = path_folder
  )
}
