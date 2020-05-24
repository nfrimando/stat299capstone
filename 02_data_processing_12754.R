library(stat299capstone)
library(glue)
library(magick)
library(stringr)

folder_path <- "data/documents/"
path_folder <- "data/documents_processed/"

# Data Processing ---------------------------------------------------------

# 12754

buyer_folder <- "b12754"

files <- list.files("data/documents/b12754")[
  str_detect(list.files("data/documents/b12754", full.names = TRUE), "la17739")
  ]

for (i in files) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{i}"),
    retained_pages = c(1), path_folder = path_folder
  )
}

files <- list.files("data/documents/b12754")[
  str_detect(list.files("data/documents/b12754", full.names = TRUE), "la20319")
  ]

for (i in files) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{i}"),
    retained_pages = c(1), path_folder = path_folder
  )
}

files <- list.files("data/documents/b12754")[
  str_detect(list.files("data/documents/b12754", full.names = TRUE), "la5631")
  ]

for (i in files) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{i}"),
    retained_pages = c(1), path_folder = path_folder
  )
}

for (i in c(8396, 8395, 8394)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la5631_a{i}.pdf"),
    retained_pages = c(1), path_folder = path_folder, degrees = 90
  )
}

# skip la5707

files <- list.files("data/documents/b12754")[
  str_detect(list.files("data/documents/b12754", full.names = TRUE), "la20319")
  ]

for (i in files) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{i}"),
    retained_pages = c(1), path_folder = path_folder
  )
}
