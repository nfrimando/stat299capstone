library(stat299capstone)
library(glue)
library(magick)
library(stringr)

folder_path <- "data/documents/"
path_folder <- "data/documents_processed/"

# Data Processing ---------------------------------------------------------

# 2441

buyer_folder <- "b2441"
#
# files <- list.files("data/documents/b2441")[
#   str_detect(list.files("data/documents/b2441", full.names = TRUE), "la28032")
#   ]
#
# for (i in files) {
#   magick_split_pdf(
#     file_name = glue::glue(folder_path, "{buyer_folder}/{i}"),
#     retained_pages = c(1), path_folder = path_folder, degrees = 270
#   )
# }

files <- list.files("data/documents/b2441")[
  str_detect(list.files("data/documents/b2441", full.names = TRUE), "la8209")
  ]

for (i in files) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{i}"),
    retained_pages = c(1), path_folder = path_folder
  )
}

