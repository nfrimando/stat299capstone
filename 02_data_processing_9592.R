library(stat299capstone)
library(glue)
library(magick)
library(stringr)

folder_path <- "data/documents/"
path_folder <- "data/documents_processed/"

# Data Processing ---------------------------------------------------------

# 9592

buyer_folder <- "b9592"

files <- list.files("data/documents/b9592")[
  str_detect(list.files("data/documents/b9592", full.names = TRUE), "la5443")
]

for (i in files) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{i}"),
    retained_pages = c(1), path_folder = path_folder, all_pages = TRUE
  )
}

for (i in c(18975:18973)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la35107_a{i}.pdf"),
    retained_pages = 1:2, path_folder = path_folder
  )
}

for (i in c(18906, 18903, 18902, 18901)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la35107_a{i}.pdf"),
    retained_pages = 1, path_folder = path_folder, all_pages = TRUE
  )
}

for (i in c(18900:18898)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la35107_a{i}.pdf"),
    retained_pages = 1:2, path_folder = path_folder
  )
}

for (i in c(18897:18892)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la35107_a{i}.pdf"),
    retained_pages = 1, path_folder = path_folder, all_pages = TRUE
  )
}

for (i in c(18891:18885)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la35107_a{i}.pdf"),
    retained_pages = 1:2, path_folder = path_folder
  )
}

magick_split_pdf(
  file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la19207_a9133.pdf"),
  retained_pages = 1, path_folder = path_folder, all_pages = TRUE
)

files <- list.files("data/documents/b9592")[
  str_detect(list.files("data/documents/b9592", full.names = TRUE), "la16125")
  ]

for (i in files) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{i}"),
    retained_pages = c(1), path_folder = path_folder, all_pages = TRUE
  )
}
