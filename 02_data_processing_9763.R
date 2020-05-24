library(stat299capstone)
library(glue)
library(magick)
library(stringr)

folder_path <- "data/documents/"
path_folder <- "data/documents_processed/"

# Data Processing ---------------------------------------------------------

# 9763

buyer_folder <- "b9763"

for (i in c(31025, 31024, 31023, 30929:30947, 24051, 24049:24047, 24045:24023,
            23704:23699, 22996:22981, 22956)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/b9763_la7767_a{i}.pdf"),
    retained_pages = c(1), path_folder = path_folder, degrees = 270
  )
}

# Skipped
# a30948

for (i in c(30794, 30793, 30790, 30788, 30786:30784)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/b9763_la61699_a{i}.pdf"),
    retained_pages = 1:2, path_folder = path_folder, degrees = 270
  )
}

for (i in c(30792, 30791, 30789, 30787, 30783:30780)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/b9763_la61699_a{i}.pdf"),
    retained_pages = 1, path_folder = path_folder, degrees = 270
  )
}

for (i in c(29221, 29219, 29218, 29216:29208, 29206:29200)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/b9763_la56734_a{i}.pdf"),
    retained_pages = 1, path_folder = path_folder, degrees = 270,
    all_pages = TRUE
  )
}

# Skipped
# a27738, 27737

for (i in c(28162:28157)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/b9763_la43204_a{i}.pdf"),
    retained_pages = 1, path_folder = path_folder, degrees = 270,
    all_pages = TRUE
  )
}

for (i in c(6048:6041, 5950)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/b9763_la15736_a{i}.pdf"),
    retained_pages = 1, path_folder = path_folder, degrees = 270,
    all_pages = TRUE
  )
}
