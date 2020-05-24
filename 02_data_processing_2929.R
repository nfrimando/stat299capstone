library(stat299capstone)
library(glue)
library(magick)
library(stringr)

folder_path <- "data/documents/"
path_folder <- "data/documents_processed/"

# Data Processing ---------------------------------------------------------

# 2929

buyer_folder <- "b2929"

magick_split_pdf(
  file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la10510_a15015.pdf"),
  retained_pages = c(1,2) , path_folder = path_folder
)

magick_split_pdf(
  file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la10510_a15538.pdf"),
  retained_pages = c(1), path_folder = path_folder, all_pages = TRUE
)

magick_split_pdf(
  file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la10510_a16004.pdf"),
  retained_pages = c(1), path_folder = path_folder, all_pages = TRUE
)

# Remove invoice pages
magick_split_pdf(
  file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la285_a26439.pdf"),
  retained_pages = c(1,2), path_folder = path_folder
)

magick_split_pdf(
  file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la285_a9312.pdf"),
  retained_pages = c(2), path_folder = path_folder
)

# la8923

for (i in c(9442, 6787, 6504, 6160, 5604, 3610, 13816, 13548, 12634, 12544)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la8923_a{i}.pdf"),
    retained_pages = c(3), path_folder = path_folder
  )
}

for (i in c(4845)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la8923_a{i}.pdf"),
    retained_pages = c(3, 4), path_folder = path_folder
  )
}

for (i in c(3928, 21336, 21252, 21251, 2063, 19734, 14005)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la8923_a{i}.pdf"),
    retained_pages = c(1), path_folder = path_folder, all_pages = TRUE
  )
}

# Skipped -- letters
# 19733, 19732, 15023, 15022, 14007, 14006

# la8347

for (i in c(9212:9208, 7475, 5019, 5016, 3283, 2248, 21140, 19584:19582, 1913, 1912,
            1256, 1218, 11666, 11655, 10427, 10316:10313)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la8347_a{i}.pdf"),
    retained_pages = c(1), path_folder = path_folder, all_pages = TRUE, degrees = 270
  )
}

for (i in c(23044, 22905)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la8347_a{i}.pdf"),
    retained_pages = c(1), path_folder = path_folder, all_pages = TRUE
  )
}

magick_split_pdf(
  file_name = glue::glue(folder_path, "{buyer_folder}/{buyer_folder}_la8347_a10428.pdf"),
  retained_pages = c(1), path_folder = path_folder, all_pages = TRUE, degrees = 90
)



