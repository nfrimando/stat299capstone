library(stat299capstone)
library(glue)

folder_path <- "data/documents/"
path_folder <- "data/documents_processed/"

# Data Processing ---------------------------------------------------------

# b9211

magick_split_pdf(
  file_name = glue::glue(folder_path, "b9211_la9132_a17927.pdf"),
  retained_pages = c(1, 3, 5, 7, 9, 11), path_folder = path_folder
)

for (i in c(2748, 2746)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "b9211_la5589_a{i}.pdf"),
    retained_pages = c(1), path_folder = path_folder
  )
}

for (i in c(9738, 9737, 9736, 9735, 9734, 9733, 9732, 9731, 9730, 9729,
            9728, 9725, 9723, 9722)) {
  magick_split_pdf(
    file_name = glue::glue(folder_path, "b9211_la20247_a{i}.pdf"),
    retained_pages = c(1), path_folder = path_folder
  )
}

# Skipped
# la12848_a7085

magick_split_pdf(
  file_name = glue::glue(folder_path, "b9211_la12848_a7084.pdf"),
  retained_pages = 1:13, path_folder = path_folder, degrees = 90
)

magick_split_pdf(
  file_name = glue::glue(folder_path, "b9211_la12848_a7082.pdf"),
  retained_pages = 1:25, path_folder = path_folder, degrees = 90
)

magick_split_pdf(
  file_name = glue::glue(folder_path, "b9211_la12848_a6730.pdf"),
  retained_pages = 1:3, path_folder = path_folder, degrees = 90
)

magick_split_pdf(
  file_name = glue::glue(folder_path, "b9211_la12516_a3913.pdf"),
  retained_pages = c(1, 3, 5, 7, 9, 11), path_folder = path_folder, degrees = 0
)

magick_split_pdf(
  file_name = glue::glue(folder_path, "b9211_la12516_a3910.pdf"),
  retained_pages = 1:3, path_folder = path_folder, degrees = 0
)
