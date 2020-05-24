library(stat299capstone)
library(glue)

folder_path <- "data/documents/"
path_folder <- "data/documents_processed/"

# Data Processing ---------------------------------------------------------

magick_split_pdf(
  file_name = glue::glue(
    folder_path, "b9211_la9132_a17927.pdf"
  ),
  retained_pages = c(1, 3, 5, 7, 9, 11),
  path_folder = path_folder
)
