#' Split PDF into separate pages
#'
#' @param file_name path to pdf
#' @param retained_pages pages in the pdf to capture and retain
#' @param path_folder folder to dump results
#' @param degrees rotation degrees (defaults to 0)
#' @param all_pages if TRUE, processes all pages and ignores retained_pages
#'
#' @import magick pdftools
#' @importFrom stringr str_replace
#' @importFrom assertthat assert_that
#' @importFrom glue glue
#'
#' @export
#'
#' @examples
#' magick_split_pdf(file_name = "data/documents/b9211_la9132_a17927.pdf", retained_pages = c(1, 3, 5, 7, 9, 11), path_folder = "data/documents_processed")
magick_split_pdf <- function(file_name, retained_pages, path_folder, degrees = 0, all_pages = FALSE) {

  if (all_pages == TRUE) {
    retained_pages <- length(image_read_pdf(file_name))
  }

  for (i in retained_pages) {

    x <- image_read_pdf(file_name, pages = i)

    new_file_name <- str_replace(
      str_replace(file_name,  ".pdf", ""),
      "documents",
      "documents_processed"
    )

    x %>%
      image_rotate(degrees = degrees) %>%
      image_write(
        path = glue::glue(
          "{new_file_name}_page{i}.pdf"
        ),
        format = "pdf"
      )

  }
}

