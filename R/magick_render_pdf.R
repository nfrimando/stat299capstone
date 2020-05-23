#' Render Raw Vector into PDF
#'
#' Turn raw vector into a pdf after specifying a path
#'
#' @param vector vector of raw data from s3
#' @param path file path where object is rendered
#' @return creates a pdf in the file path
#'
#' @import magick
#' @importFrom assertthat assert_that
#'
#' @export
#'
#' @examples
#' s3 <- paws::s3()
#' content <- s3 %>% s3_get_document(loan_application_id = 950, attachment_id = 12227)
#' content$Body %>% magick_render_pdf(path = "document.pdf")
magick_render_pdf <- function(vector, path) {
  vector %>%
    image_read() %>%
    image_write(path = path, format = "pdf")
}
