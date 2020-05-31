#' Extract array of image data from folder
#'
#' @description http://www.di.fc.ul.pt/~jpn/r/GraphicalTools/EBImage.html
#'
#' @param path path to folder with images
#' @param width resize width
#' @param height resize height
#' @return array with first dimension as image number, and 2nd and 3rd as image dimensions
#'
#' @import dplyr purrr EBImage pbapply
#'
#' @export
#'
#' @examples
#' collect_images("data/documents_processed/b12754")
collect_images <- function(path, width = 210, height = 300) {

  png_list <- list.files(path, full.names = TRUE)

  pb <- progress_estimated(length(png_list))
  test <- purrr::map(
    png_list,
    function(x) {
      pb$tick()$print()
      x %>%
        EBImage::readImage() %>%
        EBImage::imageData() %>% # Turn to array
        # {.[200:400,200:400,]} %>%
        {1 - .} %>%
        EBImage::resize(w = width, h = height) %>%
        EBImage::channel(mode = "gray") %>%
        apply(c(1,2), mean)
    }
  )

  images <- test[[1]]

  for (i in 2:length(test)) {
    images <- abind(images, test[[i]], along = 3)
  }

  return(images)

}
