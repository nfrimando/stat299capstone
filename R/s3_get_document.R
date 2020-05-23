#' Pull Document from S3
#'
#' Extracts document object from S3 based on loan application id and
#' document id
#'
#' @param s3 the s3 object created from the `paws::s3()` function
#' @param loan_application_id id of the loan application
#' @param attachment_id id of the attachment
#' @return The object retrieved from the specified ids
#'
#' @import glue paws magick
#' @importFrom assertthat assert_that
#'
#' @export
s3_get_document <- function(s3, loan_application_id, attachment_id) {

  contents <- s3$list_objects(
    Bucket = "firstcircle",
    Prefix = glue::glue(
      "production/loan_applications/{loan_application_id}/purchase_order_attachments/{attachment_id}/"
    )
  )

  assert_that(
    length(contents$Contents) == 1,
    msg = "Key specified returned more than 1 attachment"
  )

  key <- contents$Contents[[1]]$Key

  object <- s3$get_object(
    Bucket = "firstcircle",
    Key = key
  )

  return(object)
}
