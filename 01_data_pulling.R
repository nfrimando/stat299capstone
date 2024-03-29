library(tidyverse)

# Ran and downloaded from po_links.sql
buyer_loanapp.dt <- read_csv("data/buyers_loanapps.csv") %>%
  arrange(buyer_id, loan_application_id, attachment_id)

library(stat299capstone)
library(magrittr)

s3 <- paws::s3()

# Get One Document --------------------------------------------------------

loan_application_id <- 8209
attachment_id <- 833
buyer_id <- 2441 # Kantar

s3 %>%
  s3_get_document(
    loan_application_id = loan_application_id,
    attachment_id = attachment_id
  ) %$%
  Body %>%
  magick_render_pdf(
    path = glue::glue(
      "data/documents/b{buyer_id}_la{loan_application_id}_a{attachment_id}.pdf"
    )
  )

# Looping -----------------------------------------------------------------

buyer_id_specific <- 8607 # Modify Me!

if (!(glue("b{buyer_id_specific}") %in% list.files("data/documents"))) {
  dir.create(glue("data/documents/b{buyer_id_specific}"))
}

filtered_dataset <- buyer_loanapp.dt %>%
  filter(buyer_id == buyer_id_specific)

for (i in 1:nrow(filtered_dataset)) {
  loan_application_id <- filtered_dataset$loan_application_id[i]
  attachment_id <- filtered_dataset$attachment_id[i]
  buyer_id_specific <- filtered_dataset$buyer_id[i]

  s3 %>%
    s3_get_document(
      loan_application_id = loan_application_id,
      attachment_id = attachment_id
    ) %$%
    Body %>%
    magick_render_pdf(
      path = glue::glue(
        "data/documents/b{buyer_id_specific}/b{buyer_id_specific}_la{loan_application_id}_a{attachment_id}.pdf"
      )
    )
}

# Done buyer ids
# 9211, 9763, 9592, 12754, 2929
# Skipped
# 12144 -- same as SM
# 2681 -- lazada has different shiz

