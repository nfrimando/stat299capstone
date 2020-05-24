

# Ran and downloaded from po_links.sql
buyer_loanapp.dt <- read_csv("data/buyers_loanapps.csv") %>%
  arrange(buyer_id, loan_application_id, attachment_id)

library(stat299capstone)
library(magrittr)

s3 <- paws::s3()

# Get One Document --------------------------------------------------------

loan_application_id <- buyer_loanapp.dt$loan_application_id[1]
attachment_id <- buyer_loanapp.dt$attachment_id[1]
buyer_id <- buyer_loanapp.dt$buyer_id[1]

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

filtered_dataset <- buyer_loanapp.dt %>%
  filter(buyer_id == 9211)

for (i in 1:nrow(filtered_dataset)) {
  loan_application_id <- filtered_dataset$loan_application_id[i]
  attachment_id <- filtered_dataset$attachment_id[i]
  buyer_id <- filtered_dataset$buyer_id[i]

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
}

# Done buyer ids
# 9211

