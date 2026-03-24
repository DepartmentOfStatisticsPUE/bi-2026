## ----eval = FALSE-------------------------------------------------------------
# install.packages(
#                   c("rvest",     ## do web-scrapingu
#                    "xml2",       ## przetwarzanie XML
#                    "jsonlite",   ## wczytywanie plików w formacie JSON
#                    "stringi")    ## przetwarzanie napisow
#                  )


## -----------------------------------------------------------------------------
library(rvest)
library(stringi)
library(xml2)
library(jsonlite)


## -----------------------------------------------------------------------------
strona <- read_html("https://archiwum.pracuj.pl/archive/offers?Year=2026&Month=1&PageNumber=1")
strona


## -----------------------------------------------------------------------------
strona |>
  html_nodes("div.offers_item")


## -----------------------------------------------------------------------------
strona |>
  html_nodes("div.offers_item") |>
  html_nodes("span.offers_item_link_cnt_part") |>
  html_text() |>
  head(10)

