library(rvest)
library(tidyverse)
base_url <- "https://thredds.met.no/thredds/catalog/meps25epsarchive/"
base_url <- "https://thredds.met.no/thredds/catalog/aromearcticarchive/"
url_YYYY <- str_c(base_url,"catalog.html", collapse = "")
html_YYYY <- read_html(url_YYYY)
Y <- html_YYYY %>%
  html_nodes("table tr td a tt") %>% 
  html_text()%>% 
  str_extract("^[0-9]*")
YYYYl = Y[Y != "" & !is.na(Y)]
#YYYYl <- list(2016)
files_all <- list()
dates_all <- list()
hours_all <- list()

not_empty <- TRUE
Yindx <- 0
while (not_empty) {
  Yindx <- Yindx+1
  url_MM <- str_c(base_url,"/",YYYYl[Yindx],"/","catalog.html", collapse = "")
  html_MM <- read_html(url_MM)
  M <- html_MM %>%
    html_nodes("table tr td a tt") %>% 
    html_text()%>% 
    str_extract("^[0-9]*")
  MMl = M[M != "" & !is.na(M) & M<=12]
  print(YYYYl[Yindx])
  for (MM in MMl) {
    print(MM)
    url_DD <- str_c(base_url,"/",YYYYl[Yindx],"/", MM,"/","catalog.html", collapse = "")
    html_DD <- read_html(url_DD)
    D <- html_DD %>%
      html_nodes("table tr td a tt") %>% 
      html_text()%>% 
      str_extract("^[0-9]*")
    DDl = D[D != "" & !is.na(D) & D<=31][-1]
    for (DD in DDl) {
      print(DD)
      url <- str_c(base_url, YYYYl[Yindx],"/",MM,"/",DD,"/","catalog.html", collapse = "")
      print(url)
      html <- read_html(url)
      
      d <- html %>%
        html_nodes("table tr td a tt") %>% 
        html_text() %>% 
        str_extract(".*?.nc")
      
      files <- d[!is.na(d)]
      
      if (!identical(files, character(0))) {
        divider <- str_c(YYYYl[Yindx],MM,DD,"T|Z.nc")
        splitfilename <-  str_split( files, divider)
        df <- data.frame(matrix(unlist(splitfilename), nrow=length(splitfilename), byrow=T),stringsAsFactors=FALSE)[1:2]
        dates <- list(rep(str_c(YYYYl[Yindx],MM,DD, collapse = ""), length(files)))
        hours <- df[2]
        
        files_all <- c(files, files_all)
        dates_all <- c(dates, dates_all)
        hours_all <- c(hours, hours_all)
      }
    }
  }
  if (Yindx == length(YYYYl)) {
    not_empty <- FALSE
  }
}
df_files <- data.frame(File = matrix(unlist(files_all), nrow=length(files_all), byrow=T),stringsAsFactors=FALSE)
df_files$Date <- unlist(dates_all)
df_files$Hour <- unlist(hours_all)
write.csv(df_files, "./aa_files.csv", row.names = FALSE)
#print(df_files)
