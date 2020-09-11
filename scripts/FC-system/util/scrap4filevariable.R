#Webscrapping files of the model we want. Due to inconsistent filename and variables in each file while lack of documentation.
#Purpose: create some kind of documentation for ease of reading
#saves a .csv file with all the dates, files, and variables of the model. 

library(rvest)
library(tidyverse)

base_url <- "https://thredds.met.no/thredds/catalog/meps25epsarchive/"   #info about date, years and filname of our model
base_urlfile <- "https://thredds.met.no/thredds/dodsC/meps25epsarchive/" #info about variables in each file

base_url <- "https://thredds.met.no/thredds/catalog/aromearcticarchive/"
base_urlfile <- "https://thredds.met.no/thredds/dodsC/aromearcticarchive/" #info about variables in each file

#####Finding years of data that exist ###############
html_YYYY <- read_html(str_c(base_url,"catalog.html", collapse = ""))
Y <- html_YYYY %>%
  html_nodes("table tr td a tt") %>% 
  html_text()%>% 
  str_extract("^[0-9]*")
YYYYl = Y[Y != "" & !is.na(Y)] #list of years [2020, 2019, ...]
#YYYYl <- list(2019)
######################################################

files_all <- list() #list of all the files
dates_all <- list() #the valid date of a file
hours_all <- list() #the validh our of a file
var_all <- list()   #all variables in a file

not_empty <- TRUE
Yindx <- 0
 while (not_empty) { #keeps track of number of years we have data
   Yindx <- Yindx+1
   print(YYYYl[Yindx])
   
   #####Finding months of data that exist for a specific year ###############
   url_MM <- str_c(base_url,"/",YYYYl[Yindx],"/","catalog.html", collapse = "")
   html_MM <- read_html(url_MM)
   M <- html_MM %>%
     html_nodes("table tr td a tt") %>% 
     html_text()%>% 
     str_extract("^[0-9]*")
   MMl = M[M != "" & !is.na(M) & M<=12]
   #MMl = "12"
   ##########################################################################
   for (MM in MMl) {
     print(MM)
     
     #####Finding days of data that exist for a specific year and month ###############
     url_DD <- str_c(base_url,"/",YYYYl[Yindx],"/", MM,"/","catalog.html", collapse = "")
     html_DD <- read_html(url_DD)
     D <- html_DD %>%
       html_nodes("table tr td a tt") %>% 
       html_text()%>% 
       str_extract("^[0-9]*")
     DDl = D[D != "" & !is.na(D) & D<=31][-1]
     #DDl = list("14","15")
     ################################################################################
     
     for (DD in DDl) { 
        #####for every day we find what filenames we have available########
        url <- str_c(base_url, YYYYl[Yindx],"/",MM,"/",DD,"/","catalog.html", collapse = "")
        print(url)
        html <- read_html(url)
        print(html)
        d <- html %>%
          html_nodes("table tr td a tt") %>% 
          html_text() %>% 
          str_extract(".*?.nc")
        files <- d[!is.na(d)] # contains all files available for that day
        ###################################################################
        print(files)
        if (!identical(files, character(0))) { # As long as there is files in the folder do this:
          for (f in files) {
            #####for every file we find what variables we have available########
            url_file <- str_c(base_urlfile, YYYYl[Yindx],"/",MM,"/",DD,"/", f, ".html",collapse = "")
            print(url_file)
            
            htmlfile <- try(read_html(url_file))
            if(inherits(htmlfile,"try-error")) {
              next
            }
            print(htmlfile)
            dfile <- htmlfile %>%
              html_nodes("tr td b") %>% 
              html_text()
            divider <- "^(.*?)(?=: )"
            splitdfile <-  str_extract( dfile, divider) #every variable in that file
            ###################################################################
            
            #Fill lists with data and handle date/hour
            var_all <- c(list(splitdfile),var_all)
            files_all <- c(f, files_all)
            dates <- str_c(YYYYl[Yindx],MM,DD, collapse = "")
            dates_all <- c(dates, dates_all)
            divider <- "([0-9]{2})(?=Z.*.nc$)" #"^([0-9]{2})(?=Z.*)(?=.nc )"
            hours <-  str_extract( f, divider)
            hours_all <- c(hours, hours_all)
          }
        }
     }
   }
  if (Yindx == length(YYYYl)) {
       not_empty <- FALSE
  }
 }

#Fill dataframe and write to file###########################
df_files <- data.frame(File = matrix(unlist(files_all), nrow=length(files_all), byrow=T),stringsAsFactors=FALSE)
df_files$Date <- unlist(dates_all)
df_files$Hour <- unlist(hours_all)
df_files$var <- var_all #list of all variables in the file
#df_files <- apply(df_files,2,as.character) #write.csv dont like list format, alternative
df_files$var <- vapply(df_files$var, paste, collapse = ", ", character(1L)) #write.csv dont like list format

write.csv(df_files, "./aa_filesandvar.csv", row.names = FALSE)
##############################################################
