# Load required libraries
require(data.table)
# library(sqldf)
library(dplyr)
library(DT)
library(rCharts)
library(ggplot2)
library(ggcorrplot)
library(tidyr)
library(reshape2)
library(lubridate)
library(GGally)
library(plyr)
library(gridExtra)
library(Matrix)
library(glmnet)

# Read data
load("movies_merged")
data = movies_merged[movies_merged$Type=='movie',]

#explore data
table(data$Year) # 1888 ~ 2018
colnames(data)

str_to_num = function(s){
  if (grepl('h', s) && grepl('min', s)) {
    hour = as.numeric(unlist(strsplit(s, ' '))[1])
    min = as.numeric(unlist(strsplit(s, ' '))[3])
    return(60*hour+min)
  }
  else if (grepl('h', s) && !grepl('min', s)) {
    hour = as.numeric(unlist(strsplit(s, ' '))[1])
    return(60*hour)
  }
  else if (!grepl('h', s) && grepl('min', s)) {
    min = as.numeric(unlist(strsplit(s, ' '))[1])
    return(min)
  }
  else {
    return(NA)
  }
  
}

str_to_sum = function(s){
  l = unlist(strsplit(s, "[^0-9]+"))
  l = l[l != '']
  result = sum(as.numeric(l))
  return(result)
}

convert = function(n){
  if (n == 0) {
    return('no nominations or awards')
  }
  else if (n > 12) {
    return('many nominations or awards')
  }
  else {
    return('some nominations or awards')
  }
}

## Helper functions

#' Aggregate dataset by year
#' 
#' @param dt data.table
#' @param minYear
#' @param maxYear
#' @return data.table
#'
groupByYear <- function(dt, minYear, maxYear) {
  result <- dt %>% filter(Year >= minYear, Year <= maxYear) 
  return(result)
}

#' histogram plot
#'
#' @param dt data.table
#' @param feature

plotRuntime <- function(dt) {
  dt$Runtime = unname(sapply(dt$Runtime, str_to_num))
  #histogram to show the distribution of runtime value
 ggplot(dt, aes(x=Runtime)) + geom_histogram(binwidth=20, fill = "blue", alpha=0.3) + xlab("Runtime in Minutes") + ylab('Count') +
   theme_bw(base_size = 20) 

  
}

plotNoOfWins <- function(dt) {
  movies_award = dt
  movies_award$wins_or_nominations = unname(sapply(movies_award$Awards, str_to_sum))
  movies_award1 = subset(movies_award, movies_award$wins_or_nominations != 0)
  
  #a histogram show the distribution of number of wins or nominations
 ggplot(movies_award1, aes(x=wins_or_nominations)) + geom_histogram(binwidth=20, fill = "blue", alpha=0.3) +
   xlab("Number of Wins/Nominations") + ylab('Count') + coord_cartesian(xlim = c(0, 100)) + theme_bw(base_size = 20)
   

}

plotRuntimeAndYear <- function(dt) {
  dt$Runtime = unname(sapply(dt$Runtime, str_to_num))
  ggplot(dt, aes(x=Year, y=Runtime)) + geom_point(size=2, color='blue', alpha=0.3) + ylab(label="Runtime") +
    xlab("Year") + theme_bw(base_size = 20)

}

plotRuntimeAndBudget <- function(dt) {
  dt$Runtime = unname(sapply(dt$Runtime, str_to_num))
  ggplot(dt, aes(x=Budget, y=Runtime)) + geom_point(size=2, color='blue', alpha=0.3) + ylab(label="Runtime") +
    xlab("Budget") + theme_bw(base_size = 20)
}

plotTitleAndGenres <- function(dt) {
  #parse each text string in Genre into a binary vector
  movies = dt[dt$Genre != 'N/A',]
  movies$Genre = strsplit(movies$Genre, ', ')
  movies_long = unnest(movies, Genre)
  movies_wide = dcast(movies_long, Title ~ Genre, function(x) 1, fill = 0)
  movies = merge(dt, movies_wide) #add the binary vector to the original dataframe
  
  #plot the disctribution of title counts across different genres
  movies_count = data.frame(Genre=names(movies[,40:67]), Count=colSums(movies[,40:67]))
  movies_count$Proportion = (movies_count$Count/sum(movies_count$Count))*100
  ggplot(movies_count, aes(reorder(Genre, Count), Count)) + geom_bar(stat='identity',fill = "blue", alpha=0.3) + 
    coord_flip() + ylab('Count of Titles') + xlab('Genres') + theme_bw(base_size = 20)
}

plotGrossAndGenres <- function(dt) {
  #parse each text string in Genre into a binary vector
  movies = dt[dt$Genre != 'N/A',]
  movies$Genre = strsplit(movies$Genre, ', ')
  movies_long = unnest(movies, Genre)
  movies_wide = dcast(movies_long, Title ~ Genre, function(x) 1, fill = 0)
  movies = merge(dt, movies_wide) #add the binary vector to the original dataframe
  
  #plot the distribution of gross revenue across top 10 genres
  movies_gross = movies[!is.na(movies$Gross),]
  DF = movies_gross[c(38, 40:67)] #create a subset of orignial dataframe which contains only the Gross column and the indicator variables of Genres
  DF1 = melt(DF, id.vars="Gross")
  DF2 = subset(DF1, value>0) 
  DF3 = rbind(subset(DF2, variable == 'Drama'), subset(DF2, variable == 'Comedy'), subset(DF2, variable == 'Short'), subset(DF2, variable == 'Romance'), subset(DF2, variable == 'Action'), subset(DF2, variable == 'Crime'), subset(DF2, variable == 'Thriller'), subset(DF2, variable == 'Documentary'), subset(DF2, variable == 'Adventure'), subset(DF2, variable == 'Animation'))
  ggplot(DF3, aes(reorder(variable, -Gross, median), Gross)) + geom_boxplot(varwidth=T, fill="blue", alpha=0.3) + 
    coord_flip() + scale_x_discrete("Genres") + ylab(label="Gross") + theme_bw(base_size = 20)
}

plotGrossAndMonth <- function(dt) {
  movies = dt[dt$Genre != 'N/A',]
  movies$Genre = strsplit(movies$Genre, ', ')
  movies_long = unnest(movies, Genre)
  movies_wide = dcast(movies_long, Title ~ Genre, function(x) 1, fill = 0)
  
  #No discrepancy is found between Year and Release, so no rows are removed.
  movies_remove_final = dt
  
  movies_remove_final$Released_Month = month(movies_remove_final$Released)#create a release month column
  movies_release = subset(merge(movies_remove_final, movies_wide), !is.na(Gross))#add the genre indicator variables to dataframe movies_remove_final, and remove the rows where Gross is NA
  
  #plot shows the relationship between Gross and Released Month
  ggplot(movies_release, aes(x=Released_Month, y=Gross)) + geom_point(size=2, color='blue', alpha=0.3) + ylab(label="Gross Revenue") + 
    xlab("Release Month") + theme_bw(base_size = 20)
}

plotGrossAndMonthAndGenres <- function(dt) {
  movies = dt[dt$Genre != 'N/A',]
  movies$Genre = strsplit(movies$Genre, ', ')
  movies_long = unnest(movies, Genre)
  movies_wide = dcast(movies_long, Title ~ Genre, function(x) 1, fill = 0)
  
  #No discrepancy is found between Year and Release, so no rows are removed.
  movies_remove_final = dt
  
  movies_remove_final$Released_Month = month(movies_remove_final$Released)#create a release month column
  movies_release = subset(merge(movies_remove_final, movies_wide), !is.na(Gross))#add the genre indicator variables to dataframe movies_remove_final, and remove the rows where Gross is NA
  
  #create a dataframe subset that contains Gross, Released Month, and Genre indicator variables
  DF_release = movies_release[c(38, 40:68)]
  DF_release1 = melt(DF_release, id.vars=c("Gross", 'Released_Month'))
  DF_release2 = subset(DF_release1, value>0)
  DF_release3 = rbind(subset(DF_release2, variable == 'Drama'), subset(DF_release2, variable == 'Comedy'), subset(DF_release2, variable == 'Short'), subset(DF_release2, variable == 'Romance'), subset(DF_release2, variable == 'Action'), subset(DF_release2, variable == 'Crime'), subset(DF_release2, variable == 'Thriller'), subset(DF_release2, variable == 'Documentary'), subset(DF_release2, variable == 'Adventure'), subset(DF_release2, variable == 'Animation'))
  DF_release4 = rbind(subset(DF_release2, variable == 'Horror'), subset(DF_release2, variable == 'Family'), subset(DF_release2, variable == 'Mystery'), subset(DF_release2, variable == 'Sci-Fi'), subset(DF_release2, variable == 'Fantasy'), subset(DF_release2, variable == 'Musical'), subset(DF_release2, variable == 'Western'), subset(DF_release2, variable == 'Music'), subset(DF_release2, variable == 'Biography'), subset(DF_release2, variable == 'War'))
  DF_release5 = rbind(subset(DF_release2, variable == 'History'), subset(DF_release2, variable == 'Sport'), subset(DF_release2, variable == 'Adult'), subset(DF_release2, variable == 'Film-Noir'), subset(DF_release2, variable == 'Reality-TV'), subset(DF_release2, variable == 'Talk-Show'), subset(DF_release2, variable == 'News'), subset(DF_release2, variable == 'Game-Show'))
  #plot shows the relationship between Gross and Released Month for different genres,
  #since there are 28 different genres, plot them in a single facet will be too crowded. So
  #Three separate facet plots are created.
  # p1 <- qplot(x=Released_Month, y=Gross, facets=variable~., data=DF_release3, main='Gross Revenue vs. Release Month')
  # p2 <- qplot(x=Released_Month, y=Gross, facets=variable~., data=DF_release4, main='Gross Revenue vs. Release Month')
  # p3 <- qplot(x=Released_Month, y=Gross, facets=variable~., data=DF_release5, main='Gross Revenue vs. Release Month')
  p1 <- ggplot(DF_release3, aes(x=Released_Month, y=Gross)) + geom_point(size=2, color='blue', alpha=0.3) + ylab(label="Gross") +
    facet_grid(variable~.) + xlab("Released Month") + theme_bw(base_size = 20)
  p2 <- ggplot(DF_release4, aes(x=Released_Month, y=Gross)) + geom_point(size=2, color='blue', alpha=0.3) + ylab(label="Gross") +
    facet_grid(variable~.) + xlab("Released Month") + theme_bw(base_size = 20)
  p3 <- ggplot(DF_release5, aes(x=Released_Month, y=Gross)) + geom_point(size=2, color='blue', alpha=0.3) + ylab(label="Gross") +
    facet_grid(variable~.) + xlab("Released Month") + theme_bw(base_size = 20)
  grid.arrange(p1, p2, p3, ncol=1)
}

plotImdbVotesAndRating <- function(dt) {
  ggplot(dt, aes(x=imdbRating, y=imdbVotes)) + geom_point(size=2, color='blue', alpha=0.3) + ylab(label="IMDb Votes") + 
    xlab("IMDb Rating") + theme_bw(base_size = 20)
}

plotTomatoRatingAndReviews <- function(dt) {
  ggplot(dt, aes(x=tomatoRating, y=tomatoReviews)) + geom_point(size=2, color='blue', alpha=0.3) + ylab(label="Number of Reviews") + 
    xlab("Rotten Tomato Critic Rating") + theme_bw(base_size = 20)
}

plotGrossAndAward <- function(dt) {
  movies_award = dt
  movies_award$wins_or_nominations = unname(sapply(movies_award$Awards, str_to_sum))
  movies_award$award_category = unname(sapply(movies_award$wins_or_nominations, convert))
  movies_award_binary = dcast(movies_award, Title ~ award_category, function(x) 1, fill = 0)
  movies_award = merge(movies_award, movies_award_binary)
  
  #boxplot to show the distribution of gross revenue across different award categories
  ggplot(movies_award, aes(reorder(award_category, -Gross, median), Gross)) + geom_boxplot(varwidth=T, fill="blue", alpha=0.3) + 
    coord_flip() + scale_x_discrete("Award Category") + ylab(label="Gross") + theme_bw(base_size = 20)
  
}

plotYear <- function(dt) {
  movies_remove_final = dt
  movies_remove_final$decade = cut(movies_remove_final$Year, seq(1880, 2020, 10), labels = c('1880-1890', '1890-1900', '1900-1910','1910-1920', '1920-1930', '1930-1940', '1940-1950', '1950-1960', '1960-1970', '1970-1980', '1980-1990', '1990-2000', '2000-2010', '2010-2020'))#binning the Year into decades
  movies_decade = count(movies_remove_final, c('decade'))
  
  ggplot(data = movies_decade, aes(y = freq, x = decade)) +
    geom_bar(stat = "identity", width = 0.5, position = "identity", fill = "blue", alpha=0.3) + 
    guides(fill = FALSE) + xlab("Decades") + ylab("Number of movies") + 
    theme_bw(base_size = 20) + theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
}

plotDirector <- function(dt) {
  movies <- dt[!is.na(dt$Gross),]
  movies$Profit <- movies$Gross - movies$Budget
  movies_nonnum <- movies[, !sapply(movies, is.numeric)]
  movies_nonnum$Profit <- movies$Profit
  #based on observation and intuition, choose the columns that may be useful for predicting Profit
  movies_nonnum <- movies_nonnum[, names(movies_nonnum) %in% c('Title', 'Rated', 'Released','Runtime','Genre','Director', 'Writer', 'Actors', 'Language', 'Country','Awards', 'tomatoImage', 'Production', 'Profit')]
  #parse each text string in Director into a binary vector
  movies_nonnum <- movies_nonnum[movies_nonnum$Director != 'N/A',]
  movies_nonnum$Director <- strsplit(movies_nonnum$Director, "(\\s)?,(\\s)?")
  movies_long1 <- unnest(movies_nonnum, Director)
  movies_long1$Director <- paste0("Director_", gsub("\\s","_",movies_long1$Director))
  movies_wide1 <- dcast(movies_long1, Title ~ Director, function(x) 1, fill = 0)
  number_of_directors <- rowSums(movies_wide1[,-1])
  #plot the disctribution of title counts for different directors
  count <- data.frame(Director=names(movies_wide1[-1]), Count=colSums(movies_wide1[,-1]))
  count <- count[order(-count$Count),]
  ggplot(count[1:50,], aes(reorder(Director, Count), Count)) + geom_bar(stat='identity', fill = "blue", alpha=0.3) + 
    coord_flip() + ylab('Count of Titles') + xlab('Director') +  theme_bw(base_size = 20)
  
}

plotActor <- function(dt) {
  movies <- dt[!is.na(dt$Gross),]
  movies$Profit <- movies$Gross - movies$Budget
  movies_nonnum <- movies[, !sapply(movies, is.numeric)]
  movies_nonnum$Profit <- movies$Profit
  #based on observation and intuition, choose the columns that may be useful for predicting Profit
  movies_nonnum <- movies_nonnum[, names(movies_nonnum) %in% c('Title', 'Rated', 'Released','Runtime','Genre','Director', 'Writer', 'Actors', 'Language', 'Country','Awards', 'tomatoImage', 'Production', 'Profit')]
  
  movies_nonnum <- movies_nonnum[movies_nonnum$Actors != 'N/A',]
  movies_nonnum$Actors <- strsplit(movies_nonnum$Actors, "(\\s)?,(\\s)?")
  movies_long2 <- unnest(movies_nonnum, Actors)
  movies_long2$Actors <- paste0("Actor_", gsub("\\s","_",movies_long2$Actors))
  movies_wide2 <- dcast(movies_long2, Title ~ Actors, function(x) 1, fill = 0)
  number_of_actors <- rowSums(movies_wide2[,-1])
  #plot the disctribution of title counts for different directors
  count1 <- data.frame(Actors=names(movies_wide2[-1]), Count=colSums(movies_wide2[,-1]))
  count1 <- count1[order(-count1$Count),]
  ggplot(count1[1:50,], aes(reorder(Actors, Count), Count)) + geom_bar(stat='identity', fill = "blue", alpha=0.3) + 
    coord_flip() + ylab('Count of Titles') + xlab('Actor') + theme_bw(base_size = 20)
}

plotPairwiseCor <- function(dt) {
  DF_plot = dt[, c("imdbRating", "imdbVotes", "tomatoMeter", "tomatoRating", "tomatoReviews", "tomatoFresh", "tomatoRotten", "tomatoUserMeter", "tomatoUserRating", "tomatoUserReviews", "Gross")]
  DF_plot = na.omit(DF_plot)
  corr <- round(cor(DF_plot), 1)
  ggcorrplot(corr, hc.order = TRUE, 
             type = "lower", 
             lab = TRUE, 
             lab_size = 3, 
             method="circle", 
             colors = c("tomato2", "white", "springgreen3"), 
             ggtheme=theme_bw)
  }


plotProfit <- function(dt) {
  movies <- dt[!is.na(dt$Gross),]
  #add the Profit column
  movies$Profit <- movies$Gross - movies$Budget
  #remove all movies released prior to 2000
  movies <- movies[movies$Year >= 2000,]
  #drop gross, domestic_gross, and boxoffice columns
  movies <- movies[ , !(names(movies) %in% c('Gross', 'Domestic_Gross', 'BoxOffice'))]
  
  #keep only the numeric columns
  movies_numeric <- movies[, sapply(movies, is.numeric)]
  #convert Metascore to numeric and add it to movies_numeric
  movies_numeric <- cbind(as.numeric(movies$Metascore), movies_numeric)
  colnames(movies_numeric)[1] <- 'Metascore'
  #since Year and Date columns are almost identical, drop the Date column
  movies_numeric <- movies_numeric[, names(movies_numeric) != 'Date']
  
  #explore the relationships between profit and all 13 variables
  p1 <- ggplot(movies_numeric, aes(x=Metascore, y=Profit)) + geom_point(size=2, color='blue', alpha=0.3) + theme_bw(base_size = 20) + annotate("text", label = "cor = 0.19", x = 25, y = 1.8e+09, color = "red", size = 8)
  p2 <- ggplot(movies_numeric, aes(x=Year, y=Profit)) + geom_point(size=2, color='blue', alpha=0.3) + theme_bw(base_size = 20) + annotate("text", label = "cor = 0.13", x = 2005, y = 1.8e+09, color = "red", size = 8)
  p3 <- ggplot(movies_numeric, aes(x=imdbRating, y=Profit)) + geom_point(size=2, color='blue', alpha=0.3) + theme_bw(base_size = 20) + annotate("text", label = "cor = 0.23", x = 5.0, y = 1.8e+09, color = "red", size = 8)
  p4 <- ggplot(movies_numeric, aes(x=imdbVotes, y=Profit)) + geom_point(size=2, color='blue', alpha=0.3) + theme_bw(base_size = 20) + annotate("text", label = "cor = 0.66", x = 500000, y = 1.8e+09, color = "red", size = 8)
  p5 <- ggplot(movies_numeric, aes(x=tomatoMeter, y=Profit)) + geom_point(size=2, color='blue', alpha=0.3) + theme_bw(base_size = 20) + annotate("text", label = "cor = 0.19", x = 25, y = 1.8e+09, color = "red", size = 8)
  p6 <- ggplot(movies_numeric, aes(x=tomatoRating, y=Profit)) + geom_point(size=2, color='blue', alpha=0.3) + theme_bw(base_size = 20) + annotate("text", label = "cor = 0.21", x = 5.0, y = 1.8e+09, color = "red", size = 8)
  p7 <- ggplot(movies_numeric, aes(x=tomatoReviews, y=Profit)) + geom_point(size=2, color='blue', alpha=0.3) + theme_bw(base_size = 20) + annotate("text", label = "cor = 0.51", x = 100, y = 1.8e+09, color = "red", size = 8)
  p8 <- ggplot(movies_numeric, aes(x=tomatoFresh, y=Profit)) + geom_point(size=2, color='blue', alpha=0.3) + theme_bw(base_size = 20) + annotate("text", label = "cor = 0.44", x = 100, y = 1.8e+09, color = "red", size = 8)
  p9 <- ggplot(movies_numeric, aes(x=tomatoRotten, y=Profit)) + geom_point(size=2, color='blue', alpha=0.3) + theme_bw(base_size = 20) + annotate("text", label = "cor = 0.13", x = 80, y = 1.8e+09, color = "red", size = 8)
  p10 <- ggplot(movies_numeric, aes(x=tomatoUserMeter, y=Profit)) + geom_point(size=2, color='blue', alpha=0.3) + theme_bw(base_size = 20) + annotate("text", label = "cor = 0.24", x = 25, y = 1.8e+09, color = "red", size = 8)
  p11 <- ggplot(movies_numeric, aes(x=tomatoUserRating, y=Profit)) + geom_point(size=2, color='blue', alpha=0.3) + theme_bw(base_size = 20) + annotate("text", label = "cor = 0.30", x = 2, y = 1.8e+09, color = "red", size = 8)
  p12 <- ggplot(movies_numeric, aes(x=tomatoUserReviews, y=Profit)) + geom_point(size=2, color='blue', alpha=0.3) + theme_bw(base_size = 20) + annotate("text", label = "cor = 0.22", x = 1e+07, y = 1.8e+09, color = "red", size = 8)
  p13 <- ggplot(movies_numeric, aes(x=Budget, y=Profit)) + geom_point(size=2, color='blue', alpha=0.3) + theme_bw(base_size = 20) + annotate("text", label = "cor = 0.64", x = 1e+08, y = 1.8e+09, color = "red", size = 8)
  grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, ncol=4, nrow=4)
}

plotLRNumeric <- function(dt) {
  movies <- dt[!is.na(dt$Gross),]
  #add the Profit column
  movies$Profit <- movies$Gross - movies$Budget
  #remove all movies released prior to 2000
  movies <- movies[movies$Year >= 2000,]
  #drop gross, domestic_gross, and boxoffice columns
  movies <- movies[ , !(names(movies) %in% c('Gross', 'Domestic_Gross', 'BoxOffice'))]
  
  ##code for assignment 1
  #keep only the numeric columns
  movies_numeric <- movies[, sapply(movies, is.numeric)]
  #convert Metascore to numeric and add it to movies_numeric
  movies_numeric <- cbind(as.numeric(movies$Metascore), movies_numeric)
  colnames(movies_numeric)[1] <- 'Metascore'
  #since Year and Date columns are almost identical, drop the Date column
  movies_numeric <- movies_numeric[, names(movies_numeric) != 'Date']
  
  final_MSE_train <- NULL
  final_MSE_test <- NULL
  
  f <- seq(0.05, 0.95, by = 0.05)
  for (fraction in f) {
    #divide data into training and test sets
    smp_size <- floor(fraction * nrow(movies_numeric))
    
    all_MSE_train <- NULL
    all_MSE_test <- NULL
    
    #repeat the random partition of dataset 10 times
    for (n in c(12, 47, 35, 67, 85, 91, 55, 102, 219, 49)) {
      set.seed(n)
      train_ind <- sample(seq_len(nrow(movies_numeric)), size = smp_size)
      train <- movies_numeric[train_ind,]
      test <- movies_numeric[-train_ind,]
      
      mylm <- lm(Profit ~., train)
      MSE_train <- mean(residuals(mylm)^2)
      all_MSE_train <- rbind(all_MSE_train, MSE_train)
      
      test_fitted <- predict(mylm, newdata=test[,names(test) != 'Profit'])
      MSE_test <- mean((test$Profit-test_fitted)^2, na.rm = TRUE)
      all_MSE_test <- rbind(all_MSE_test, MSE_test)
    }
    
    final_MSE_train <- rbind(final_MSE_train, mean(all_MSE_train))
    final_MSE_test <- rbind(final_MSE_test, mean(all_MSE_test))
    
  }
  
  training_set_size <- floor(f * nrow(movies_numeric))
  MSE1 <- data.frame(cbind(training_set_size, final_MSE_train, final_MSE_test))
  colnames(MSE1)[2] <- 'MSE_Train'
  colnames(MSE1)[3] <- 'MSE_Test'
  
  MSE11 <- melt(MSE1, id.vars = 'training_set_size')
  ggplot(MSE11, aes(x=training_set_size, y=value, colour=variable)) + geom_line(size=2) + 
    geom_point(size=4, color='blue', alpha=0.3) + ylab(label="MSE") + 
    xlab("Size of Training Set") + theme_bw(base_size = 20)
    
}

plotLRTrans <- function(dt) {
  movies <- dt[!is.na(dt$Gross),]
  #add the Profit column
  movies$Profit <- movies$Gross - movies$Budget
  #remove all movies released prior to 2000
  movies <- movies[movies$Year >= 2000,]
  #drop gross, domestic_gross, and boxoffice columns
  movies <- movies[ , !(names(movies) %in% c('Gross', 'Domestic_Gross', 'BoxOffice'))]
  
  ##code for assignment 1
  #keep only the numeric columns
  movies_numeric <- movies[, sapply(movies, is.numeric)]
  #convert Metascore to numeric and add it to movies_numeric
  movies_numeric <- cbind(as.numeric(movies$Metascore), movies_numeric)
  colnames(movies_numeric)[1] <- 'Metascore'
  #since Year and Date columns are almost identical, drop the Date column
  movies_numeric <- movies_numeric[, names(movies_numeric) != 'Date']
  
  X <- movies_numeric[,!grepl("Profit", colnames(movies_numeric))]
  X <- apply(X, 2, as.numeric)
  Y <- as.matrix(movies_numeric$Profit)
  
  #create transformed variables data frame based on highest correlation result
  X <- as.data.frame(X)
  X$Year <- X$Year^3
  X$imdbRating <- X$imdbRating^3
  X$tomatoRating <- X$tomatoRating^3
  X$tomatoReviews <- X$tomatoReviews^3
  X$tomatoFresh <- X$tomatoFresh^2
  X$tomatoRotten <- X$tomatoRotten^3
  X$tomatoUserMeter <- X$tomatoUserMeter^2
  X$tomatoUserRating <- X$tomatoUserRating^3
  X$tomatoUserReviews <- X$tomatoUserReviews^(1/3)
  X$Budget <- X$Budget^2
  colnames(X) <- c('Metascore', 'Year^3', 'imdbRating^3', 'imdbVotes', 'tomatoMeter', 'tomatoRating^3', 'tomatoReviews^3', 'tomatoFresh^2', 'tomatoRotten^3', 'tomatoUserMeter^2', 'tomatoUserRating^3', 'tomatoUserReviews^(1/3)', 'Budget^2')
  
  #based on intuition and the fact that imdbVotes and Budget have the highest correlations with Profit
  #create product of two variables
  X$`imdbVotesBudget^2` <- X$imdbVotes * X$`Budget^2`
  X$`tomatoReviews^3Budget^2` <- X$`tomatoReviews^3` * X$`Budget^2`
  X$`imdbRating^3Budget^2` <- X$`imdbRating^3` * X$`Budget^2`
  X$`tomatoRating^3Budget^2` <- X$`tomatoRating^3` * X$`Budget^2`
  X$`tomatoMeterBudget^2` <- X$tomatoMeter * X$`Budget^2`
  X$`tomatoFresh^2Budget^2` <- X$`tomatoFresh^2` * X$`Budget^2`
  X$`tomatoUserMeter^2Budget^2` <- X$`tomatoUserMeter^2` * X$`Budget^2`
  X$`tomatoUserRating^3Budget^2` <- X$`tomatoUserRating^3` * X$`Budget^2`
  X$`MetascoreBudget^2` <- X$Metascore * X$`Budget^2`
  
  #in the plot of tomatoUserReviews and Profit, a clear separation of data can be seen. so a
  #new variable is_tomatoUserReviews_smaller_than_5M is added which is derived from binning the tomatoUserReviews variable
  X$is_tomatoUserReviews_smaller_than_5M <- ifelse(movies_numeric$tomatoUserReviews < 5e+06, 1, 0)
  
  movies_numeric_transformed <- cbind(movies$Title, X, Y)
  names(movies_numeric_transformed)[1] <- 'Title'
  names(movies_numeric_transformed)[ncol(movies_numeric_transformed)] <- 'Profit'
  
  movies_numeric_transformed <- na.omit(movies_numeric_transformed)
  
  #divide the data into training and test set, train the model using selected transformed variables
  final_MSE_train <- NULL
  final_MSE_test <- NULL
  
  f <- seq(0.05, 0.95, by = 0.05)
  for (fraction in f) {
    #divide data into training and test sets
    smp_size <- floor(fraction * nrow(movies_numeric_transformed))
    
    all_MSE_train <- NULL
    all_MSE_test <- NULL
    
    #repeat the random partition of dataset 10 times
    for (n in c(12, 47, 35, 67, 85, 91, 55, 102, 219, 49)) {
      set.seed(n)
      train_ind <- sample(seq_len(nrow(movies_numeric_transformed)), size = smp_size)
      train <- movies_numeric_transformed[train_ind,-1]
      test <- movies_numeric_transformed[-train_ind,-1]
      
      mylm <- lm(Profit~., train)
      MSE_train <- mean(residuals(mylm)^2)
      all_MSE_train <- rbind(all_MSE_train, MSE_train)
      
      test_fitted <- predict(mylm, newdata=test[,names(test) != 'Profit'])
      MSE_test <- mean((test$Profit-test_fitted)^2)
      all_MSE_test <- rbind(all_MSE_test, MSE_test)
    }
    
    final_MSE_train <- rbind(final_MSE_train, mean(all_MSE_train))
    final_MSE_test <- rbind(final_MSE_test, mean(all_MSE_test))
    
  }
  
  training_set_size <- floor(f * nrow(movies_numeric_transformed))
  MSE2 <- data.frame(cbind(training_set_size, final_MSE_train, final_MSE_test))
  colnames(MSE2)[2] <- 'MSE_Train'
  colnames(MSE2)[3] <- 'MSE_Test'
  
  MSE21 <- melt(MSE2, id.vars = 'training_set_size')
  ggplot(MSE21, aes(x=training_set_size, y=value, colour=variable)) + geom_line(size=2) + 
    geom_point(size=4, color='blue', alpha=0.3) +
    ylab(label="MSE") + 
    xlab("Size of Training Set") + theme_bw(base_size = 20)
  
}

plotLRCate <- function(dt) {
  movies <- dt[!is.na(dt$Gross),]
  #add the Profit column
  movies$Profit <- movies$Gross - movies$Budget
  #remove all movies released prior to 2000
  movies <- movies[movies$Year >= 2000,]
  #drop gross, domestic_gross, and boxoffice columns
  movies <- movies[ , !(names(movies) %in% c('Gross', 'Domestic_Gross', 'BoxOffice'))]
  
  #keep only the non-numeric columns
  movies_nonnum <- movies[, !sapply(movies, is.numeric)]
  movies_nonnum$Profit <- movies$Profit
  #based on observation and intuition, choose the columns that may be useful for predicting Profit
  movies_nonnum <- movies_nonnum[, names(movies_nonnum) %in% c('Title', 'Rated', 'Released','Runtime','Genre','Director', 'Writer', 'Actors', 'Language', 'Country','Awards', 'tomatoImage', 'Production', 'Profit')]
  
  
  #convert a Runtime string into a numeric value in minutes
  #code from project part I
  str_to_num <- function(s){
    if (grepl('h', s) && grepl('min', s)) {
      hour = as.numeric(unlist(strsplit(s, ' '))[1])
      min = as.numeric(unlist(strsplit(s, ' '))[3])
      return(60*hour+min)
    }
    else if (grepl('h', s) && !grepl('min', s)) {
      hour <- as.numeric(unlist(strsplit(s, ' '))[1])
      return(60*hour)
    }
    else if (!grepl('h', s) && grepl('min', s)) {
      min <- as.numeric(unlist(strsplit(s, ' '))[1])
      return(min)
    }
    else {
      return(NA)
    }
    
  }
  movies_nonnum$Runtime = unname(sapply(movies_nonnum$Runtime, str_to_num))
  #end code from project part I
  
  #convert Awards string to total number of wins and nominations
  #code from project part I
  str_to_sum <- function(s){
    if (s == 'N/A') {
      return(NA)
    }
    else {
      l <- unlist(strsplit(s, "[^0-9]+"))
      l <- l[l != '']
      result <- sum(as.numeric(l))
      return(result)
    }
    
  }
  movies_nonnum$Awards <- unname(sapply(movies_nonnum$Awards, str_to_sum))
  #end code from project part I
  
  #convert Released to Released_Month (Released Year information is captured in numeric variables, month information was shown
  #to be related to Gross in project part I)
  movies_nonnum$Released <- month(movies_nonnum$Released)
  colnames(movies_nonnum)[3] <- 'Released_Month'
  
  #parse each text string in Genre into a binary vector
  #code from project part I
  movies_nonnum <- movies_nonnum[movies_nonnum$Genre != 'N/A',]
  movies_nonnum$Genre <- strsplit(movies_nonnum$Genre, ', ')
  movies_long <- unnest(movies_nonnum, Genre)
  movies_long$Genre <- paste0("Genre_", gsub("\\s","_",movies_long$Genre))
  movies_wide <- dcast(movies_long, Title ~ Genre, function(x) 1, fill = 0)
  movies_nonnum <- merge(movies_nonnum, movies_wide)
  #end code from project part I
  
  #parse each text string in Director into a binary vector
  movies_nonnum <- movies_nonnum[movies_nonnum$Director != 'N/A',]
  movies_nonnum$Director <- strsplit(movies_nonnum$Director, "(\\s)?,(\\s)?")
  movies_long1 <- unnest(movies_nonnum, Director)
  movies_long1$Director <- paste0("Director_", gsub("\\s","_",movies_long1$Director))
  movies_wide1 <- dcast(movies_long1, Title ~ Director, function(x) 1, fill = 0)
  number_of_directors <- rowSums(movies_wide1[,-1])

  
  #consider the top 123 directors where count of titles >= 5
  count$Director <- as.character(count$Director)
  movies_wide1_top <- movies_wide1[,c('Title',count$Director[1:123])]
  #add a 124th variable,Director_others, if there is no other directors, Director_others equals to 0, else, it equals to 1
  movies_wide1_top$number_of_top_directors <- rowSums(movies_wide1_top[,-1])
  movies_wide1_top <- cbind(movies_wide1_top, number_of_directors)
  movies_wide1_top$Director_others <- ifelse((movies_wide1_top$number_of_directors-movies_wide1_top$number_of_top_directors) == 0, 0, 1)
  movies_wide1_top <- movies_wide1_top[, !colnames(movies_wide1_top) %in% c('number_of_directors','number_of_top_directors')]
  
  movies_nonnum <- merge(movies_nonnum, movies_wide1_top)
  
  #parse each text string in Actor into a binary vector
  movies_nonnum <- movies_nonnum[movies_nonnum$Actors != 'N/A',]
  movies_nonnum$Actors <- strsplit(movies_nonnum$Actors, "(\\s)?,(\\s)?")
  movies_long2 <- unnest(movies_nonnum, Actors)
  movies_long2$Actors <- paste0("Actor_", gsub("\\s","_",movies_long2$Actors))
  movies_wide2 <- dcast(movies_long2, Title ~ Actors, function(x) 1, fill = 0)
  number_of_actors <- rowSums(movies_wide2[,-1])

  #consider the top 167 actors where count of titles > 10
  count1$Actors <- as.character(count1$Actors)
  movies_wide2_top <- movies_wide2[,c('Title',count1$Actors[1:167])]
  #add a 168th variable,Actor_others, if there is no other actors, Actor_others equals to 0, else, it equals to 1
  movies_wide2_top$number_of_top_actors <- rowSums(movies_wide2_top[,-1])
  movies_wide2_top <- cbind(movies_wide2_top, number_of_actors)
  movies_wide2_top$Actor_others <- ifelse((movies_wide2_top$number_of_actors-movies_wide2_top$number_of_top_actors) == 0, 0, 1)
  movies_wide2_top <- movies_wide2_top[, !colnames(movies_wide2_top) %in% c('number_of_actors','number_of_top_actors')]
  
  movies_nonnum <- merge(movies_nonnum, movies_wide2_top)
  
  #parse each text string in Writer into a binary vector
  movies_nonnum <- movies_nonnum[movies_nonnum$Writer != 'N/A',]
  movies_nonnum$Writer <- strsplit(movies_nonnum$Writer, "(\\s)?,(\\s)?")
  movies_long3 <- unnest(movies_nonnum, Writer)
  movies_long3$Writer <- gsub("\\s*\\([^\\)]+\\)","",movies_long3$Writer)
  movies_long3$Writer <- paste0("Writer_", gsub("\\s","_",movies_long3$Writer))
  movies_wide3 <- dcast(movies_long3, Title ~ Writer, function(x) 1, fill = 0)
  number_of_writers <- rowSums(movies_wide3[,-1])

  #consider the top 129 writers where count of titles > 5
  count2$Writer <- as.character(count2$Writer)
  movies_wide3_top <- movies_wide3[,c('Title',count2$Writer[1:129])]
  
  #add a 130th variable,Writer_others, if there is no other writers, Writer_others equals to 0, else, it equals to 1
  movies_wide3_top$number_of_top_writers <- rowSums(movies_wide3_top[,-1])
  movies_wide3_top <- cbind(movies_wide3_top, number_of_writers)
  movies_wide3_top$Writer_others <- ifelse((movies_wide3_top$number_of_writers-movies_wide3_top$number_of_top_writers) == 0, 0, 1)
  movies_wide3_top <- movies_wide3_top[, !colnames(movies_wide3_top) %in% c('number_of_writers','number_of_top_writers')]
  
  movies_nonnum <- merge(movies_nonnum, movies_wide3_top)
  
  #convert Language to number of languages
  movies_nonnum$Language <- strsplit(movies_nonnum$Language, ', ')
  movies_nonnum$Language <- sapply(movies_nonnum$Language, length)
  
  #convert Country to number of countries
  movies_nonnum$Country <- strsplit(movies_nonnum$Country, ', ')
  movies_nonnum$Country <- sapply(movies_nonnum$Country, length)
  
  #convert Rated into a binary vector
  movies_nonnum <- movies_nonnum[movies_nonnum$Rated != 'N/A',]
  convert <- function(s) {
    if(s == 'UNRATED') {
      s <- 'NOT RATED'
      return(s)
    }
    else {
      return(s)
    }
  }
  movies_nonnum$Rated <- sapply(movies_nonnum$Rated, convert)
  movies_nonnum_wide <- dcast(movies_nonnum, Title ~ Rated, function(x) 1, fill = 0)
  movies_nonnum <- merge(movies_nonnum, movies_nonnum_wide)
  
  #convert tomatoImage into a binary vector
  movies_nonnum <- movies_nonnum[movies_nonnum$tomatoImage != 'N/A',]
  movies_nonnum_wide1 <- dcast(movies_nonnum, Title ~ tomatoImage, function(x) 1, fill = 0)
  movies_nonnum <- merge(movies_nonnum, movies_nonnum_wide1)
  
  #convert Production into a binary vector
  movies_nonnum <- movies_nonnum[movies_nonnum$Production != 'N/A',]
  movies_wide4 <- dcast(movies_nonnum, Title ~ Production, function(x) 1, fill = 0)
  
  
  #consider the top 49 productions where count of titles >= 10
  count3$Production <- as.character(count3$Production)
  movies_wide4_top <- movies_wide4[,c('Title',count3$Production[1:49])]
  
  #add a 50th variable,Production_others, if production is not in top productions, Writer_others equals to 1, else, it equals to 0
  movies_wide4_top$Production_others <- ifelse(movies_nonnum$Production %in% count3$Production[1:49], 0, 1)
  
  movies_nonnum <- merge(movies_nonnum, movies_wide4_top)
  
  #drop the Genre, Director, and Actors columns
  movies_nonnum <- movies_nonnum[,!colnames(movies_nonnum) %in% c('Genre','Director','Actors','Rated','Writer','tomatoImage','Production')]
  
  ##code for assignment 4
  movies_nonnum_lm <- movies_nonnum[, names(movies_nonnum) != 'Title']
  
  final_MSE_train <- NULL
  final_MSE_test <- NULL
  
  f <- seq(0.05, 0.95, by = 0.05)
  for (fraction in f) {
    #divide data into training and test sets
    smp_size <- floor(fraction * nrow(movies_nonnum_lm))
    
    all_MSE_train <- NULL
    all_MSE_test <- NULL
    
    #repeat the random partition of dataset 10 times
    for (n in c(12, 47, 35, 67, 85, 91, 55, 102, 219, 49)) {
      set.seed(n)
      train_ind <- sample(seq_len(nrow(movies_nonnum_lm)), size = smp_size)
      train <- movies_nonnum_lm[train_ind,]
      test <- movies_nonnum_lm[-train_ind,]
      
      mylm <- lm(Profit ~., train)
      MSE_train <- mean(residuals(mylm)^2)
      all_MSE_train <- rbind(all_MSE_train, MSE_train)
      
      test_fitted <- predict(mylm, newdata=test[,names(test) != 'Profit'])
      MSE_test <- mean((test$Profit-test_fitted)^2, na.rm = TRUE)
      all_MSE_test <- rbind(all_MSE_test, MSE_test)
    }
    
    final_MSE_train <- rbind(final_MSE_train, mean(all_MSE_train))
    final_MSE_test <- rbind(final_MSE_test, mean(all_MSE_test))
    
  }
  
  training_set_size <- floor(f * nrow(movies_nonnum_lm))
  MSE4 <- data.frame(cbind(training_set_size, final_MSE_train, final_MSE_test))
  colnames(MSE4)[2] <- 'MSE_Train'
  colnames(MSE4)[3] <- 'MSE_Test'
  
  MSE41 <- melt(MSE4, id.vars = 'training_set_size')
  ggplot(MSE41, aes(x=training_set_size, y=value, colour=variable)) + geom_line(size=2) + 
    geom_point(size=4, color='blue', alpha=0.3) +
    ylab(label="MSE") + 
    xlab("Size of Training Set") + coord_cartesian(ylim=c(0,5e+16)) + theme_bw(base_size = 20)
}


plotLRAll <- function(dt) {
  movies <- dt[!is.na(dt$Gross),]
  #add the Profit column
  movies$Profit <- movies$Gross - movies$Budget
  #remove all movies released prior to 2000
  movies <- movies[movies$Year >= 2000,]
  #drop gross, domestic_gross, and boxoffice columns
  movies <- movies[ , !(names(movies) %in% c('Gross', 'Domestic_Gross', 'BoxOffice'))]
  
  ##code for assignment 1
  #keep only the numeric columns
  movies_numeric <- movies[, sapply(movies, is.numeric)]
  #convert Metascore to numeric and add it to movies_numeric
  movies_numeric <- cbind(as.numeric(movies$Metascore), movies_numeric)
  colnames(movies_numeric)[1] <- 'Metascore'
  #since Year and Date columns are almost identical, drop the Date column
  movies_numeric <- movies_numeric[, names(movies_numeric) != 'Date']
  
  X <- movies_numeric[,!grepl("Profit", colnames(movies_numeric))]
  X <- apply(X, 2, as.numeric)
  Y <- as.matrix(movies_numeric$Profit)
  
  #create transformed variables data frame based on highest correlation result
  X <- as.data.frame(X)
  X$Year <- X$Year^3
  X$imdbRating <- X$imdbRating^3
  X$tomatoRating <- X$tomatoRating^3
  X$tomatoReviews <- X$tomatoReviews^3
  X$tomatoFresh <- X$tomatoFresh^2
  X$tomatoRotten <- X$tomatoRotten^3
  X$tomatoUserMeter <- X$tomatoUserMeter^2
  X$tomatoUserRating <- X$tomatoUserRating^3
  X$tomatoUserReviews <- X$tomatoUserReviews^(1/3)
  X$Budget <- X$Budget^2
  colnames(X) <- c('Metascore', 'Year^3', 'imdbRating^3', 'imdbVotes', 'tomatoMeter', 'tomatoRating^3', 'tomatoReviews^3', 'tomatoFresh^2', 'tomatoRotten^3', 'tomatoUserMeter^2', 'tomatoUserRating^3', 'tomatoUserReviews^(1/3)', 'Budget^2')
  
  #based on intuition and the fact that imdbVotes and Budget have the highest correlations with Profit
  #create product of two variables
  X$`imdbVotesBudget^2` <- X$imdbVotes * X$`Budget^2`
  X$`tomatoReviews^3Budget^2` <- X$`tomatoReviews^3` * X$`Budget^2`
  X$`imdbRating^3Budget^2` <- X$`imdbRating^3` * X$`Budget^2`
  X$`tomatoRating^3Budget^2` <- X$`tomatoRating^3` * X$`Budget^2`
  X$`tomatoMeterBudget^2` <- X$tomatoMeter * X$`Budget^2`
  X$`tomatoFresh^2Budget^2` <- X$`tomatoFresh^2` * X$`Budget^2`
  X$`tomatoUserMeter^2Budget^2` <- X$`tomatoUserMeter^2` * X$`Budget^2`
  X$`tomatoUserRating^3Budget^2` <- X$`tomatoUserRating^3` * X$`Budget^2`
  X$`MetascoreBudget^2` <- X$Metascore * X$`Budget^2`
  
  #in the plot of tomatoUserReviews and Profit, a clear separation of data can be seen. so a
  #new variable is_tomatoUserReviews_smaller_than_5M is added which is derived from binning the tomatoUserReviews variable
  X$is_tomatoUserReviews_smaller_than_5M <- ifelse(movies_numeric$tomatoUserReviews < 5e+06, 1, 0)
  
  movies_numeric_transformed <- cbind(movies$Title, X, Y)
  names(movies_numeric_transformed)[1] <- 'Title'
  names(movies_numeric_transformed)[ncol(movies_numeric_transformed)] <- 'Profit'
  
  movies_numeric_transformed <- na.omit(movies_numeric_transformed)
  
  #keep only the non-numeric columns
  movies_nonnum <- movies[, !sapply(movies, is.numeric)]
  movies_nonnum$Profit <- movies$Profit
  #based on observation and intuition, choose the columns that may be useful for predicting Profit
  movies_nonnum <- movies_nonnum[, names(movies_nonnum) %in% c('Title', 'Rated', 'Released','Runtime','Genre','Director', 'Writer', 'Actors', 'Language', 'Country','Awards', 'tomatoImage', 'Production', 'Profit')]
  
  
  #convert a Runtime string into a numeric value in minutes
  #code from project part I
  str_to_num <- function(s){
    if (grepl('h', s) && grepl('min', s)) {
      hour = as.numeric(unlist(strsplit(s, ' '))[1])
      min = as.numeric(unlist(strsplit(s, ' '))[3])
      return(60*hour+min)
    }
    else if (grepl('h', s) && !grepl('min', s)) {
      hour <- as.numeric(unlist(strsplit(s, ' '))[1])
      return(60*hour)
    }
    else if (!grepl('h', s) && grepl('min', s)) {
      min <- as.numeric(unlist(strsplit(s, ' '))[1])
      return(min)
    }
    else {
      return(NA)
    }
    
  }
  movies_nonnum$Runtime = unname(sapply(movies_nonnum$Runtime, str_to_num))
  #end code from project part I
  
  #convert Awards string to total number of wins and nominations
  #code from project part I
  str_to_sum <- function(s){
    if (s == 'N/A') {
      return(NA)
    }
    else {
      l <- unlist(strsplit(s, "[^0-9]+"))
      l <- l[l != '']
      result <- sum(as.numeric(l))
      return(result)
    }
    
  }
  movies_nonnum$Awards <- unname(sapply(movies_nonnum$Awards, str_to_sum))
  #end code from project part I
  
  #convert Released to Released_Month (Released Year information is captured in numeric variables, month information was shown
  #to be related to Gross in project part I)
  movies_nonnum$Released <- month(movies_nonnum$Released)
  colnames(movies_nonnum)[3] <- 'Released_Month'
  
  #parse each text string in Genre into a binary vector
  #code from project part I
  movies_nonnum <- movies_nonnum[movies_nonnum$Genre != 'N/A',]
  movies_nonnum$Genre <- strsplit(movies_nonnum$Genre, ', ')
  movies_long <- unnest(movies_nonnum, Genre)
  movies_long$Genre <- paste0("Genre_", gsub("\\s","_",movies_long$Genre))
  movies_wide <- dcast(movies_long, Title ~ Genre, function(x) 1, fill = 0)
  movies_nonnum <- merge(movies_nonnum, movies_wide)
  #end code from project part I
  
  #parse each text string in Director into a binary vector
  movies_nonnum <- movies_nonnum[movies_nonnum$Director != 'N/A',]
  movies_nonnum$Director <- strsplit(movies_nonnum$Director, "(\\s)?,(\\s)?")
  movies_long1 <- unnest(movies_nonnum, Director)
  movies_long1$Director <- paste0("Director_", gsub("\\s","_",movies_long1$Director))
  movies_wide1 <- dcast(movies_long1, Title ~ Director, function(x) 1, fill = 0)
  number_of_directors <- rowSums(movies_wide1[,-1])
  
  
  #consider the top 123 directors where count of titles >= 5
  count$Director <- as.character(count$Director)
  movies_wide1_top <- movies_wide1[,c('Title',count$Director[1:123])]
  #add a 124th variable,Director_others, if there is no other directors, Director_others equals to 0, else, it equals to 1
  movies_wide1_top$number_of_top_directors <- rowSums(movies_wide1_top[,-1])
  movies_wide1_top <- cbind(movies_wide1_top, number_of_directors)
  movies_wide1_top$Director_others <- ifelse((movies_wide1_top$number_of_directors-movies_wide1_top$number_of_top_directors) == 0, 0, 1)
  movies_wide1_top <- movies_wide1_top[, !colnames(movies_wide1_top) %in% c('number_of_directors','number_of_top_directors')]
  
  movies_nonnum <- merge(movies_nonnum, movies_wide1_top)
  
  #parse each text string in Actor into a binary vector
  movies_nonnum <- movies_nonnum[movies_nonnum$Actors != 'N/A',]
  movies_nonnum$Actors <- strsplit(movies_nonnum$Actors, "(\\s)?,(\\s)?")
  movies_long2 <- unnest(movies_nonnum, Actors)
  movies_long2$Actors <- paste0("Actor_", gsub("\\s","_",movies_long2$Actors))
  movies_wide2 <- dcast(movies_long2, Title ~ Actors, function(x) 1, fill = 0)
  number_of_actors <- rowSums(movies_wide2[,-1])
  
  #consider the top 167 actors where count of titles > 10
  count1$Actors <- as.character(count1$Actors)
  movies_wide2_top <- movies_wide2[,c('Title',count1$Actors[1:167])]
  #add a 168th variable,Actor_others, if there is no other actors, Actor_others equals to 0, else, it equals to 1
  movies_wide2_top$number_of_top_actors <- rowSums(movies_wide2_top[,-1])
  movies_wide2_top <- cbind(movies_wide2_top, number_of_actors)
  movies_wide2_top$Actor_others <- ifelse((movies_wide2_top$number_of_actors-movies_wide2_top$number_of_top_actors) == 0, 0, 1)
  movies_wide2_top <- movies_wide2_top[, !colnames(movies_wide2_top) %in% c('number_of_actors','number_of_top_actors')]
  
  movies_nonnum <- merge(movies_nonnum, movies_wide2_top)
  
  #parse each text string in Writer into a binary vector
  movies_nonnum <- movies_nonnum[movies_nonnum$Writer != 'N/A',]
  movies_nonnum$Writer <- strsplit(movies_nonnum$Writer, "(\\s)?,(\\s)?")
  movies_long3 <- unnest(movies_nonnum, Writer)
  movies_long3$Writer <- gsub("\\s*\\([^\\)]+\\)","",movies_long3$Writer)
  movies_long3$Writer <- paste0("Writer_", gsub("\\s","_",movies_long3$Writer))
  movies_wide3 <- dcast(movies_long3, Title ~ Writer, function(x) 1, fill = 0)
  number_of_writers <- rowSums(movies_wide3[,-1])
  
  #consider the top 129 writers where count of titles > 5
  count2$Writer <- as.character(count2$Writer)
  movies_wide3_top <- movies_wide3[,c('Title',count2$Writer[1:129])]
  
  #add a 130th variable,Writer_others, if there is no other writers, Writer_others equals to 0, else, it equals to 1
  movies_wide3_top$number_of_top_writers <- rowSums(movies_wide3_top[,-1])
  movies_wide3_top <- cbind(movies_wide3_top, number_of_writers)
  movies_wide3_top$Writer_others <- ifelse((movies_wide3_top$number_of_writers-movies_wide3_top$number_of_top_writers) == 0, 0, 1)
  movies_wide3_top <- movies_wide3_top[, !colnames(movies_wide3_top) %in% c('number_of_writers','number_of_top_writers')]
  
  movies_nonnum <- merge(movies_nonnum, movies_wide3_top)
  
  #convert Language to number of languages
  movies_nonnum$Language <- strsplit(movies_nonnum$Language, ', ')
  movies_nonnum$Language <- sapply(movies_nonnum$Language, length)
  
  #convert Country to number of countries
  movies_nonnum$Country <- strsplit(movies_nonnum$Country, ', ')
  movies_nonnum$Country <- sapply(movies_nonnum$Country, length)
  
  #convert Rated into a binary vector
  movies_nonnum <- movies_nonnum[movies_nonnum$Rated != 'N/A',]
  convert <- function(s) {
    if(s == 'UNRATED') {
      s <- 'NOT RATED'
      return(s)
    }
    else {
      return(s)
    }
  }
  movies_nonnum$Rated <- sapply(movies_nonnum$Rated, convert)
  movies_nonnum_wide <- dcast(movies_nonnum, Title ~ Rated, function(x) 1, fill = 0)
  movies_nonnum <- merge(movies_nonnum, movies_nonnum_wide)
  
  #convert tomatoImage into a binary vector
  movies_nonnum <- movies_nonnum[movies_nonnum$tomatoImage != 'N/A',]
  movies_nonnum_wide1 <- dcast(movies_nonnum, Title ~ tomatoImage, function(x) 1, fill = 0)
  movies_nonnum <- merge(movies_nonnum, movies_nonnum_wide1)
  
  #convert Production into a binary vector
  movies_nonnum <- movies_nonnum[movies_nonnum$Production != 'N/A',]
  movies_wide4 <- dcast(movies_nonnum, Title ~ Production, function(x) 1, fill = 0)
  
  
  #consider the top 49 productions where count of titles >= 10
  count3$Production <- as.character(count3$Production)
  movies_wide4_top <- movies_wide4[,c('Title',count3$Production[1:49])]
  
  #add a 50th variable,Production_others, if production is not in top productions, Writer_others equals to 1, else, it equals to 0
  movies_wide4_top$Production_others <- ifelse(movies_nonnum$Production %in% count3$Production[1:49], 0, 1)
  
  movies_nonnum <- merge(movies_nonnum, movies_wide4_top)
  
  #drop the Genre, Director, and Actors columns
  movies_nonnum <- movies_nonnum[,!colnames(movies_nonnum) %in% c('Genre','Director','Actors','Rated','Writer','tomatoImage','Production')]
  
  movies_nonnum <- movies_nonnum[, names(movies_nonnum) != 'Profit']
  movies_all <- merge(movies_numeric_transformed, movies_nonnum)
  movies_all <- na.omit(movies_all)
  
  #in project part I, it was found that movies with budget greater than 1e+08 are mostly
  #Action, Adventure, and Comedy. create interaction features based on this insight
  movies_all$`is_budget_greater_than_1e+08` <- ifelse(movies_all$`Budget^2` > 1e+08, 1, 0)
  movies_all$`is_budget_greater_than_1e+08*Genre_Action` <- movies_all$`is_budget_greater_than_1e+08` * movies_all$Genre_Action
  movies_all$`is_budget_greater_than_1e+08*Genre_Adventure` <- movies_all$`is_budget_greater_than_1e+08` * movies_all$Genre_Adventure
  movies_all$`is_budget_greater_than_1e+08*Genre_Comedy` <- movies_all$`is_budget_greater_than_1e+08` * movies_all$Genre_Comedy
  
  #explore the correlations between transformed variables and profit
  high_cor(movies_all$Released_Month, movies_all$Profit)
  high_cor(movies_all$Runtime, movies_all$Profit)
  high_cor(movies_all$Language, movies_all$Profit)
  high_cor(movies_all$Country, movies_all$Profit)
  high_cor(movies_all$Awards, movies_all$Profit)
  #based on the correlation results, create new transformed variables
  movies_all$`log(Released_Month)` <- log(movies_all$Released_Month)
  movies_all$`Runtime^3` <- movies_all$Runtime^3
  movies_all$`log(Language)` <- log(movies_all$Language)
  movies_all$`Country^2` <- movies_all$Country^2
  movies_all$`Awards^(1/3)` <- movies_all$Awards^(1/3)
  movies_all <- movies_all[!names(movies_all) %in% c("Released_Month", "Runtime","Language","Country","Awards")]
  
  X_all <- Matrix(as.matrix(movies_all[!names(movies_all) %in% c("Title", "Profit")]), sparse = TRUE)
  Y_all <- movies_all$Profit
  
  final_MSE_train <- NULL
  final_MSE_test <- NULL
  
  f <- seq(0.05, 0.95, by = 0.05)
  for (fraction in f) {
    #divide data into training and test sets
    smp_size <- floor(fraction * nrow(movies_all))
    
    all_MSE_train <- NULL
    all_MSE_test <- NULL
    
    #repeat the random partition of dataset 10 times
    for (n in c(29, 35, 67, 85, 102, 219, 175, 199, 143, 139)) {
      set.seed(n)
      train_ind <- sample(seq_len(nrow(movies_all)), size = smp_size)
      X_train <- X_all[train_ind,]
      X_test <- X_all[-train_ind,]
      Y_train <- Y_all[train_ind]
      Y_test <- Y_all[-train_ind]
      
      #use LASSO to select from a large number of variables  
      cvfit <- cv.glmnet(X_train,  
                         Y_train, 
                         family = "gaussian",   ## linear regression
                         alpha = 1,   ## select Lasso
                         type.measure = "mse",  ## train to minimize mse
                         nfolds = 5)   ## 5-folds cross-validation
      
      train_fitted <- predict(cvfit, newx = X_train, s = "lambda.min")
      MSE_train <- mean((train_fitted-Y_train)^2)
      all_MSE_train <- rbind(all_MSE_train, MSE_train)
      
      test_fitted <- predict(cvfit, newx = X_test, s = "lambda.min")
      MSE_test <- mean((Y_test-test_fitted)^2)
      all_MSE_test <- rbind(all_MSE_test, MSE_test)
    }
    
    final_MSE_train <- rbind(final_MSE_train, mean(all_MSE_train))
    final_MSE_test <- rbind(final_MSE_test, mean(all_MSE_test))
    
  }
  
  training_set_size <- floor(f * nrow(movies_all))
  MSE <- data.frame(cbind(training_set_size, final_MSE_train, final_MSE_test))
  colnames(MSE)[2] <- 'MSE_Train_q5'
  colnames(MSE)[3] <- 'MSE_Test_q5'
  MSE$training_set_size <- seq(0.05, 0.95, by = 0.05)
  colnames(MSE)[1] <- 'Percentage of Training Set Size'
  
  #combine all the MSE results from previous questions, and make comparison in one plot
  colnames(MSE1)[2] <- 'MSE_Train_q1'
  colnames(MSE1)[3] <- 'MSE_Test_q1'
  
  colnames(MSE2)[2] <- 'MSE_Train_q2'
  colnames(MSE2)[3] <- 'MSE_Test_q2'
  
  colnames(MSE4)[2] <- 'MSE_Train_q4'
  colnames(MSE4)[3] <- 'MSE_Test_q4'
  
  MSE_all <- cbind(MSE,MSE1[-1], MSE2[-1], MSE4[-1])
  
  MSE_all_1 <- melt(MSE_all, id.vars = 'Percentage of Training Set Size')
  ggplot(MSE_all_1, aes(x=`Percentage of Training Set Size`, y=value, colour=variable)) + geom_line(size=2) + 
    geom_point(size=4, color='blue', alpha=0.3) +
    ylab(label="MSE") + 
    xlab("Percentage of Training Set") + coord_cartesian(ylim=c(0,5e+16)) + theme_bw(base_size = 20)
  
}

