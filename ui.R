# The user-interface definition of the Shiny web app.
library(shiny)
library(shinyjs)
library(BH)
library(rCharts)
require(markdown)
require(data.table)
library(dplyr)
library(DT)


shinyUI(
  tagList (
    useShinyjs(),
    navbarPage("IMDb Movie Dataset Visualizer", id = 'imdb',
    # multi-page user-interface that includes a navigation bar.
        tabPanel("Explore the Data", 
             sidebarPanel(
                sliderInput("timeline", 
                            "Timeline:", 
                            min = 1888,
                            max = 2018,
                            value = c(1888, 2018)),
                            #format = "####"),
                
                conditionalPanel(condition="input.tabselected==2",
                                 selectInput('type', 'Choose Visualizer Type:',
                                             c('', 'Single Variable Distribution', 'Relationship Among Variables')),
                                 conditionalPanel(
                                   condition = "input.type == 'Single Variable Distribution'",
                                   selectInput("histo", "Choose a Feature:",
                                               c('', 'Runtime', 'Number of Wins or Nominations', 'Year'))
                                 ),
                                 
                                 conditionalPanel(
                                   condition = "input.type == 'Relationship Among Variables'",
                                   selectInput("relation", "Choose features:",
                                               c('', 'Runtime and Year', 'Runtime and Budget', 'Title Count and Genres',
                                                 'Gross Revenue and Genres', 'Gross Revenue and Month', 'Gross Revenue, Month, and Genres',
                                                 'IMDb Votes and IMDb Rating', 'tomatoRating and tomatoReviews', 'Gross Revenue and Award',
                                                 'Pairwise correlation between gross revenue and different ratings', "Profit (Gross Revenue - Budget)"))
                                 )
                                 
                )
             ),
             mainPanel(
                 tabsetPanel(
                   # Data 
                   tabPanel("Dataset", value=1,
                            dataTableOutput(outputId="dTable")
                   ), # end of "Dataset" tab panel
                   tabPanel("Visualize the Data", value=2,
                            conditionalPanel(
                              condition = "input.type == 'Single Variable Distribution' && input.histo == 'Runtime'",
                              h4('Distribution of Movie Runtime', align = "center"),
                              plotOutput("Runtime")
                            ),
                            
                            conditionalPanel(
                              condition = "input.type == 'Single Variable Distribution' && input.histo == 'Number of Wins or Nominations'",
                              h4('Distribution of Number of Wins or Nominations', align = "center"),
                              plotOutput("NoOfWins")
                            ),
                            
                            conditionalPanel(
                              condition = "input.type == 'Single Variable Distribution' && input.histo == 'Year'",
                              h4('Number of Movies Across Decades', align = "center"),
                              plotOutput("Year")
                            ),
                            
                            # conditionalPanel(
                            #   condition = "input.type == 'Single Variable Distribution' && input.histo == 'Director'",
                            #   h4('Top 50 Directors', align = "center"),
                            #   plotOutput("Director", height = "800px")
                            # ),
                            # 
                            # conditionalPanel(
                            #   condition = "input.type == 'Single Variable Distribution' && input.histo == 'Actor'",
                            #   h4('Top 50 Actors', align = "center"),
                            #   plotOutput("Actor", height = "800px")
                            # ),
                            # 
                            conditionalPanel(
                              condition = "input.type == 'Relationship Among Variables' && input.relation == 'Runtime and Year'",
                              h4('Movie Runtime vs. Released Year', align = "center"),
                              plotOutput("RuntimeAndYear")
                            ),
                            
                            conditionalPanel(
                              condition = "input.type == 'Relationship Among Variables' && input.relation == 'Runtime and Budget'",
                              h4('Movie Runtime vs. Budget', align = "center"),
                              plotOutput("RuntimeAndBudget")
                            ),
                            
                            conditionalPanel(
                              condition = "input.type == 'Relationship Among Variables' && input.relation == 'Title Count and Genres'",
                              h4('Movie Counts Across Genres', align = "center"),
                              plotOutput("TitleAndGenres")
                            ),
                            
                            conditionalPanel(
                              condition = "input.type == 'Relationship Among Variables' && input.relation == 'Gross Revenue and Genres'",
                              h4('Movie Gross Revenue Across Top 10 Genres', align = "center"),
                              plotOutput("GrossAndGenres")
                            ),
                            
                            conditionalPanel(
                              condition = "input.type == 'Relationship Among Variables' && input.relation == 'Gross Revenue and Month'",
                              h4('Movie Gross Revenue vs. Released Month', align = "center"),
                              plotOutput("GrossAndMonth")
                            ),
                            
                            conditionalPanel(
                              condition = "input.type == 'Relationship Among Variables' && input.relation == 'Gross Revenue, Month, and Genres'",
                              h4('Movie Gross Revenue vs. Released Month Across Genres', align = "center"),
                              plotOutput("GrossAndMonthAndGenres", height = "3000px")
                            ),
                            
                            conditionalPanel(
                              condition = "input.type == 'Relationship Among Variables' && input.relation == 'IMDb Votes and IMDb Rating'",
                              h4('IMDb Votes vs. IMDb Rating', align = "center"),
                              plotOutput("ImdbVotesAndRating")
                            ),
                            
                            conditionalPanel(
                              condition = "input.type == 'Relationship Among Variables' && input.relation == 'tomatoRating and tomatoReviews'",
                              h4('Rotten Tomato Rating vs. Reviews', align = "center"),
                              plotOutput("TomatoRatingAndReviews")
                            ),
                            
                            conditionalPanel(
                              condition = "input.type == 'Relationship Among Variables' && input.relation == 'Gross Revenue and Award'",
                              h4('Gross Revenue vs. Award', align = "center"),
                              plotOutput("GrossAndAward")
                            ),
                            
                            conditionalPanel(
                              condition = "input.type == 'Relationship Among Variables' && input.relation == 'Pairwise correlation between gross revenue and different ratings'",
                              h4('Pairwise Correlation Between Gross Revenue and Different Ratings', align = "center"),
                              plotOutput("PairwiseCor")
                            ),
                            
                            conditionalPanel(
                              condition = "input.type == 'Relationship Among Variables' && input.relation == 'Profit (Gross Revenue - Budget)'",
                              h4('Relationships between Profit and 13 Numeric Variables', align = "center"),
                              plotOutput("Profit", height = "800px", width = '1500px')
                            )
                            
                   ), # end of "Visualize the Data" tab panel
                   id = "tabselected"

                 )
                   
            )     
        ), # end of "Explore Dataset" tab panel
    
    tabPanel('Predictive Modeling', 
             sidebarPanel(selectInput('pm', 'Choose a Predictive Model:',
                                       c('', 'Linear Regression')),
                          conditionalPanel(
                            condition = "input.pm == 'Linear Regression'",
                            selectInput("feature", "Choose Feature Type:",
                                        c('', 'Numeric Features', 'Transformed Numeric Features'))
                          )
                           ),
             mainPanel(
               conditionalPanel(
                 condition = "input.pm == 'Linear Regression' && input.feature == 'Numeric Features'",
                 h4('MSE as a Function of Training Set Size (Numeric Variables Only)', align = 'center'),
                 plotOutput("LRNumeric")
               ),
               
               conditionalPanel(
                 condition = "input.pm == 'Linear Regression' && input.feature == 'Transformed Numeric Features'",
                 h4('MSE as a Function of Training Set Size (Transformed Numeric Variables)', align = 'center'),
                 plotOutput("LRTrans")
               )
               
               # conditionalPanel(
               #   condition = "input.pm == 'Linear Regression' && input.feature == 'Categorical Features'",
               #   h4('MSE as a Function of Training Set Size (Non-Numeric Variables Only)', align = 'center'),
               #   plotOutput("LRCate")
               # ),
               # 
               # conditionalPanel(
               #   condition = "input.pm == 'Linear Regression' && input.feature == 'All Features'",
               #   h4('MSE as a Function of Training Set Size', align = 'center'),
               #   plotOutput("LRAll")
               # )
             
             )
             )
          
    )  
    )  
)
