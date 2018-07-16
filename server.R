library(shiny)

# Load data processing file
source("test.R")
themes <- sort(unique(data$theme))

# Shiny server
shinyServer(
  function(input, output, session) {
    
    # Initialize reactive values
    values <- reactiveValues()
    values$themes <- themes
    
    #reset pages after clicking other tabs
    observeEvent(input$tabselected, {
      shinyjs::reset("type")
      shinyjs::reset("histo")
      shinyjs::reset("relation")
    })
    
    observeEvent(input$imdb, {
      shinyjs::reset("type")
      shinyjs::reset("histo")
      shinyjs::reset("relation")
    })    
    
    observeEvent(input$imdb, {
      shinyjs::reset("pm")
      shinyjs::reset("feature")
    })    
    
    # Prepare dataset
    dataTable <- reactive({
      groupByYear(data, input$timeline[1], 
                   input$timeline[2])
    })
    
    
    # Render data table
    output$dTable <- renderDataTable({
      dataTable()
    } #, options = list(bFilter = FALSE, iDisplayLength = 50)
    )
    
    output$Runtime <- renderPlot({
      plotRuntime(dataTable())
    })
    
    output$NoOfWins <- renderPlot({
      plotNoOfWins(dataTable())
    })
    
    output$RuntimeAndYear <- renderPlot({
      plotRuntimeAndYear(dataTable())
    })
    
    output$RuntimeAndBudget <- renderPlot({
      plotRuntimeAndBudget(dataTable())
    })
    
    output$TitleAndGenres <- renderPlot({
      plotTitleAndGenres(dataTable())
    })
    
    output$GrossAndGenres <- renderPlot({
      plotGrossAndGenres(dataTable())
    })
    
    output$GrossAndMonth <- renderPlot({
      plotGrossAndMonth(dataTable())
    })
    
    output$GrossAndMonthAndGenres <- renderPlot({
      plotGrossAndMonthAndGenres(dataTable())
    })
    
    output$ImdbVotesAndRating <- renderPlot({
      plotImdbVotesAndRating(dataTable())
    })
    
    output$TomatoRatingAndReviews <- renderPlot({
      plotTomatoRatingAndReviews(dataTable())
    })
    
    output$GrossAndAward <- renderPlot({
      plotGrossAndAward(dataTable())
    })
    
    output$Year <- renderPlot({
      plotYear(dataTable())
    })
    
    output$Director <- renderPlot({
      plotDirector(dataTable())
    })
    
    output$Actor <- renderPlot({
      plotActor(dataTable())
    })
    
    output$PairwiseCor <- renderPlot({
      plotPairwiseCor(dataTable())
    })
    
    output$Profit <- renderPlot({
      plotProfit(dataTable())
    })
    
    output$LRNumeric <- renderPlot({
      plotLRNumeric(dataTable())
    })
    
    output$LRTrans <- renderPlot({
      plotLRTrans(dataTable())
    })
    
    output$LRCate <- renderPlot({
      plotLRCate(dataTable())
    })
    
    output$LRAll <- renderPlot({
      plotLRAll(dataTable())
    })
    
    
 
    
  } # end of function(input, output)
)