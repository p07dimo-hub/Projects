# Required Libraries
library(shiny)
library(ggplot2)
library(plotly)
library(DT)
library(caret)
library(randomForest)
library(class)
library(pROC)
library(umap)
library(shinythemes)


ui <- fluidPage(
  theme = shinytheme("flatly"),
  
  titlePanel("Data Mining & Analysis Application"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload your CSV, Excel, .data or .name file",
                accept = c(".csv", ".xlsx", ".data", ".name")),
      
      
      numericInput("num_features", "Select Number of Features:", value = 5, min = 1),
      
      selectInput("algorithm", "Choose Classification Algorithm:",
                  choices = c("K-Nearest Neighbors", "Random Forest")),
      
      sliderInput("test_size", "Test Size:", min = 0.1, max = 0.5, value = 0.2),
      
      numericInput("knn_k", "K value for KNN:", value = 5, min = 1),
      numericInput("rf_trees", "Number of Trees for Random Forest:", value = 100, min = 10)
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Data Preview", DTOutput("data_table")),
        tabPanel("2D & 3D Visualizations",
                 plotlyOutput("pca_plot"),
                 plotlyOutput("umap_plot")),
        tabPanel("Feature Selection", DTOutput("selected_features")),
        tabPanel("Classification Results", 
                 verbatimTextOutput("results_original"),
                 verbatimTextOutput("results_reduced")),
        tabPanel("Info", 
                 h3("About this Application"),
                 p("This application helps users to perform data analysis and machine learning."))
      )
    )
  )
)

server <- function(input, output) {
  
  dataset <- reactive({
    req(input$file)
    ext <- tools::file_ext(input$file$name)
    
   
    if (ext == "csv") {
      data <- read.csv(input$file$datapath)
    } else if (ext == "xlsx") {
      data <- readxl::read_excel(input$file$datapath)
    } else if (ext == "data" || ext == "name") {
      data <- read.table(input$file$datapath, header = FALSE, sep = ",")
    }
    
    return(data)
  })
  
  output$data_table <- renderDT({
    datatable(dataset())
  })
  
  
  output$pca_plot <- renderPlotly({
    data <- dataset()
    pca <- prcomp(data[, -ncol(data)], scale. = TRUE)
    pca_2d <- data.frame(pca$x[, 1:2], label = data[, ncol(data)])
    plot_ly(pca_2d, x = ~PC1, y = ~PC2, color = ~label, type = "scatter", mode = "markers")
  })
  
  output$umap_plot <- renderPlotly({
    data <- dataset()
    umap_3d <- umap(data[, -ncol(data)], n_components = 3)
    umap_3d_df <- data.frame(umap_3d$layout, label = data[, ncol(data)])
    plot_ly(umap_3d_df, x = ~X1, y = ~X2, z = ~X3, color = ~label, type = "scatter3d", mode = "markers")
  })
  
  output$selected_features <- renderDT({
    data <- dataset()
    control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
    results <- rfe(data[, -ncol(data)], data[, ncol(data)], sizes = input$num_features, rfeControl = control)
    selected_data <- data[, predictors(results)]
    
    datatable(selected_data)
  })
  
  output$results_original <- renderPrint({
    data <- dataset()
    split <- createDataPartition(data[, ncol(data)], p = input$test_size, list = FALSE)
    train_data <- data[split, ]
    test_data <- data[-split, ]
    
    if (input$algorithm == "K-Nearest Neighbors") {
      knn_model <- knn(train_data[, -ncol(train_data)], test_data[, -ncol(test_data)],
                       train_data[, ncol(train_data)], k = input$knn_k)
      acc <- sum(knn_model == test_data[, ncol(test_data)]) / nrow(test_data)
      cat("KNN Accuracy on Original Data:", acc, "\n")
    } else if (input$algorithm == "Random Forest") {
      rf_model <- randomForest(train_data[, -ncol(train_data)], train_data[, ncol(train_data)],
                               ntree = input$rf_trees)
      pred <- predict(rf_model, test_data[, -ncol(test_data)])
      acc <- sum(pred == test_data[, ncol(test_data)]) / nrow(test_data)
      cat("Random Forest Accuracy on Original Data:", acc, "\n")
    }
  })
  
  output$results_reduced <- renderPrint({
    data <- dataset()
    control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
    results <- rfe(data[, -ncol(data)], data[, ncol(data)], sizes = input$num_features, rfeControl = control)
    reduced_data <- data[, c(predictors(results), ncol(data))]
    
    split <- createDataPartition(reduced_data[, ncol(reduced_data)], p = input$test_size, list = FALSE)
    train_data <- reduced_data[split, ]
    test_data <- reduced_data[-split, ]
    
    if (input$algorithm == "K-Nearest Neighbors") {
      knn_model <- knn(train_data[, -ncol(train_data)], test_data[, -ncol(test_data)],
                       train_data[, ncol(train_data)], k = input$knn_k)
      acc <- sum(knn_model == test_data[, ncol(test_data)]) / nrow(test_data)
      cat("KNN Accuracy on Reduced Data:", acc, "\n")
    } else if (input$algorithm == "Random Forest") {
      rf_model <- randomForest(train_data[, -ncol(train_data)], train_data[, ncol(train_data)],
                               ntree = input$rf_trees)
      pred <- predict(rf_model, test_data[, -ncol(test_data)])
      acc <- sum(pred == test_data[, ncol(test_data)]) / nrow(test_data)
      cat("Random Forest Accuracy on Reduced Data:", acc, "\n")
    }
  })
}

shinyApp(ui = ui, server = server)
