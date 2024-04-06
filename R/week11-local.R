#### Script Settings and Resources ####
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
library(caret)
set.seed(8712)

#### Data Import and Cleaning #### 
# Import SPSS data (Road haven package)
gss_original_tbl <- read_sav(file = "../data/GSS2016.sav")

# Create a variable gss_tbl
gss_tbl <- gss_original_tbl %>%
  # Remove rows where MOSTHRS is missing by using filter and !is.na
  filter(!is.na(MOSTHRS)) %>%
  # Rename MOSTHRS to workhours by using rename
  rename(workhours = MOSTHRS) %>%
  # Remove HRS1 and HRS2 variables by using select 
  select(-c(HRS1, HRS2)) %>%
  # Retain only variables with less than 75% missingness by using select
  select(where(~mean(is.na(.)) < 0.75)) %>%
  # Convert to data to numeric variables
  sapply(as.numeric) %>% 
  # Convert to data frame to ensure compatibility with dplyr functions by using as.data.frame
  as_tibble()  

#### Visualization #### 
# Making the histogram
ggplot(gss_tbl, aes(x = workhours)) +
  geom_histogram() +
  labs(x = "Working hours",
       y = "Number of data",
       title = "Histogram of working hours")

#### Analysis #### 
## Prepare data set to run ML
# Create sample rows  
gss_sample <- sample(nrow(gss_tbl))
# Shuffle using the sampled indices 
gss_shuffled <- gss_tbl[gss_sample,]
# Calculate the number of rows for a 75/25 split
gss_75per <- round(nrow(gss_shuffled) * 0.75)
# Create the training set using the first 75% of the shuffled data
gss_train <- gss_shuffled[1:gss_75per,]
# Create the test set using the remaining 25% of the shuffled data
gss_test <- gss_shuffled[(gss_75per + 1):nrow(gss_shuffled), ]
# Create 10 folds for cross-validation using the workhours column from the training set
gss_folds <- createFolds(gss_train$workhours, 10)
# Set up train control for all model
train_control <- trainControl(method = "cv", 
                              number = 10, 
                              index = gss_folds, 
                              verboseIter = TRUE)

## Run OLS,  elastic net, random forest, and eXtreme Gradient Boosting
# Train the OLS regression model using train
model_OLS <- train(
  workhours ~ .,
  data = gss_train,
  method = "lm", 
  metric = "Rsquared",
  preProcess = "medianImpute",
  na.action = na.pass,
  trControl = train_control
  )

OLS_predict <- predict(model_OLS, gss_test, na.action = na.pass)

# Train the elastic net model using train
model_elastic <- train(
  workhours ~ .,
  data = gss_train,
  method = "glmnet", 
  metric = "Rsquared",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = train_control
  )

# Train the random forest model using train
model_random <- train(
  workhours ~ .,
  data = gss_train,
  method = "ranger", 
  metric = "Rsquared",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = train_control
)

# Train the random XGB using train
model_XGB <- train(
  workhours ~ .,
  data = gss_train,
  method = "xgbLinear", 
  metric = "Rsquared",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = train_control
)

#### Publication #### 
# Make predictions using the models
OLS_predict <- predict(model_OLS, gss_test, na.action = na.pass)
elastic_predict <- predict(model_elastic, gss_test, na.action = na.pass)
random_predict <- predict(model_random, gss_test, na.action = na.pass)
XGB_predict <- predict(model_XGB, gss_test, na.action = na.pass)

# Calculate R-squared values for holdout CV
ho_rsq_OLS <- cor(OLS_predict, gss_test$workhours)^2
ho_rsq_elastic <- cor(elastic_predict, gss_test$workhours)^2
ho_rsq_random <- cor(random_predict, gss_test$workhours)^2
ho_rsq_XGB <- cor(XGB_predict, gss_test$workhours)^2

# Create a tibble with the desired structure
table1_tbl <- tibble(
  algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
  cv_rsq = c(
    str_remove(format(round(model_OLS$results$Rsquared, 2), nsmall = 2), pattern = "^0"),
    str_remove(format(round(max(model_elastic$results$Rsquared), 2), nsmall = 2), pattern = "^0"),
    str_remove(format(round(max(model_random$results$Rsquared), 2), nsmall = 2), pattern = "^0"),
    str_remove(format(round(max(model_XGB$results$Rsquared), 2), nsmall = 2), pattern = "^0")),
  ho_rsq = c(
    str_remove(format(round(ho_rsq_OLS, 2), nsmall = 2), pattern = "^0"),
    str_remove(format(round(ho_rsq_elastic, 2), nsmall = 2), pattern = "^0"),
    str_remove(format(round(ho_rsq_random, 2), nsmall = 2), pattern = "^0"),
    str_remove(format(round(ho_rsq_XGB, 2), nsmall = 2), pattern = "^0")
  )
)

# Print the table
print(table1_tbl)

# A1. 
# The results, measured in R-squared values, varied significantly across the different models. 
# The OLS Regression model showed the lowest performance with an R-squared value of 0.00, indicating a poor fit to the data. 
# Conversely, Elastic Net, Random Forest, and eXtreme Gradient Boosting (XGB) models performed relatively better, 
# with R-squared values of 0.49, 0.40, and 0.32, respectively.
# I believe this variation in results can be attributed to the differences in how each model handles the data 
# and captures the underlying relationships between variables. 

# A2.
# The k-fold cross-validation (CV) R-squared values were generally lower than the holdout CV R-squared values. 
# This is because k-fold CV divides the data into multiple folds for training and testing, 
# which tends to give a more conservative estimate of how well the model performs.

# A3.
# For real-life predictions, I'd choose the Random Forest model 
# because it had the best prediction accuracy among the models tested. 
# It's good with big datasets and can handle complex relationships well. 
# However, it might overfit and take longer to train with large amounts of data.
