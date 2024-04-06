#### Script Settings and Resources ####
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
library(caret)

install.packages("parallel")
install.packages("doParallel")
install.packages("tictoc")

library(parallel)
library(doParallel)
library(tictoc)
set.seed(8712)

#### Data Import and Cleaning #### 
gss_original_tbl <- read_sav(file = "../data/GSS2016.sav")

# Create a variable gss_tbl
gss_tbl <- gss_original_tbl %>%
  filter(!is.na(MOSTHRS)) %>%
  rename(workhours = MOSTHRS) %>%
  select(-c(HRS1, HRS2)) %>%
  select(where(~mean(is.na(.)) < 0.75)) %>%
  sapply(as.numeric) %>% 
  as_tibble()  

#### Visualization #### 
ggplot(gss_tbl, aes(x = workhours)) +
  geom_histogram() +
  labs(x = "Working hours",
       y = "Number of data",
       title = "Histogram of working hours")

#### Analysis #### 
# I fixed the week 10 based on debrief video and code to improve Project 11. I added comments to indicate changes made.
# Create holdout indices for data partitioning
holdout_indices <- createDataPartition(gss_tbl$workhours,
                                       p = .25,
                                       list = T)$Resample1
# Create a test set using the holdout indices
test_tbl <- gss_tbl[holdout_indices,]
# Create a training set by excluding the holdout indices
training_tbl <- gss_tbl[-holdout_indices,]
# Create cross-validation folds for the training data
training_folds <- createFolds(training_tbl$workhours)

## OLS Model
# Train the OLS regression model using train function
model_OLS <- train(
  workhours ~ .,
  training_tbl,
  method="lm",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = training_folds)
)
# Extracting cross-validation results (R-squared) from the OLS model
cv_OLS <- model_OLS$results$Rsquared
# Evaluating the OLS model on the holdout test set
holdout_OLS <- cor(
  predict(model_OLS, test_tbl, na.action = na.pass),
  test_tbl$workhours
)^2

## Elastic net model
# Train the elastic net model using train
model_elastic <- train(
  workhours ~ .,
  training_tbl,
  method="glmnet",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = training_folds)
)
# Extracting cross-validation results (R-squared) from the Elastic net model
cv_elastic <- max(model_elastic$results$Rsquared)
# Evaluating the Elastic net model on the holdout test set
holdout_elastic <- cor(
  predict(model_elastic, test_tbl, na.action = na.pass),
  test_tbl$workhours
)^2

## Random forest model
# Train the random forest model using train
model_random <- train(
  workhours ~ .,
  training_tbl,
  method="ranger",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = training_folds)
)
# Extracting cross-validation results (R-squared) from the random forest model
cv_random <- max(model_random$results$Rsquared)
# Evaluating the random forest model on the holdout test set
holdout_random <- cor(
  predict(model_random, test_tbl, na.action = na.pass),
  test_tbl$workhours
)^2

## Random XGB
# Train the random XGB using train
model_XGB <- train(
  workhours ~ .,
  training_tbl,
  method="xgbLinear",
  na.action = na.pass,
  tuneLength = 1,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = training_folds)
)
# Extracting cross-validation results (R-squared) from the random XGB
cv_XGB <- max(model_XGB$results$Rsquared)
# Evaluating the random XGB on the holdout test set
holdout_XGB <- cor(
  predict(model_XGB, test_tbl, na.action = na.pass),
  test_tbl$workhours
)^2

## Summarize and Visualization
# Calculate the summary of resampling results for multiple models
summary(resamples(list(model_OLS, model_elastic, model_random, model_XGB)), metric="Rsquared")
# Visualize the resampling results for multiple models using a dotplot
dotplot(resamples(list(model_OLS, model_elastic, model_random, model_XGB)), metric="Rsquared")




#### Publication #### 
# Create formats numerical values to make them look visually appealing
make_it_pretty <- function (formatme) {
  formatme <- formatC(formatme, format="f", digits=2)
  formatme <- str_remove(formatme, "^0")
  return(formatme)
}

# Create a tibble to store algorithm names, cross-validation R-squared, and holdout R-squared values
table1_tbl <- tibble(
  algo = c("regression","elastic net","random forests","xgboost"),
  cv_rqs = c(
    make_it_pretty(cv_OLS),
    make_it_pretty(cv_elastic),
    make_it_pretty(cv_random),
    make_it_pretty(cv_XGB)
  ),
  ho_rqs = c(
    make_it_pretty(holdout_OLS),
    make_it_pretty(holdout_elastic),
    make_it_pretty(holdout_random),
    make_it_pretty(holdout_XGB)
  )
)
table1_tbl

# Create a tibble to store the four algorithms, original & algorithms, and seconds
table2_tbl <- tibble


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
