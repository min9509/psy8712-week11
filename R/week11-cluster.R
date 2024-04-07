#### Script Settings and Resources ####
library(tidyverse)
library(haven)
library(caret)
library(parallel)
library(doParallel)
library(tictoc)
set.seed(8712)

#### Data Import and Cleaning #### 
gss_original_tbl <- read_sav(file = "data/GSS2016.sav")

gss_tbl <- gss_original_tbl %>%
  filter(!is.na(MOSTHRS)) %>%
  rename(workhours = MOSTHRS) %>%
  select(-c(HRS1, HRS2)) %>%
  select(where(~mean(is.na(.)) < 0.75)) %>%
  sapply(as.numeric) %>% 
  as_tibble()  

#### Analysis #### 
holdout_indices <- createDataPartition(gss_tbl$workhours,
                                       p = .25,
                                       list = T)$Resample1

test_tbl <- gss_tbl[holdout_indices,]
training_tbl <- gss_tbl[-holdout_indices,]
training_folds <- createFolds(training_tbl$workhours)

# OLS Model
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
cv_OLS <- model_OLS$results$Rsquared
holdout_OLS <- cor(
  predict(model_OLS, test_tbl, na.action = na.pass),
  test_tbl$workhours
)^2

# Elastic net model
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
cv_elastic <- max(model_elastic$results$Rsquared)
holdout_elastic <- cor(
  predict(model_elastic, test_tbl, na.action = na.pass),
  test_tbl$workhours
)^2

# Random forest model 
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
cv_random <- max(model_random$results$Rsquared)
holdout_random <- cor(
  predict(model_random, test_tbl, na.action = na.pass),
  test_tbl$workhours
)^2

# Random XGB
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
cv_XGB <- max(model_XGB$results$Rsquared)
holdout_XGB <- cor(
  predict(model_XGB, test_tbl, na.action = na.pass),
  test_tbl$workhours
)^2

summary(resamples(list(model_OLS, model_elastic, model_random, model_XGB)), metric="Rsquared")
dotplot(resamples(list(model_OLS, model_elastic, model_random, model_XGB)), metric="Rsquared")

## Supercomputer running times ##
# OLS model
tic()
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
toc_OLS <- toc()
toc_OLS

# Elastic net model
tic()
model_elastic_Par <- train(
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
toc_elastic <- toc()

# Random forest model
tic()
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
toc_random <- toc()

# XGB model
tic()
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
toc_XGB <- toc()

## Super computing running times ##
local_cluster <- makeCluster(detectCores() - 1)
registerDoParallel(local_cluster)

# OLS 
tic()
model_OLS_Par <- train(
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
toc_OLS_Par <- toc()

# Elastic net model - Add tic and toc check running times for parallel.
tic()
model_elastic_Par <- train(
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
toc_elastic_Par <- toc()

# Random forest model
tic() 
model_random_Par <- train(
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
toc_random_Par <- toc()

# XGB model
tic()
model_XGB_Par <- train(
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
toc_XGB_Par <- toc()

# Stop local cluster and parallel
stopCluster(local_cluster)
registerDoSEQ()

#### Publication #### 
# Create formats numerical values to make them look visually appealing
make_it_pretty <- function (formatme) {
  formatme <- formatC(formatme, format="f", digits=2)
  formatme <- str_remove(formatme, "^0")
  return(formatme)
}

# Create a tibble to store algorithm names, cross-validation R-squared, and holdout R-squared values
table3_tbl <- tibble(
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
table3_tbl

# Create a tibble to store the four algorithms, original & algorithms, and seconds
table4_tbl <- tibble(
  algo = c("regression","elastic net","random forests","xgboost"),
  supercomputer = c(toc_OLS$callback_msg, toc_elastic$callback_msg, toc_random$callback_msg, toc_XGB$callback_msg),
  supercomputer_n = c(toc_OLS_Par$callback_msg, toc_elastic_Par$callback_msg, toc_random_Par$callback_msg, toc_XGB_Par$callback_msg)
)
table4_tbl

# Export csv file
write_csv(table3_tbl, "data/table3.csv")
write_csv(table4_tbl, "data/table4.csv")

# A1
# Based on results, it appears that the Random Forests algorithm benefited the most from moving to the supercomputer, 
# as it showed a significant improvement in both CV (.92) and Ho RQS(.61) metrics.

# A2
# Generally, as the number of cores increases, the computation time tends to decrease. 
# This is because parallel processing allows tasks to be divided among multiple cores, speeding up overall computation. 
# However, there may be diminishing returns as the number of cores increases significantly due to overhead and resource contention.

# A3
# Elastic Net shows a significant improvement in RQS metrics when using the supercomputer, 
# but it also requires a relatively high number of cores. T
# herefore, the decision to use the supercomputer for Elastic Net should consider the availability of resources and the specific requirements of the production model.

# I tried to use putty, but I couldn't. It said that network does not connect. 
# So, I created RStudio Server, connected git hub, and run all codes. Then, answered three questions in objective 26.

# > table3_tbl
# A tibble: 4 × 3
# algo           cv_rqs ho_rqs
# <chr>          <chr>  <chr> 
# 1 regression     .17    .01   
# 2 elastic net    .81    .48   
# 3 random forests .91    .60   
# 4 xgboost        .96    .49   
# > table4_tbl
# A tibble: 4 × 3
# algo           supercomputer       supercomputer_n    
# <chr>          <chr>               <chr>              
# 1 regression     7.581 sec elapsed   7.469 sec elapsed  
# 2 elastic net    15.851 sec elapsed  16.545 sec elapsed 
# 3 random forests 126.069 sec elapsed 135.285 sec elapsed
# 4 xgboost        8.209 sec elapsed   8.212 sec elapsed  




