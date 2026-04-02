############################
#### 0. Load Packages   ####
############################
#install.packages("randomForest")
#install.packages("e1071")
#install.packages("adabag")
#install.packages("gbm")
#install.packages("caret")
#install.packages("nnet")

library(randomForest)  
library(e1071)         
library(adabag)       
library(gbm)           
library(caret)         
library(nnet)   

############################
#### 1. Data Loading    ####
############################
white_wine <- read.csv("winequality-white.csv")
hist(white_wine$quality)

set.seed(123)
white_index <- sample(nrow(white_wine), 3000)
white_reg_train <- white_wine[white_index, -13]
white_reg_test  <- white_wine[-white_index, -13]

#################################################
#### 2. Random Forest Regression             #####
#################################################
rf_mtry_grid     <- c(4, 5, 6, 7, 8)
rf_ntree_grid    <- c(40, 50, 60, 70, 80)
rf_nodesize_grid <- c(10, 15, 20, 25)

rf_tuned <- tune.randomForest(
  x = white_reg_train[, -12],
  y = white_reg_train[, 12],
  mtry = rf_mtry_grid,
  ntree = rf_ntree_grid,
  nodesize = rf_nodesize_grid
)
rf_tuned$best.model
rf_tuned$best.parameters

rf_white_model <- randomForest(
  quality ~ .,
  data = white_reg_train,
  mtry = rf_tuned$best.parameters$mtry,
  ntree = rf_tuned$best.parameters$ntree,
  nodesize = rf_tuned$best.parameters$nodesize,
  importance = TRUE,
  proximity = TRUE,
  na.action = na.omit
)
importance(rf_white_model)
plot(rf_white_model)
print(rf_white_model)

rf_white_train_pred <- predict(rf_white_model, newdata = white_reg_train)
mean((white_reg_train$quality - rf_white_train_pred)^2)

rf_white_test_pred <- predict(rf_white_model, newdata = white_reg_test)
mean((white_reg_test$quality - rf_white_test_pred)^2)

#################################################
#### 3. Gradient Boosting Regression        #####
#################################################
gbm_cv_control <- trainControl(method = "cv", number = 3)
gbm_grid <- expand.grid(
  n.trees = seq(20, 80, 10),
  interaction.depth = c(3, 4, 5),
  shrinkage = c(0.01, 0.05, 0.1),
  n.minobsinnode = c(20, 30, 40)
)

gbm_white_tuned <- train(
  quality ~ .,
  data = white_wine,
  method = "gbm",
  trControl = gbm_cv_control,
  tuneGrid = gbm_grid,
  train.fraction = 0.5
)
gbm_white_tuned

gbm_white_tuned$bestTune
gbm_white_model <- gbm(
  quality ~ .,
  data = white_reg_train,
  n.trees = gbm_white_tuned$bestTune[1],
  interaction.depth = gbm_white_tuned$bestTune[2],
  shrinkage = gbm_white_tuned$bestTune[3],
  n.minobsinnode = gbm_white_tuned$bestTune[4]
)
summary(gbm_white_model)

gbm_white_train_pred <- predict(
  gbm_white_model,
  newdata = white_reg_train,
  n.trees = gbm_white_tuned$bestTune$n.trees
)
mean((white_reg_train$quality - gbm_white_train_pred)^2)

gbm_white_test_pred <- predict(
  gbm_white_model,
  newdata = white_reg_test,
  n.trees = gbm_white_tuned$bestTune$n.trees
)
mean((white_reg_test$quality - gbm_white_test_pred)^2)

#################################################
#### 4. ANN Regression (nnet)               #####
#################################################
ann_train_mse <- 0
ann_test_mse  <- 0
ann_rang <- 1 / max(abs(white_reg_train[, -12]))

for (i in 5:20)
{
  ann_temp_model <- nnet(
    quality ~ .,
    data = white_reg_train,
    maxit = 300,
    rang = ann_rang,
    size = i,
    decay = 5e-4,
    linout = TRUE
  )
  
  ann_temp_train_pred <- predict(ann_temp_model, white_reg_train[, -12], type = "raw")
  ann_train_mse[i - 4] <- mean((white_reg_train$quality - ann_temp_train_pred)^2)
  
  ann_temp_test_pred <- predict(ann_temp_model, white_reg_test[, -12], type = "raw")
  ann_test_mse[i - 4] <- mean((white_reg_test$quality - ann_temp_test_pred)^2)
}  

plot(
  5:20, ann_train_mse, 'l',
  col = 1, lty = 1, ylab = "MSE", xlab = "Hidden Neurons",
  ylim = c(min(min(ann_train_mse), min(ann_test_mse)),
           max(max(ann_train_mse), max(ann_test_mse)))
)
lines(5:20, ann_test_mse, col = 2, lty = 3)
points(5:20, ann_train_mse, col = 1, pch = "+")
points(5:20, ann_test_mse, col = 2, pch = "o")

ann_white_model <- nnet(
  quality ~ .,
  data = white_reg_train,
  size = 15,
  rang = ann_rang,
  decay = 5e-4,
  maxit = 300,
  linout = TRUE
)
summary(ann_white_model)

ann_white_train_pred <- predict(ann_white_model, white_reg_train[, -12], type = "raw")
mean((white_reg_train$quality - ann_white_train_pred)^2)

ann_white_test_pred <- predict(ann_white_model, white_reg_test[, -12], type = "raw")
mean((white_reg_test$quality - ann_white_test_pred)^2)

########################################
#### Regression Model Comparison    ####
########################################
result_table <- array(0, c(3, 3))
rownames(result_table) <- c("MSE", "MAE", "MAPE")
colnames(result_table) <- c("RandomForest", "GradientBoosting", "ANN")

# Random Forest
result_table[1, 1] <- mean((white_reg_test$quality - rf_white_test_pred)^2)
result_table[2, 1] <- mean(abs(white_reg_test$quality - rf_white_test_pred))
result_table[3, 1] <- mean(abs((white_reg_test$quality - rf_white_test_pred) / white_reg_test$quality))

# Gradient Boosting
result_table[1, 2] <- mean((white_reg_test$quality - gbm_white_test_pred)^2)
result_table[2, 2] <- mean(abs(white_reg_test$quality - gbm_white_test_pred))
result_table[3, 2] <- mean(abs((white_reg_test$quality - gbm_white_test_pred) / white_reg_test$quality))

# ANN
result_table[1, 3] <- mean((white_reg_test$quality - ann_white_test_pred)^2)
result_table[2, 3] <- mean(abs(white_reg_test$quality - ann_white_test_pred))
result_table[3, 3] <- mean(abs((white_reg_test$quality - ann_white_test_pred) / white_reg_test$quality))

round(result_table, 4)

#################################################
#### 5. Extended Analysis: Classification   #####
#################################################

############################
#### 5.1 Data Processing ####
############################
rating_label <- 0
for (i in 1:nrow(white_wine))
{
  if (white_wine[i, 12] > 6) rating_label[i] <- "good"
  else if (white_wine[i, 12] == 6) rating_label[i] <- "medium"
  else rating_label[i] <- "bad"
}
white_wine$rating <- factor(rating_label)
summary(white_wine$rating)

#################################################
#### 5.2 Bagging Classification            #####
#################################################
white_cls_train <- white_wine[white_index, -12]
white_cls_test  <- white_wine[-white_index, -12]

bagging_mfinal_grid <- c(40, 50, 60, 70, 80)
bagging_accuracy <- 0

for (i in 1:length(bagging_mfinal_grid))
{
  bagging_white_model <- bagging(rating ~ ., white_cls_train, mfinal = bagging_mfinal_grid[i])
  bagging_white_train_pred <- predict(bagging_white_model, white_cls_train)
  bagging_accuracy[i] <- sum(as.numeric(bagging_white_train_pred$class == white_cls_train[, 12])) / nrow(white_cls_train)
}
plot(bagging_accuracy, xlab = "mfinal", ylab = "acuracy")
bagging_best_index <- which.max(bagging_accuracy)

bagging_white_model <- bagging(rating ~ ., white_cls_train, mfinal = bagging_mfinal_grid[bagging_best_index])
bagging_white_model$importance

bagging_white_train_pred <- predict(bagging_white_model, white_cls_train)
bagging_white_train_pred$confusion
sum(as.numeric(bagging_white_train_pred$class == white_cls_train[, 12])) / nrow(white_cls_train)

bagging_white_test_pred <- predict(bagging_white_model, white_cls_test)
bagging_white_test_pred$confusion
sum(as.numeric(bagging_white_test_pred$class == white_cls_test[, 12])) / nrow(white_cls_test)