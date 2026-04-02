########################
## Required packages  ##
########################
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("e1071")
# install.packages("earth")
# install.packages("FNN")
# install.packages("randomForest")
# install.packages("adabag")
# install.packages("gbm")
# install.packages("caret")
# install.packages("neuralnet")
# install.packages("nnet")
# install.packages("psych")
# install.packages("klaR")

library(rpart)
library(rpart.plot)
library(e1071)
library(earth)
library(FNN)
library(randomForest)
library(adabag)
library(gbm)
library(caret)
library(neuralnet)
library(nnet)
library(psych)
library(klaR)

########################
## Data preprocessing ##
########################
red_wine <- read.csv("winequality-red.csv")
hist(red_wine$quality)

rating_char <- 0
for (i in 1:nrow(red_wine))
{
  if (red_wine[i, 12] > 6) rating_char[i] <- "excellent"
  else if (red_wine[i, 12] == 6) rating_char[i] <- "good"
  else if (red_wine[i, 12] == 5) rating_char[i] <- "medium"
  else rating_char[i] <- "bad"
}
red_wine$rating <- factor(rating_char)
summary(red_wine$rating)

red_id <- sample(nrow(red_wine), 1000)

## classification data
red_cls_train <- red_wine[red_id, -12]
red_cls_test  <- red_wine[-red_id, -12]

## regression data
red_reg_train <- red_wine[red_id, -13]
red_reg_test  <- red_wine[-red_id, -13]

######################################################
################ Part 1. Classification ##############
######################################################

#############################
## 1. CART classification  ##
#############################
minsplit_set <- c(15, 20, 25)
minbucket_set <- c(10, 15, 20)
maxdepth_set <- c(3, 4, 5, 6)

tune_cart_cls <- tune.rpart(
  rating ~ .,
  data = red_cls_train,
  minsplit = minsplit_set,
  minbucket = minbucket_set,
  maxdepth = maxdepth_set
)
tune_cart_cls$best.performance
tune_cart_cls$best.parameters

cart_red_cls <- rpart(
  rating ~ .,
  red_cls_train,
  method = "class",
  minsplit = 15,
  minbucket = 10,
  maxdepth = 3
)
print(cart_red_cls)
summary(cart_red_cls)
rpart.plot(cart_red_cls, fallen.leaves = TRUE)
cart_red_cls$variable.importance

pred_cart_red_cls_train <- predict(cart_red_cls, red_cls_train, type = "class")
table(red_cls_train$rating, pred_cart_red_cls_train)
sum(as.numeric(pred_cart_red_cls_train == red_cls_train[, 12])) / nrow(red_cls_train)

pred_cart_red_cls_test <- predict(cart_red_cls, red_cls_test, type = "class")
table(red_cls_test$rating, pred_cart_red_cls_test)
sum(as.numeric(pred_cart_red_cls_test == red_cls_test[, 12])) / nrow(red_cls_test)

################################
## 2. Random Forest class     ##
################################
mtry_set <- c(4, 6, 8, 10)
ntree_set <- c(30, 40, 50, 60, 70)
nodesize_set <- c(10, 15, 20, 25)

tune_rf_cls <- tune.randomForest(
  x = red_cls_train[, -12],
  y = red_cls_train$rating,
  mtry = mtry_set,
  ntree = ntree_set,
  nodesize = nodesize_set
)
tune_rf_cls$best.model
tune_rf_cls$best.parameters

rf_red_cls <- randomForest(
  rating ~ .,
  data = red_cls_train,
  ntree = 40,
  mtry = 6,
  nodesize = 10,
  importance = TRUE,
  proximity = TRUE,
  na.action = na.omit
)
importance(rf_red_cls)
plot(rf_red_cls)
print(rf_red_cls)

pred_rf_red_cls_train <- predict(rf_red_cls, newdata = red_cls_train)
table(red_cls_train$rating, pred_rf_red_cls_train)
sum(as.numeric(pred_rf_red_cls_train == red_cls_train[, 12])) / nrow(red_cls_train)

pred_rf_red_cls_test <- predict(rf_red_cls, newdata = red_cls_test)
table(red_cls_test$rating, pred_rf_red_cls_test)
sum(as.numeric(pred_rf_red_cls_test == red_cls_test[, 12])) / nrow(red_cls_test)

#############################
## 3. AdaBoost class       ##
#############################
acc_ada_red <- 0
for (j in 1:length(ntree_set))
{
  ada_red_cls_temp <- boosting(rating ~ ., red_cls_train, mfinal = ntree_set[j], boos = TRUE)
  pred_ada_red_temp <- predict(ada_red_cls_temp, red_cls_train)
  acc_ada_red[j] <- sum(as.numeric(pred_ada_red_temp$class == red_cls_train[, 12])) / nrow(red_cls_train)
}
plot(acc_ada_red, xlab = "mfinal", ylab = "acuracy")
best_ada_idx <- which.max(acc_ada_red)

ada_red_cls <- boosting(rating ~ ., red_cls_train, mfinal = ntree_set[best_ada_idx], boos = TRUE)
ada_red_cls$importance

pred_ada_red_cls_train <- predict(ada_red_cls, red_cls_train)
pred_ada_red_cls_train$confusion
sum(as.numeric(pred_ada_red_cls_train$class == red_cls_train[, 12])) / nrow(red_cls_train)

pred_ada_red_cls_test <- predict(ada_red_cls, red_cls_test)
pred_ada_red_cls_test$confusion
sum(as.numeric(pred_ada_red_cls_test$class == red_cls_test[, 12])) / nrow(red_cls_test)

##################################
## 4. Gradient Boosting class   ##
##################################
cv_control_gbm_cls <- trainControl(method = "cv", number = 3)
grid_gbm_cls <- expand.grid(
  n.trees = seq(20, 80, 10),
  interaction.depth = c(3, 4, 5, 6),
  shrinkage = c(0.01, 0.05, 0.1),
  n.minobsinnode = c(10, 20, 30)
)

cv_gbm_red_cls <- train(
  rating ~ .,
  data = red_wine,
  method = "gbm",
  trControl = cv_control_gbm_cls,
  tuneGrid = grid_gbm_cls,
  train.fraction = 0.5
)
cv_gbm_red_cls
confusionMatrix(cv_gbm_red_cls, norm = "none")

cv_gbm_red_cls$bestTune
gbm_red_cls <- gbm(
  rating ~ .,
  data = red_cls_train,
  n.trees = cv_gbm_red_cls$bestTune[1],
  interaction.depth = cv_gbm_red_cls$bestTune[2],
  shrinkage = cv_gbm_red_cls$bestTune[3],
  n.minobsinnode = cv_gbm_red_cls$bestTune[4]
)
summary(gbm_red_cls)

pred_gbm_red_cls_train <- predict(
  gbm_red_cls,
  newdata = red_cls_train,
  n.trees = cv_gbm_red_cls$bestTune$n.trees,
  type = "response"
)
pred_gbm_red_cls_train <- as.data.frame(pred_gbm_red_cls_train)
colnames(pred_gbm_red_cls_train) <- c("bad", "excellent", "good", "medium")
pred_gbm_red_cls_train <- colnames(pred_gbm_red_cls_train)[apply(pred_gbm_red_cls_train, 1, which.max)]
table(red_cls_train$rating, pred_gbm_red_cls_train)
sum(as.numeric(pred_gbm_red_cls_train == red_cls_train[, 12])) / nrow(red_cls_train)

pred_gbm_red_cls_test <- predict(
  gbm_red_cls,
  newdata = red_cls_test,
  n.trees = cv_gbm_red_cls$bestTune$n.trees,
  type = "response"
)
pred_gbm_red_cls_test <- as.data.frame(pred_gbm_red_cls_test)
colnames(pred_gbm_red_cls_test) <- c("bad", "excellent", "good", "medium")
pred_gbm_red_cls_test <- colnames(pred_gbm_red_cls_test)[apply(pred_gbm_red_cls_test, 1, which.max)]
table(red_cls_test$rating, pred_gbm_red_cls_test)
sum(as.numeric(pred_gbm_red_cls_test == red_cls_test[, 12])) / nrow(red_cls_test)

###########################
## 5. SVM classification ##
###########################
tune_svm_red_cls <- tune.svm(
  rating ~ .,
  data = red_cls_train,
  gamma = 10^(-2:1),
  cost = 10^(-1:3)
)
summary(tune_svm_red_cls)
tune_svm_red_cls$best.parameters

svm_red_cls <- tune_svm_red_cls$best.model

pred_svm_red_cls_train <- predict(svm_red_cls, red_cls_train[, -12])
table(red_cls_train[, 12], pred_svm_red_cls_train)
sum(as.numeric(pred_svm_red_cls_train == red_cls_train[, 12])) / nrow(red_cls_train)

pred_svm_red_cls_test <- predict(svm_red_cls, red_cls_test[, -12])
table(red_cls_test[, 12], pred_svm_red_cls_test)
sum(as.numeric(pred_svm_red_cls_test == red_cls_test[, 12])) / nrow(red_cls_test)

##################################
## 6. ANN classification        ##
##    neuralnet version         ##
##################################
red_ann_cls_train <- red_cls_train
red_ann_cls_test  <- red_cls_test
red_ann_cls_train$rating <- as.numeric(red_ann_cls_train$rating)
red_ann_cls_test$rating  <- as.numeric(red_ann_cls_test$rating)

ann_red_cls <- neuralnet(
  rating ~ .,
  red_ann_cls_train,
  hidden = 10,
  threshold = 2,
  learningrate = 0.01
)
summary(ann_red_cls)

class_name_red <- c("bad", "excellent", "good", "medium")

pred_ann_red_cls_train <- compute(ann_red_cls, red_ann_cls_train[, 1:11])$net.result
pred_ann_red_cls_train[which(round(pred_ann_red_cls_train) > 4)] <- 4
pred_ann_red_cls_train[which(round(pred_ann_red_cls_train) < 1)] <- 1
table(red_wine[red_id, 13], class_name_red[round(pred_ann_red_cls_train)])
sum(as.numeric(class_name_red[round(pred_ann_red_cls_train)] == red_wine[red_id, 13])) / nrow(red_ann_cls_train)

pred_ann_red_cls_test <- compute(ann_red_cls, red_ann_cls_test[, 1:11])$net.result
pred_ann_red_cls_test[which(round(pred_ann_red_cls_test) > 4)] <- 4
pred_ann_red_cls_test[which(round(pred_ann_red_cls_test) < 1)] <- 1
table(red_wine[-red_id, 13], class_name_red[round(pred_ann_red_cls_test)])
sum(as.numeric(class_name_red[round(pred_ann_red_cls_test)] == red_wine[-red_id, 13])) / nrow(red_ann_cls_test)

##################################
## 7. BPN / nnet classification ##
##################################
acc_bpn_red_train <- 0
acc_bpn_red_test <- 0
rang_red_cls <- 1 / max(abs(red_cls_train[, -12]))

for (i in 5:20)
{
  bpn_red_cls_temp <- nnet(
    rating ~ .,
    data = red_cls_train,
    maxit = 250,
    rang = rang_red_cls,
    size = i,
    decay = 5e-4
  )
  
  pred_bpn_red_train_temp <- predict(bpn_red_cls_temp, red_cls_train[, -12], type = "class")
  table(red_cls_train[, 12], pred_bpn_red_train_temp)
  acc_bpn_red_train[i - 4] <- sum(as.numeric(pred_bpn_red_train_temp == red_cls_train[, 12])) / nrow(red_cls_train)
  
  pred_bpn_red_test_temp <- predict(bpn_red_cls_temp, red_cls_test[, -12], type = "class")
  table(red_cls_test[, 12], pred_bpn_red_test_temp)
  acc_bpn_red_test[i - 4] <- sum(as.numeric(pred_bpn_red_test_temp == red_cls_test[, 12])) / nrow(red_cls_test)
}

plot(
  5:20, acc_bpn_red_train, "l", col = 1, lty = 1,
  ylab = "Accuracy", xlab = "Hidden Neurons",
  ylim = c(min(min(acc_bpn_red_train), min(acc_bpn_red_test)),
           max(max(acc_bpn_red_train), max(acc_bpn_red_test)))
)
lines(5:20, acc_bpn_red_test, col = 2, lty = 3)
points(5:20, acc_bpn_red_train, col = 1, pch = "+")
points(5:20, acc_bpn_red_test, col = 2, pch = "o")
legend(15, 0.7, "train", bty = "n", cex = 1)
legend(15, 0.6, "test", bty = "n", cex = 1)

bpn_red_cls <- nnet(
  rating ~ .,
  data = red_cls_train,
  size = 7,
  rang = rang_red_cls,
  maxit = 300
)
summary(bpn_red_cls)

pred_bpn_red_cls_train <- predict(bpn_red_cls, red_cls_train[, -12], type = "class")
table(red_cls_train[, 12], pred_bpn_red_cls_train)
sum(as.numeric(pred_bpn_red_cls_train == red_cls_train[, 12])) / nrow(red_cls_train)

pred_bpn_red_cls_test <- predict(bpn_red_cls, red_cls_test[, -12], type = "class")
table(red_cls_test[, 12], pred_bpn_red_cls_test)
sum(as.numeric(pred_bpn_red_cls_test == red_cls_test[, 12])) / nrow(red_cls_test)

#############################
## Classification summary  ##
#############################
acc_table <- array(0, c(1, 7))
rownames(acc_table) <- c("Accuracy")
colnames(acc_table) <- c("CART", "RandomForest", "AdaBoost", "GradientBoosting", "SVM", "ANN", "BPN")

acc_table[1, 1] <- sum(as.numeric(pred_cart_red_cls_test == red_cls_test$rating)) / nrow(red_cls_test)
acc_table[1, 2] <- sum(as.numeric(pred_rf_red_cls_test == red_cls_test$rating)) / nrow(red_cls_test)
acc_table[1, 3] <- sum(as.numeric(pred_ada_red_cls_test$class == red_cls_test$rating)) / nrow(red_cls_test)
acc_table[1, 4] <- sum(as.numeric(pred_gbm_red_cls_test == red_cls_test$rating)) / nrow(red_cls_test)
acc_table[1, 5] <- sum(as.numeric(pred_svm_red_cls_test == red_cls_test$rating)) / nrow(red_cls_test)
acc_table[1, 6] <- sum(as.numeric(class_name_red[round(pred_ann_red_cls_test)] == red_cls_test$rating)) / nrow(red_cls_test)
acc_table[1, 7] <- sum(as.numeric(pred_bpn_red_cls_test == red_cls_test$rating)) / nrow(red_cls_test)

round(acc_table, 4)

##################################################
################ Part 2. Regression ##############
##################################################

#########################
## 1. CART regression  ##
#########################
tune_cart_red_reg <- tune.rpart(
  quality ~ .,
  data = red_reg_train,
  minsplit = minsplit_set,
  minbucket = minbucket_set,
  maxdepth = maxdepth_set
)
tune_cart_red_reg$best.performance
tune_cart_red_reg$best.parameters

cart_red_reg <- rpart(
  quality ~ .,
  red_reg_train,
  method = "anova",
  minsplit = 15,
  minbucket = 10,
  maxdepth = 6
)
print(cart_red_reg)
rpart.plot(cart_red_reg, type = 4, fallen.leaves = FALSE)
cart_red_reg$variable.importance

pred_cart_red_reg_train <- predict(cart_red_reg, red_reg_train)
pred_cart_red_reg_train
mean((red_reg_train$quality - pred_cart_red_reg_train)^2)

pred_cart_red_reg_test <- predict(cart_red_reg, red_reg_test)
pred_cart_red_reg_test
mean((red_reg_test$quality - pred_cart_red_reg_test)^2)

#########################
## 2. MLR regression   ##
#########################
mlr_red_reg <- lm(quality ~ ., data = red_reg_train)
summary(mlr_red_reg)
mean((red_reg_train$quality - mlr_red_reg$fitted.values)^2)

pred_mlr_red_reg_test <- predict(mlr_red_reg, red_reg_test[, -12])
mean((red_reg_test$quality - pred_mlr_red_reg_test)^2)

#########################
## 3. MARS regression  ##
#########################
mars_red_reg <- earth(quality ~ ., degree = 3, trace = 2, data = red_reg_train)
summary(mars_red_reg)
evimp(mars_red_reg, trim = FALSE)
mean((red_reg_train$quality - mars_red_reg$fitted.values)^2)

pred_mars_red_reg_test <- predict(mars_red_reg, red_reg_test[, -12])
mean((red_reg_test$quality - pred_mars_red_reg_test)^2)

#########################
## 4. KNN regression   ##
#########################
knn_red_reg <- knn.reg(
  red_reg_train[, -12],
  red_reg_train[, -12],
  red_reg_train[, 12],
  k = 2,
  algorithm = "brute"
)
mean((red_reg_train$quality - knn_red_reg$pred)^2)

pred_knn_red_reg_test <- knn.reg(
  red_reg_train[, -12],
  red_reg_test[, -12],
  red_reg_train[, 12],
  k = 2,
  algorithm = "brute"
)$pred
mean((red_reg_test$quality - pred_knn_red_reg_test)^2)

#########################
## 5. SVR regression   ##
#########################
tune_svr_red_reg <- tune(
  svm,
  quality ~ .,
  data = red_reg_train,
  ranges = list(epsilon = seq(0, 0.4, 0.05), cost = 10^(-1:3))
)
svr_red_reg <- tune_svr_red_reg$best.model

mean((red_reg_train$quality - svr_red_reg$fitted)^2)^0.5
pred_svr_red_reg_test <- predict(svr_red_reg, red_reg_test[, -12])
mean((red_reg_test$quality - pred_svr_red_reg_test)^2)^0.5

##################################
## 6. ANN regression            ##
##    neuralnet version         ##
##################################
ann_red_reg <- neuralnet(
  quality ~ .,
  red_reg_train,
  hidden = 10,
  threshold = 2,
  learningrate = 0.01
)
summary(ann_red_reg)
ann_red_reg$weights

pred_ann_red_reg_train <- compute(ann_red_reg, red_reg_train[, 1:11])$net.result
mean((red_reg_train$quality - pred_ann_red_reg_train)^2)

pred_ann_red_reg_test <- compute(ann_red_reg, red_reg_test[, 1:11])$net.result
mean((red_reg_test$quality - pred_ann_red_reg_test)^2)

#########################
## Regression summary  ##
#########################
result_table <- array(0, c(3, 6))
rownames(result_table) <- c("MSE", "MAE", "MAPE")
colnames(result_table) <- c("CART", "MLR", "MARS", "KNN", "SVR", "ANN")

# CART
result_table[1, 1] <- mean((red_reg_test$quality - pred_cart_red_reg_test)^2)
result_table[2, 1] <- mean(abs(red_reg_test$quality - pred_cart_red_reg_test))
result_table[3, 1] <- mean(abs((red_reg_test$quality - pred_cart_red_reg_test) / red_reg_test$quality))

# MLR
result_table[1, 2] <- mean((red_reg_test$quality - pred_mlr_red_reg_test)^2)
result_table[2, 2] <- mean(abs(red_reg_test$quality - pred_mlr_red_reg_test))
result_table[3, 2] <- mean(abs((red_reg_test$quality - pred_mlr_red_reg_test) / red_reg_test$quality))

# MARS
result_table[1, 3] <- mean((red_reg_test$quality - pred_mars_red_reg_test)^2)
result_table[2, 3] <- mean(abs(red_reg_test$quality - pred_mars_red_reg_test))
result_table[3, 3] <- mean(abs((red_reg_test$quality - pred_mars_red_reg_test) / red_reg_test$quality))

# KNN
result_table[1, 4] <- mean((red_reg_test$quality - pred_knn_red_reg_test)^2)
result_table[2, 4] <- mean(abs(red_reg_test$quality - pred_knn_red_reg_test))
result_table[3, 4] <- mean(abs((red_reg_test$quality - pred_knn_red_reg_test) / red_reg_test$quality))

# SVR
result_table[1, 5] <- mean((red_reg_test$quality - pred_svr_red_reg_test)^2)
result_table[2, 5] <- mean(abs(red_reg_test$quality - pred_svr_red_reg_test))
result_table[3, 5] <- mean(abs((red_reg_test$quality - pred_svr_red_reg_test) / red_reg_test$quality))

# ANN
result_table[1, 6] <- mean((red_reg_test$quality - pred_ann_red_reg_test)^2)
result_table[2, 6] <- mean(abs(red_reg_test$quality - pred_ann_red_reg_test))
result_table[3, 6] <- mean(abs((red_reg_test$quality - pred_ann_red_reg_test) / red_reg_test$quality))

round(result_table, 4)

#############################################
################ Part 3. PCA ################
############### placed at end ###############
#############################################

######################
## 1. PCA for red   ##
######################
red_pca <- prcomp(~., data = red_wine[, -12], center = TRUE, scale = TRUE)
summary(red_pca)
plot(red_pca, type = "line", main = "Scree Plot for redwine")
red_pca$sdev^2

red_rpca <- principal(red_wine[, -12], nfactors = 5, scores = TRUE)
print(red_rpca)
red_rpca$loadings

red_pca_data <- data.frame(red_rpca$scores)
red_pca_data$rating <- red_wine$rating
red_pca_data$quality <- red_wine$quality

red_pca_cls_train <- red_pca_data[red_id, 1:6]
red_pca_cls_test  <- red_pca_data[-red_id, 1:6]

red_pca_reg_train <- red_pca_data[red_id, c(1:5, 7)]
red_pca_reg_test  <- red_pca_data[-red_id, c(1:5, 7)]

##################################
## 2. PCA classification - KNN  ##
##################################
pred_pca_knn_cls_test <- knn(
  train = red_pca_cls_train[, 1:5],
  test  = red_pca_cls_test[, 1:5],
  cl    = red_pca_cls_train[, 6],
  k = 3
)
table(red_pca_cls_test[, 6], pred_pca_knn_cls_test)
sum(as.numeric(pred_pca_knn_cls_test == red_pca_cls_test[, 6])) / nrow(red_pca_cls_test)

##########################################
## 3. PCA classification - Naive Bayes  ##
##########################################
pca_nb_red_cls <- NaiveBayes(rating ~ ., data = red_pca_cls_train)
pred_pca_nb_cls_test <- predict(pca_nb_red_cls, red_pca_cls_test[, -6])$class
table(red_pca_cls_test[, 6], pred_pca_nb_cls_test)
sum(as.numeric(pred_pca_nb_cls_test == red_pca_cls_test[, 6])) / nrow(red_pca_cls_test)

#############################
## PCA classification table##
#############################
pca_acc_table <- array(0, c(1, 2))
rownames(pca_acc_table) <- c("Accuracy")
colnames(pca_acc_table) <- c("PCA_KNN", "PCA_NaiveBayes")

pca_acc_table[1, 1] <- sum(as.numeric(pred_pca_knn_cls_test == red_pca_cls_test[, 6])) / nrow(red_pca_cls_test)
pca_acc_table[1, 2] <- sum(as.numeric(pred_pca_nb_cls_test == red_pca_cls_test[, 6])) / nrow(red_pca_cls_test)

round(pca_acc_table, 4)

##############################
## 4. PCA regression - KNN  ##
##############################
pred_pca_knn_reg_test <- knn.reg(
  train = red_pca_reg_train[, 1:5],
  test  = red_pca_reg_test[, 1:5],
  y     = red_pca_reg_train[, 6],
  k = 2,
  algorithm = "brute"
)$pred

sqrt(mean((red_pca_reg_test$quality - pred_pca_knn_reg_test)^2))
mean(abs(red_pca_reg_test$quality - pred_pca_knn_reg_test))
mean(abs((red_pca_reg_test$quality - pred_pca_knn_reg_test) / red_pca_reg_test$quality))

###############################
## 5. PCA regression - MARS  ##
###############################
mars_pca_red_reg <- earth(quality ~ ., degree = 3, trace = 2, data = red_pca_reg_train)
summary(mars_pca_red_reg)

pred_pca_mars_reg_test <- predict(mars_pca_red_reg, red_pca_reg_test[, -6])

sqrt(mean((red_pca_reg_test$quality - pred_pca_mars_reg_test)^2))
mean(abs(red_pca_reg_test$quality - pred_pca_mars_reg_test))
mean(abs((red_pca_reg_test$quality - pred_pca_mars_reg_test) / red_pca_reg_test$quality))

#########################
## PCA regression table##
#########################
pca_result_table <- array(0, c(3, 2))
rownames(pca_result_table) <- c("MSE", "MAE", "MAPE")
colnames(pca_result_table) <- c("PCA_KNN", "PCA_MARS")

# PCA KNN
pca_result_table[1, 1] <- mean((red_pca_reg_test$quality - pred_pca_knn_reg_test)^2)
pca_result_table[2, 1] <- mean(abs(red_pca_reg_test$quality - pred_pca_knn_reg_test))
pca_result_table[3, 1] <- mean(abs((red_pca_reg_test$quality - pred_pca_knn_reg_test) / red_pca_reg_test$quality))

# PCA MARS
pca_result_table[1, 2] <- mean((red_pca_reg_test$quality - pred_pca_mars_reg_test)^2)
pca_result_table[2, 2] <- mean(abs(red_pca_reg_test$quality - pred_pca_mars_reg_test))
pca_result_table[3, 2] <- mean(abs((red_pca_reg_test$quality - pred_pca_mars_reg_test) / red_pca_reg_test$quality))

round(pca_result_table, 4)