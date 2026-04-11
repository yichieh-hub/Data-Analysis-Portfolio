# install.packages("arules")
# install.packages("class")
# install.packages("e1071")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("ROCR")
# install.packages("RSNNS")
# install.packages("clusterSim")

library(arules)
library(class)
library(e1071)
library(rpart)
library(rpart.plot)
library(ROCR)
library(RSNNS)
library(clusterSim)

############################
##### 1. Data preprocessing
############################
pima_data <- read.csv("Pima.csv")
pima_data[pima_data[, 9] == 0, 9] <- "no"
pima_data[pima_data[, 9] == 1, 9] <- "yes"
pima_data[, 9] <- factor(pima_data$Outcome, levels = c("no", "yes"))

########################################
##### 2. Association rules
########################################
pima_assoc <- pima_data
for (i in 1:8) {
  pima_assoc[, i] <- cut(pima_assoc[, i], breaks = 2, include.lowest = TRUE)
}

rule <- apriori(
  pima_assoc,
  parameter = list(minlen = 3, supp = 0.05, conf = 0.6),
  appearance = list(rhs = "Outcome=yes")
)
inspect(sort(rule, by = "confidence"))

########################################
##### 3. Train / Test split
########################################
set.seed(123)
id <- sample(1:nrow(pima_data), 0.3 * nrow(pima_data))
train <- pima_data[-id, ]
test  <- pima_data[id, ]

########################################
##### 4. KNN
########################################
k_set <- c(3, 5, 7, 9)
acc <- rep(0, length(k_set))

for (i in 1:length(k_set)) {
  pred <- knn(train[, -9], train[, -9], train[, 9], k = k_set[i])
  acc[i] <- mean(pred == train[, 9])
}

k_best <- k_set[which.max(acc)]
k_best

knn_tr <- knn(train[, -9], train[, -9], train[, 9], k = k_best)
table(train$Outcome, knn_tr)
sum(as.numeric(knn_tr == train[, 9])) / nrow(train)

knn_ts <- knn(train[, -9], test[, -9], train[, 9], k = k_best, prob = TRUE)
table(test$Outcome, knn_ts)
sum(as.numeric(knn_ts == test[, 9])) / nrow(test)

# ROC
knn_prob <- attr(knn_ts, "prob")
knn_prob_yes <- ifelse(knn_ts == "yes", knn_prob, 1 - knn_prob)

pred_knn <- prediction(knn_prob_yes, test$Outcome)
perf_knn <- performance(pred_knn, measure = "tpr", x.measure = "fpr")
auc_knn <- performance(pred_knn, "auc")

plot(perf_knn, col = "blue", main = paste("ROC curve using KNN (k =", k_best, ")"), xlab = "FPR", ylab = "TPR")
abline(0, 1)
auc_value_knn <- round(auc_knn@y.values[[1]], 3)
text(0.4, 0.6, as.character(auc_value_knn))

########################################
##### 5. Logistic Regression
########################################
logit <- glm(Outcome ~ ., data = train, family = binomial(link = "logit"))
summary(logit)

logit_tr_prob <- predict.glm(logit, train, type = "response")
logit_tr <- ifelse(logit_tr_prob > 0.5, "yes", "no")
table(train$Outcome, logit_tr)
sum(as.numeric(logit_tr == train[, 9])) / nrow(train)

logit_ts_prob <- predict.glm(logit, test, type = "response")
logit_ts <- ifelse(logit_ts_prob > 0.5, "yes", "no")
table(test$Outcome, logit_ts)
sum(as.numeric(logit_ts == test[, 9])) / nrow(test)

# ROC
pred_logit <- prediction(logit_ts_prob, test$Outcome)
perf_logit <- performance(pred_logit, measure = "tpr", x.measure = "fpr")
auc_logit <- performance(pred_logit, "auc")

plot(perf_logit, col = "green", main = "ROC curve using Logistic Regression", xlab = "FPR", ylab = "TPR")
abline(0, 1)
auc_value_logit <- round(auc_logit@y.values[[1]], 3)
text(0.4, 0.6, as.character(auc_value_logit))

########################################
##### 6. CART
########################################
tune_cart <- tune.rpart(
  Outcome ~ .,
  data = train,
  minsplit = c(5, 10, 15),
  minbucket = c(5, 10, 15),
  maxdepth = c(3, 4, 5)
)

tune_cart$best.performance
tune_cart$best.parameters

best_cart <- tune_cart$best.parameters

cart <- rpart(
  Outcome ~ .,
  data = train,
  method = "class",
  minsplit = best_cart$minsplit,
  minbucket = best_cart$minbucket,
  maxdepth = best_cart$maxdepth
)

print(cart)
rpart.plot(cart, fallen.leaves = TRUE)
cart$variable.importance

cart_tr_prob <- predict(cart, train, type = "prob")
cart_tr <- ifelse(cart_tr_prob[, "yes"] > 0.5, "yes", "no")
table(train$Outcome, cart_tr)
sum(as.numeric(cart_tr == train[, 9])) / nrow(train)

cart_ts_prob <- predict(cart, test, type = "prob")
cart_ts <- ifelse(cart_ts_prob[, "yes"] > 0.5, "yes", "no")
table(test$Outcome, cart_ts)
sum(as.numeric(cart_ts == test[, 9])) / nrow(test)

# ROC
pred_cart <- prediction(cart_ts_prob[, "yes"], test$Outcome)
perf_cart <- performance(pred_cart, measure = "tpr", x.measure = "fpr")
auc_cart <- performance(pred_cart, "auc")

plot(perf_cart, col = "red", main = "ROC curve using CART", xlab = "FPR", ylab = "TPR")
abline(0, 1)
auc_value_cart <- round(auc_cart@y.values[[1]], 3)
text(0.4, 0.6, as.character(auc_value_cart))

########################################
##### 7. SVM
########################################
tune_svm <- tune.svm(
  Outcome ~ .,
  data = train,
  gamma = 10^(-3:1),
  cost = 10^(-1:2)
)

summary(tune_svm)
tune_svm$best.parameters

best_svm <- tune_svm$best.parameters

svm_model <- svm(
  Outcome ~ .,
  data = train,
  cost = best_svm$cost,
  gamma = best_svm$gamma,
  probability = TRUE
)

svm_tr <- predict(svm_model, train, probability = TRUE)
table(train$Outcome, svm_tr)
sum(as.numeric(svm_tr == train[, 9])) / nrow(train)

svm_ts <- predict(svm_model, test, probability = TRUE)
table(test$Outcome, svm_ts)
sum(as.numeric(svm_ts == test[, 9])) / nrow(test)

# ROC
svm_prob <- attr(svm_ts, "probabilities")
pred_svm <- prediction(svm_prob[, "yes"], test$Outcome)
perf_svm <- performance(pred_svm, measure = "tpr", x.measure = "fpr")
auc_svm <- performance(pred_svm, "auc")

plot(perf_svm, col = "orange", main = "ROC curve using SVM", xlab = "FPR", ylab = "TPR")
abline(0, 1)
auc_value_svm <- round(auc_svm@y.values[[1]], 3)
text(0.4, 0.6, as.character(auc_value_svm))

########################################
##### 8. MLP
########################################
norm <- data.Normalization(pima_data[, 1:8], type = "n4")
norm <- data.frame(norm, Outcome = pima_data$Outcome)

train_n <- norm[-id, ]
test_n  <- norm[id, ]

x_tr <- train_n[, 1:8]
y_tr <- decodeClassLabels(train_n[, 9])
x_ts <- test_n[, 1:8]
y_ts <- decodeClassLabels(test_n[, 9])

mlp <- mlp(x_tr, y_tr, size = 12, maxit = 300)

mlp_tr_prob <- predict(mlp, x_tr)
mlp_tr_prob <- as.matrix(mlp_tr_prob)
colnames(mlp_tr_prob) <- colnames(y_tr)

mlp_tr <- colnames(mlp_tr_prob)[max.col(mlp_tr_prob)]
table(train_n$Outcome, mlp_tr)
sum(as.numeric(mlp_tr == train_n[, 9])) / nrow(train_n)

mlp_ts_prob <- predict(mlp, x_ts)
mlp_ts_prob <- as.matrix(mlp_ts_prob)
colnames(mlp_ts_prob) <- colnames(y_ts)

mlp_ts <- colnames(mlp_ts_prob)[max.col(mlp_ts_prob)]
table(test_n$Outcome, mlp_ts)
sum(as.numeric(mlp_ts == test_n[, 9])) / nrow(test_n)

# ROC
pred_mlp <- prediction(mlp_ts_prob[, "yes"], test_n$Outcome)
perf_mlp <- performance(pred_mlp, measure = "tpr", x.measure = "fpr")
auc_mlp <- performance(pred_mlp, "auc")

plot(perf_mlp, col = "brown", main = "ROC curve using MLP", xlab = "FPR", ylab = "TPR")
abline(0, 1)
auc_value_mlp <- round(auc_mlp@y.values[[1]], 3)
text(0.4, 0.6, as.character(auc_value_mlp))