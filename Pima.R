##### Pima dataset portfolio version

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
##### 4. ROC plot function
########################################
roc_plot <- function(prob, truth, name) {
  pred_obj <- prediction(prob, truth)
  perf_obj <- performance(pred_obj, "tpr", "fpr")
  auc_obj <- performance(pred_obj, "auc")
  auc_value <- round(auc_obj@y.values[[1]], 3)
  
  plot(
    perf_obj,
    col = "blue",
    lwd = 3,
    main = paste("ROC Curve -", name),
    xlab = "False Positive Rate",
    ylab = "True Positive Rate"
  )
  
  abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)
  
  text(
    x = 0.6,
    y = 0.2,
    labels = paste("AUC =", auc_value),
    cex = 1.1
  )
  
  legend(
    "bottomright",
    legend = c("ROC curve", "Baseline", paste("AUC =", auc_value)),
    col = c("blue", "red", NA),
    lwd = c(3, 2, NA),
    lty = c(1, 2, NA),
    bty = "n"
  )
  
  return(auc_value)
}

########################################
##### 5. KNN
########################################
k_set <- c(1, 3, 5, 7, 9)
acc <- rep(0, length(k_set))

for (i in 1:length(k_set)) {
  pred <- knn(train[, -9], train[, -9], train[, 9], k = k_set[i])
  acc[i] <- mean(pred == train[, 9])
}

k_best <- k_set[which.max(acc)]
k_best

pred_tr <- knn(train[, -9], train[, -9], train[, 9], k = k_best)
pred_ts <- knn(train[, -9], test[, -9], train[, 9], k = k_best, prob = TRUE)

cat("KNN Train Accuracy:", mean(pred_tr == train[, 9]), "\n")
cat("KNN Test Accuracy:", mean(pred_ts == test[, 9]), "\n")

prob <- attr(pred_ts, "prob")
prob_yes <- ifelse(pred_ts == "yes", prob, 1 - prob)

auc_knn <- roc_plot(prob_yes, test[, 9], paste("KNN (k =", k_best, ")"))
cat("KNN Test AUC:", auc_knn, "\n")

########################################
##### 6. Logistic
########################################
logit <- glm(Outcome ~ ., data = train, family = binomial)

prob_tr <- predict(logit, train, type = "response")
pred_tr <- ifelse(prob_tr > 0.5, "yes", "no")

prob_ts <- predict(logit, test, type = "response")
pred_ts <- ifelse(prob_ts > 0.5, "yes", "no")

cat("Logit Train Accuracy:", mean(pred_tr == train[, 9]), "\n")
cat("Logit Test Accuracy:", mean(pred_ts == test[, 9]), "\n")

auc_logit <- roc_plot(prob_ts, test[, 9], "Logistic")
cat("Logit Test AUC:", auc_logit, "\n")

########################################
##### 7. Probit
########################################
probit <- glm(Outcome ~ ., data = train, family = quasibinomial(link = "probit"))

prob_tr <- predict(probit, train, type = "response")
pred_tr <- ifelse(prob_tr > 0.5, "yes", "no")

prob_ts <- predict(probit, test, type = "response")
pred_ts <- ifelse(prob_ts > 0.5, "yes", "no")

cat("Probit Train Accuracy:", mean(pred_tr == train[, 9]), "\n")
cat("Probit Test Accuracy:", mean(pred_ts == test[, 9]), "\n")

auc_probit <- roc_plot(prob_ts, test[, 9], "Probit")
cat("Probit Test AUC:", auc_probit, "\n")

########################################
##### 8. CART
########################################
tune_cart <- tune.rpart(
  Outcome ~ .,
  data = train,
  minsplit = c(5, 10, 15),
  minbucket = c(5, 10, 15),
  maxdepth = c(3, 4, 5)
)

best <- tune_cart$best.parameters

cart <- rpart(
  Outcome ~ .,
  data = train,
  minsplit = best$minsplit,
  minbucket = best$minbucket,
  maxdepth = best$maxdepth
)

rpart.plot(cart)

prob_tr <- predict(cart, train, type = "prob")[, "yes"]
pred_tr <- ifelse(prob_tr > 0.5, "yes", "no")

prob_ts <- predict(cart, test, type = "prob")[, "yes"]
pred_ts <- ifelse(prob_ts > 0.5, "yes", "no")

cat("CART Train Accuracy:", mean(pred_tr == train[, 9]), "\n")
cat("CART Test Accuracy:", mean(pred_ts == test[, 9]), "\n")

auc_cart <- roc_plot(prob_ts, test[, 9], "CART")
cat("CART Test AUC:", auc_cart, "\n")

########################################
##### 9. SVM
########################################
tune_svm <- tune.svm(
  Outcome ~ .,
  data = train,
  gamma = 10^(-3:1),
  cost = 10^(-1:2)
)

best <- tune_svm$best.parameters

svm_model <- svm(
  Outcome ~ .,
  data = train,
  cost = best$cost,
  gamma = best$gamma,
  probability = TRUE
)

pred_tr <- predict(svm_model, train)
pred_ts <- predict(svm_model, test, probability = TRUE)

cat("SVM Train Accuracy:", mean(pred_tr == train[, 9]), "\n")
cat("SVM Test Accuracy:", mean(pred_ts == test[, 9]), "\n")

prob <- attr(pred_ts, "probabilities")[, "yes"]

auc_svm <- roc_plot(prob, test[, 9], "SVM")
cat("SVM Test AUC:", auc_svm, "\n")

########################################
##### 10. MLP
########################################
norm <- data.Normalization(pima_data[, 1:8], type = "n4")
norm <- data.frame(norm, Outcome = pima_data$Outcome)

train_n <- norm[-id, ]
test_n  <- norm[id, ]

x_tr <- train_n[, 1:8]
y_tr <- decodeClassLabels(train_n[, 9])
x_ts <- test_n[, 1:8]

mlp <- mlp(x_tr, y_tr, size = 12, maxit = 300)

pred_tr <- predict(mlp, x_tr)
pred_ts <- predict(mlp, x_ts)

class_tr <- colnames(pred_tr)[max.col(pred_tr)]
class_ts <- colnames(pred_ts)[max.col(pred_ts)]

cat("MLP Train Accuracy:", mean(class_tr == train_n[, 9]), "\n")
cat("MLP Test Accuracy:", mean(class_ts == test_n[, 9]), "\n")

auc_mlp <- roc_plot(pred_ts[, "yes"], test_n[, 9], "MLP")
cat("MLP Test AUC:", auc_mlp, "\n")