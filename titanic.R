############################
# install.packages("klaR")
# install.packages("C50")
# install.packages("rpart")
# install.packages("ROCR")
# install.packages("arules")

library(arules)
library(klaR)
library(C50)
library(rpart)
library(ROCR)

########################################
##### Titanic data preprocessing ########
########################################
titanic_raw <- read.csv("titanic.csv", header = TRUE, sep = ",", stringsAsFactors = TRUE)
titanic_na_removed <- na.exclude(titanic_raw)

invalid_row_id <- c(which(titanic_na_removed$age == 9999),
                    which(titanic_na_removed$fare == 9999))

titanic_clean <- titanic_na_removed[-invalid_row_id, 2:5]
titanic_clean$survival <- titanic_na_removed[-invalid_row_id, 10]

titanic_clean[which(titanic_clean[,1] == 0), 1] <- "male"
titanic_clean[which(titanic_clean[,1] == 1), 1] <- "female"
titanic_clean[,1] <- as.factor(titanic_clean$gender)

titanic_clean[which(titanic_clean[,5] == 1), 5] <- "yes"
titanic_clean[which(titanic_clean[,5] == 0), 5] <- "no"
titanic_clean[,5] <- as.factor(titanic_clean[,"survival"])

summary(titanic_clean)

########################################
##### Discretization ###################
########################################

titanic_assoc <- titanic_clean

char <- rep(NA, nrow(titanic_assoc))
for (i in 1:nrow(titanic_assoc))
{
  if (titanic_assoc[i,2] < 20) char[i] <- "young"
  else if (titanic_assoc[i,2] < 50) char[i] <- "adult"
  else char[i] <- "old"
}
titanic_assoc$age <- factor(char)

# fare 分箱
char <- rep(NA, nrow(titanic_assoc))
for (i in 1:nrow(titanic_assoc))
{
  if (titanic_assoc[i,4] < 14) char[i] <- "low"
  else if (titanic_assoc[i,4] < 33) char[i] <- "medium"
  else char[i] <- "high"
}
titanic_assoc$fare <- factor(char)

summary(titanic_assoc)

########################################
##### Association Rules ################
########################################

# survival = yes / no 都看
rule0 <- apriori(
  titanic_assoc,
  parameter = list(minlen = 2, supp = 0.05, conf = 0.8),
  appearance = list(rhs = c("survival=no", "survival=yes"), default = "lhs")
)

sorted0 <- sort(rule0, by = "lift")
inspect(sorted0)

# 只看 survival = yes
rule1 <- apriori(
  titanic_assoc,
  parameter = list(minlen = 3, supp = 0.05, conf = 0.8),
  appearance = list(rhs = c("survival=yes"), default = "lhs")
)

sorted1 <- sort(rule1, by = "confidence")
inspect(sorted1)

# 只看 survival = no
rule2 <- apriori(
  titanic_assoc,
  parameter = list(minlen = 3, supp = 0.05, conf = 0.8),
  appearance = list(rhs = c("survival=no"), default = "lhs")
)

sorted2 <- sort(rule2, by = "confidence")
inspect(sorted2)

########################################
##### Supervised learning split #########
########################################
set.seed(123)
test_row_id <- sample(1:nrow(titanic_clean), round(0.3 * nrow(titanic_clean)))
titanic_train <- titanic_clean[-test_row_id, ]
titanic_test  <- titanic_clean[test_row_id, ]

################################################
##### Naive Bayes - Titanic ####################
################################################
nb_model <- NaiveBayes(titanic_train[,-5], titanic_train[,5])
names(nb_model)
nb_model$apriori
nb_model$tables

## training accuracy only
nb_train_pred <- predict(nb_model, titanic_train[,-5])
nb_train_confusion <- table(titanic_train[,5], nb_train_pred$class)
acc_nb_train <- sum(diag(nb_train_confusion)) / sum(nb_train_confusion)
acc_nb_train

## testing table + accuracy
nb_test_pred <- predict(nb_model, titanic_test[,-5])
nb_test_confusion <- table(titanic_test[,5], nb_test_pred$class)
nb_test_confusion

acc_nb_test <- sum(diag(nb_test_confusion)) / sum(nb_test_confusion)
acc_nb_test

# ROC - testing only
nb_prob <- as.data.frame(nb_test_pred$posterior)
pred_nb <- prediction(nb_prob$yes, titanic_test$survival)
perf_nb <- performance(pred_nb, measure = "tpr", x.measure = "fpr")
auc_nb <- performance(pred_nb, "auc")
auc_value_nb <- round(auc_nb@y.values[[1]], 3)

plot(perf_nb, col = "blue", main = "ROC curve using Naive Bayes", xlab = "FPR", ylab = "TPR")
abline(0, 1)
text(0.4, 0.6, as.character(auc_value_nb))

################################################
##### Logistic Regression - Titanic ############
################################################
logit_model <- glm(
  formula = survival ~ fare + age + class + gender,
  data = titanic_train,
  family = binomial(link = "logit")
)
summary(logit_model)

## training accuracy only
logit_train_prob <- predict.glm(logit_model, type = "response", newdata = titanic_train)
logit_train_class <- ifelse(logit_train_prob >= 0.5, "yes", "no")
logit_train_confusion <- table(titanic_train$survival, logit_train_class)
acc_logit_train <- sum(diag(logit_train_confusion)) / sum(logit_train_confusion)
acc_logit_train

## testing table + accuracy
logit_test_prob <- predict.glm(logit_model, type = "response", newdata = titanic_test)
logit_test_class <- ifelse(logit_test_prob >= 0.5, "yes", "no")
logit_test_confusion <- table(titanic_test$survival, logit_test_class)
logit_test_confusion

acc_logit_test <- sum(diag(logit_test_confusion)) / sum(logit_test_confusion)
acc_logit_test

# ROC - testing only
pred_logit <- prediction(logit_test_prob, titanic_test$survival)
perf_logit <- performance(pred_logit, measure = "tpr", x.measure = "fpr")
auc_logit <- performance(pred_logit, "auc")
auc_value_logit <- round(auc_logit@y.values[[1]], 3)

plot(perf_logit, col = "green", main = "ROC curve using Logistic Regression", xlab = "FPR", ylab = "TPR")
abline(0, 1)
text(0.4, 0.6, as.character(auc_value_logit))

################################################
##### C5.0 - Titanic ###########################
################################################
c50_model <- C5.0(
  titanic_train[,-5],
  titanic_train[,5],
  rules = FALSE,
  control = C5.0Control(minCases = 10)
)
summary(c50_model)
plot(c50_model, main = "Titanic C5.0 Tree")

## training accuracy only
c50_train_pred <- predict(c50_model, titanic_train, type = "class")
c50_train_confusion <- table(titanic_train$survival, c50_train_pred)
acc_c50_train <- sum(diag(c50_train_confusion)) / sum(c50_train_confusion)
acc_c50_train

## testing table + accuracy
c50_test_pred <- predict(c50_model, titanic_test, type = "class")
c50_test_confusion <- table(titanic_test$survival, c50_test_pred)
c50_test_confusion

acc_c50_test <- sum(diag(c50_test_confusion)) / sum(c50_test_confusion)
acc_c50_test

# ROC - testing only
c50_prob <- predict(c50_model, titanic_test, type = "prob")
c50_prob <- as.data.frame(c50_prob)
pred_c50 <- prediction(c50_prob$yes, titanic_test$survival)
perf_c50 <- performance(pred_c50, measure = "tpr", x.measure = "fpr")
auc_c50 <- performance(pred_c50, "auc")
auc_value_c50 <- round(auc_c50@y.values[[1]], 3)

plot(perf_c50, col = "purple", main = "ROC curve using C5.0", xlab = "FPR", ylab = "TPR")
abline(0, 1)
text(0.4, 0.6, as.character(auc_value_c50))

################################################
##### CART - Titanic ###########################
################################################
cart_model <- rpart(
  survival ~ .,
  data = titanic_train,
  method = "class",
  minsplit = 15,
  minbucket = 15
)

## training accuracy only
cart_train_prob <- predict(cart_model, newdata = titanic_train, type = "prob")
cart_train_prob <- as.data.frame(cart_train_prob)
cart_train_class <- ifelse(cart_train_prob$yes > 0.5, "yes", "no")
cart_train_confusion <- table(titanic_train$survival, cart_train_class)
acc_cart_train <- sum(diag(cart_train_confusion)) / sum(cart_train_confusion)
acc_cart_train

## testing table + accuracy
cart_test_prob <- predict(cart_model, newdata = titanic_test, type = "prob")
cart_test_prob <- as.data.frame(cart_test_prob)
cart_test_class <- ifelse(cart_test_prob$yes > 0.5, "yes", "no")
cart_test_confusion <- table(titanic_test$survival, cart_test_class)
cart_test_confusion

acc_cart_test <- sum(diag(cart_test_confusion)) / sum(cart_test_confusion)
acc_cart_test

# ROC - testing only
pred_cart <- prediction(cart_test_prob$yes, titanic_test$survival)
perf_cart <- performance(pred_cart, measure = "tpr", x.measure = "fpr")
auc_cart <- performance(pred_cart, "auc")
auc_value_cart <- round(auc_cart@y.values[[1]], 3)

plot(perf_cart, col = "red", main = "ROC curve using CART", xlab = "FPR", ylab = "TPR")
abline(0, 1)
text(0.4, 0.6, as.character(auc_value_cart))

################################################
##### Compare all ROC curves ###################
################################################
plot(perf_nb, col = "blue", lwd = 5, main = "ROC curve", xlab = "FPR", ylab = "TPR")
leg_nb <- paste("Naive Bayes, AUC = ", auc_value_nb)

plot(perf_logit, add = TRUE, col = "green", lwd = 5)
leg_logit <- paste("Logistic Regression, AUC = ", auc_value_logit)

plot(perf_c50, add = TRUE, col = "purple", lwd = 5)
leg_c50 <- paste("C5.0, AUC = ", auc_value_c50)

plot(perf_cart, add = TRUE, col = "red", lwd = 5)
leg_cart <- paste("CART, AUC = ", auc_value_cart)

abline(0, 1, lwd = 3)
legend(
  "bottomright",
  legend = c(leg_nb, leg_logit, leg_c50, leg_cart, "baseline"),
  col = c("blue", "green", "purple", "red", "black"),
  lty = 1,
  lwd = c(5, 5, 5, 5, 3),
  cex = 0.5,        
  bty = "n",        
  seg.len = 2,      
  y.intersp = 1.2   
)

################################################
##### Final comparison table ###################
################################################
table <- array(0, c(2, 4))
rownames(table) <- c("Accuracy", "AUC")
colnames(table) <- c("Naive Bayes", "Logistic", "C5.0", "CART")

table[1, 1] <- acc_nb
table[2, 1] <- auc_value_nb

table[1, 2] <- acc_logit
table[2, 2] <- auc_value_logit

table[1, 3] <- acc_c50
table[2, 3] <- auc_value_c50

table[1, 4] <- acc_cart
table[2, 4] <- auc_value_cart

table