############################
## 0) Packages
############################

# install.packages("ROCR")
# install.packages("randomForest")
# install.packages("adabag")
# install.packages("gbm")
# install.packages("mlr")
# install.packages("xgboost")
# install.packages("caret")

library(ROCR)
library(randomForest)
library(adabag)
library(gbm)
library(mlr)
library(caret)

############################
## 1) Read data
############################
bank_raw <- read.csv("bank.csv", header = TRUE, sep = ";", stringsAsFactors = TRUE)

############################
## 2) Statistical analysis data
############################
bank_stat <- bank_raw
summary(bank_stat)
hist(bank_stat$age)

############################
## 3) Age grouping
############################
bank_stat$agg <- ifelse(bank_stat$age < 35, "young",
                        ifelse(bank_stat$age < 50, "middle", "old"))
bank_stat$agg <- factor(bank_stat$agg)
summary(bank_stat$agg)

############################
## 4) Chi-square / Proportion / T-test
############################

### age vs. loan
young_y  <- sum((bank_stat$agg == "young")  & (bank_stat$loan == "yes"))
young_n  <- sum((bank_stat$agg == "young")  & (bank_stat$loan == "no"))
middle_y <- sum((bank_stat$agg == "middle") & (bank_stat$loan == "yes"))
middle_n <- sum((bank_stat$agg == "middle") & (bank_stat$loan == "no"))
old_y    <- sum((bank_stat$agg == "old")    & (bank_stat$loan == "yes"))
old_n    <- sum((bank_stat$agg == "old")    & (bank_stat$loan == "no"))

age_y <- c(young = young_y, middle = middle_y, old = old_y)
age_n <- c(young = young_n, middle = middle_n, old = old_n)
rbind(age_y, age_n)
chisq.test(rbind(age_y, age_n))

### marital status vs. loan
mar_y <- c(
  single   = nrow(subset(bank_stat, marital == "single"   & loan == "yes")),
  married  = nrow(subset(bank_stat, marital == "married"  & loan == "yes")),
  divorced = nrow(subset(bank_stat, marital == "divorced" & loan == "yes"))
)

mar_n <- c(
  single   = nrow(subset(bank_stat, marital == "single"   & loan == "no")),
  married  = nrow(subset(bank_stat, marital == "married"  & loan == "no")),
  divorced = nrow(subset(bank_stat, marital == "divorced" & loan == "no"))
)

rbind(mar_y, mar_n)
chisq.test(rbind(mar_y, mar_n))

### married vs. unmarried on loan
married_y <- sum((bank_stat$marital == "married") & (bank_stat$loan == "yes"))
unmarried_y <- sum((bank_stat$marital != "married") & (bank_stat$loan == "yes"))
prop_x <- c(married_y, unmarried_y)
prop_n <- c(sum(bank_stat$marital == "married"),
            sum(bank_stat$marital != "married"))
prop.test(prop_x, prop_n)

### single vs. non-single on balance
single_grp <- subset(bank_stat, marital == "single")
nonsingle_grp <- subset(bank_stat, marital != "single")

var.test(single_grp$balance, nonsingle_grp$balance)

t.test(single_grp$balance, nonsingle_grp$balance, var.equal = FALSE)

############################
## 5) ANOVA / Regression
############################

## (1) marital
fit_m1 <- lm(balance ~ marital, data = bank_stat)
summary(fit_m1)

ano_m1 <- aov(balance ~ marital, data = bank_stat)
summary(ano_m1)

## (2) marital + agg
fit_m2 <- lm(balance ~ marital + agg, data = bank_stat)
summary(fit_m2)

ano_m2 <- aov(balance ~ marital + agg, data = bank_stat)
summary(ano_m2)

## (3) marital + age
fit_m3 <- lm(balance ~ marital + age, data = bank_stat)
summary(fit_m3)

## (4) marital + agg + interaction
fit_m4 <- lm(balance ~ marital + agg + marital * agg, data = bank_stat)
summary(fit_m4)

ano_m4 <- aov(balance ~ marital + agg + marital * agg, data = bank_stat)
summary(ano_m4)

interaction.plot(
  bank_stat$agg,
  bank_stat$marital,
  bank_stat$balance,
  col = 1:3
)

############################
## 6) Machine learning data
############################
bank_ml <- bank_raw[, -c(2, 10, 11)]

set.seed(123)
test_id <- sample(1:nrow(bank_ml), round(nrow(bank_ml) / 3))
tr.bank <- bank_ml[-test_id, ]
ts.bank <- bank_ml[test_id, ]

############################
## 7) Random Forest
############################
rf_mtry <- c(6, 8, 10, 12)
rf_ntree <- c(60, 80, 100)
rf_nodesize <- c(10, 15, 20)

tune.rf <- tune.randomForest(
  x = tr.bank[, -14],
  y = tr.bank[, 14],
  mtry = rf_mtry,
  ntree = rf_ntree,
  nodesize = rf_nodesize
)

tune.rf$best.model
tune.rf$best.parameters

rf.bank <- randomForest(
  y ~ .,
  data = tr.bank,
  nodesize = tune.rf$best.parameters$nodesize,
  mtry = tune.rf$best.parameters$mtry,
  ntree = tune.rf$best.parameters$ntree,
  importance = TRUE,
  proximity = TRUE
)

importance(rf.bank)
plot(rf.bank)
print(rf.bank)

rf.bk.ts <- predict(rf.bank, newdata = ts.bank, type = "response")
table(ts.bank$y, rf.bk.ts)
acc_rf <- sum(as.numeric(rf.bk.ts == ts.bank$y)) / nrow(ts.bank)
acc_rf

# ROC
rf.prob.ts <- predict(rf.bank, newdata = ts.bank, type = "prob")
pred_rf <- prediction(rf.prob.ts[, "yes"], ts.bank$y)
perf_rf <- ROCR::performance(pred_rf, measure = "tpr", x.measure = "fpr")
auc_rf <- ROCR::performance(pred_rf, "auc")
plot(perf_rf, col = "red", main = "ROC curve using random forest", xlab = "FPR", ylab = "TPR")
abline(0, 1)
auc_value_rf <- round(auc_rf@y.values[[1]], 3)
text(0.4, 0.6, as.character(auc_value_rf))

############################
## 8) Bagging
############################
bag_mfinal <- c(60, 80, 100)
bag_acc <- numeric(length(bag_mfinal))

for (i in 1:length(ada_mfinal))
{
  bag.bank <- bagging(y ~ ., tr.bank, mfinal = bag_mfinal[i])
  bag_pred_tr <- predict(bag.bank, tr.bank)
  bag_acc[i] <- sum(as.numeric(bag_pred_tr$class == tr.bank$y)) / nrow(tr.bank)
}

plot(bag_mfinal, bag_acc, xlab = "mfinal", ylab = "acuracy")
best_i <- which.max(bag_acc)

bag.bk <- bagging(y ~ ., tr.bank, mfinal = bag_mfinal[best_i])
bag.bk
bag.bk$importance

bg.bk <- predict(bag.bk, ts.bank)
bg.bk$confusion
acc_bg <- sum(as.numeric(bg.bk$class == ts.bank$y)) / nrow(ts.bank)
acc_bg

# ROC
bg.prob.ts <- predict(bag.bk, ts.bank)$prob
pred_bg <- prediction(bg.prob.ts[, 2], ts.bank$y)
perf_bg <- ROCR::performance(pred_bg, measure = "tpr", x.measure = "fpr")
auc_bg <- ROCR::performance(pred_bg, "auc")
plot(perf_bg, col = "red", main = "ROC curve using bagging", xlab = "FPR", ylab = "TPR")
abline(0, 1)
auc_value_bg <- round(auc_bg@y.values[[1]], 3)
text(0.4, 0.6, as.character(auc_value_bg))

############################
## 9) AdaBoost
############################
ada_mfinal <- c(60, 80, 100)
ada_acc <- numeric(length(ada_mfinal))

for (j in 1:length(ada_mfinal))
{
  boo.bank <- boosting(y ~ ., tr.bank, mfinal = ada_mfinal[j])
  boo_pred_tr <- predict(boo.bank, tr.bank)
  ada_acc[j] <- sum(as.numeric(boo_pred_tr$class == tr.bank$y)) / nrow(tr.bank)
}

plot(ada_mfinal, ada_acc, xlab = "mfinal", ylab = "acuracy")
best_j <- which.max(ada_acc)

boo.bk <- boosting(y ~ ., tr.bank, mfinal = ada_mfinal[best_j])
boo.bk$importance

bt.bk <- predict(boo.bk, ts.bank)
bt.bk$confusion
acc_boo <- sum(as.numeric(bt.bk$class == ts.bank$y)) / nrow(ts.bank)
acc_boo

# ROC
boo.prob.ts <- predict(boo.bk, ts.bank)$prob
pred_boo <- prediction(boo.prob.ts[, 2], ts.bank$y)
perf_boo <- ROCR::performance(pred_boo, measure = "tpr", x.measure = "fpr")
auc_boo <- ROCR::performance(pred_boo, "auc")
plot(perf_boo, col = "red", main = "ROC curve using adaboost", xlab = "FPR", ylab = "TPR")
abline(0, 1)
auc_value_boo <- round(auc_boo@y.values[[1]], 3)
text(0.4, 0.6, as.character(auc_value_boo))

############################
## 10) Gradient Boosting
############################
numfolds_gb <- trainControl(method = "cv", number = 3)
Grid_gb <- expand.grid(
  n.trees = seq(60, 100, 20),
  interaction.depth = c(4, 6, 8),
  shrinkage = c(0.01, 0.05, 0.1),
  n.minobsinnode = c(10, 15, 20)
)

cv_gb <- caret::train(
  y ~ .,
  data = tr.bank,
  method = "gbm",
  trControl = numfolds_gb,
  tuneGrid = Grid_gb,
  train.fraction = 0.5
)

cv_gb
confusionMatrix(cv_gb, norm = "none")
cv_gb$bestTune

gb.bk <- gbm(
  y ~ .,
  data = tr.bank,
  distribution = "multinomial",
  n.trees = cv_gb$bestTune$n.trees,
  interaction.depth = cv_gb$bestTune$interaction.depth,
  shrinkage = cv_gb$bestTune$shrinkage,
  n.minobsinnode = cv_gb$bestTune$n.minobsinnode
)
summary(gb.bk)

gb.bk.prets <- predict(
  gb.bk,
  newdata = ts.bank,
  n.trees = cv_gb$bestTune$n.trees,
  type = "response"
)
gb.bk.prets <- as.data.frame(gb.bk.prets)
colnames(gb.bk.prets) <- c("no", "yes")
gb.bk.prets.y <- colnames(gb.bk.prets)[apply(gb.bk.prets, 1, which.max)]

table(ts.bank$y, gb.bk.prets.y)
acc_gb <- sum(as.numeric(gb.bk.prets.y == ts.bank$y)) / nrow(ts.bank)
acc_gb

# ROC
pred_gb <- prediction(gb.bk.prets$yes, ts.bank$y)
perf_gb <- ROCR::performance(pred_gb, measure = "tpr", x.measure = "fpr")
auc_gb <- ROCR::performance(pred_gb, "auc")
plot(perf_gb, col = "red", main = "ROC curve using gradient boosting", xlab = "FPR", ylab = "TPR")
abline(0, 1)
auc_value_gb <- round(auc_gb@y.values[[1]], 3)
text(0.4, 0.6, as.character(auc_value_gb))

############################
## 11) XGBoost
############################

## dummy variables
dmy_xgb <- dummyVars(" ~ .", data = tr.bank[, -14], fullRank = TRUE)

Dtr_xgb_x <- data.frame(predict(dmy_xgb, newdata = tr.bank))
Dtr_xgb <- cbind(Dtr_xgb_x, y = tr.bank[, 14])
Dtr_xgb$y <- factor(Dtr_xgb$y, levels = c("no", "yes"), labels = c(0, 1))

Dts_xgb_x <- data.frame(predict(dmy_xgb, newdata = ts.bank))
Dts_xgb <- cbind(Dts_xgb_x, y = ts.bank[, 14])
Dts_xgb$y <- factor(Dts_xgb$y, levels = c("no", "yes"), labels = c(0, 1))

## create task
tr.bank.xgb <- makeClassifTask(data = Dtr_xgb, target = "y", positive = 1)
ts.bank.xgb <- makeClassifTask(data = Dts_xgb, target = "y", positive = 1)

lrn.bank <- makeLearner("classif.xgboost", predict.type = "prob")
lrn.bank$par.vals <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "error",
  nrounds = 100
)

params <- makeParamSet(
  makeIntegerParam("max_depth", lower = 2, upper = 10),
  makeNumericParam("eta", lower = 0.01, upper = 0.05),
  makeNumericParam("gamma", lower = 0.05, upper = 0.1),
  makeNumericParam("subsample", lower = 0.5, upper = 1),
  makeNumericParam("lambda", lower = 1, upper = 10),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 1)
)

mytune <- tuneParams(
  learner = lrn.bank,
  task = tr.bank.xgb,
  measures = acc,
  par.set = params,
  show.info = TRUE,
  resampling = makeResampleDesc("CV", stratify = TRUE, iters = 5),
  control = makeTuneControlRandom(maxit = 10)
)

lrn_tune <- setHyperPars(lrn.bank, par.vals = mytune$x)

xgb.bank <- mlr::train(learner = lrn_tune, task = tr.bank.xgb)
getFeatureImportance(xgb.bank)$res

bankts.xgb <- predict(xgb.bank, ts.bank.xgb)

conf.ts <- table(bankts.xgb$data$truth, bankts.xgb$data$response)
row.names(conf.ts) <- c("no", "yes")
colnames(conf.ts) <- c("no", "yes")
print(conf.ts)
acc_xgb <- sum(as.numeric(bankts.xgb$data$response == bankts.xgb$data$truth)) / nrow(bankts.xgb$data)
acc_xgb

# ROC
pred_xgb <- prediction(bankts.xgb$data$prob.1, ts.bank$y)
perf_xgb <- ROCR::performance(pred_xgb, measure = "tpr", x.measure = "fpr")
auc_xgb <- ROCR::performance(pred_xgb, "auc")
plot(perf_xgb, col = "red", main = "ROC curve using eXtreme gradient boosting", xlab = "FPR", ylab = "TPR")
abline(0, 1)
auc_value_xgb <- round(auc_xgb@y.values[[1]], 3)
text(0.4, 0.6, as.character(auc_value_xgb))

############################
## 12) Compare all models
############################
table <- array(0, c(2, 5))
rownames(table) <- c("Accuracy", "AUC")
colnames(table) <- c("RF", "Bagging", "AdaBoost", "GBM", "XGBoost")

table[1, 1] <- acc_rf
table[2, 1] <- auc_value_rf

table[1, 2] <- acc_bg
table[2, 2] <- auc_value_bg

table[1, 3] <- acc_boo
table[2, 3] <- auc_value_boo

table[1, 4] <- acc_gb
table[2, 4] <- auc_value_gb

table[1, 5] <- acc_xgb
table[2, 5] <- auc_value_xgb

table