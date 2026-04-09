#install.packages("class")
#install.packages("klaR")
#install.packages("neuralnet")
library(class)
library(klaR)
library(neuralnet)

############################################
############### Data loading ###############
############################################
cancer_data <- read.csv("breast_cancer.csv", header = TRUE, sep = ",", stringsAsFactors = TRUE)
cancer_data <- cancer_data[cancer_data$Bare.Nuclei != 9999, 2:11]
cancer_data <- na.omit(cancer_data)

m <- ncol(cancer_data)
n <- nrow(cancer_data)

############################################
########### 3-fold Cross Validation ########
############################################
cv <- 3
num <- floor(n / cv)

table.KNN <- array(0, c(4, 4))
colnames(table.KNN) <- c("Recall", "Precision", "F_measure", "Accuracy")
rownames(table.KNN) <- c("1st", "2nd", "3rd", "Avg")

table.NB <- array(0, c(4, 4))
colnames(table.NB) <- c("Recall", "Precision", "F_measure", "Accuracy")
rownames(table.NB) <- c("1st", "2nd", "3rd", "Avg")

table.LR <- array(0, c(4, 4))
colnames(table.LR) <- c("Recall", "Precision", "F_measure", "Accuracy")
rownames(table.LR) <- c("1st", "2nd", "3rd", "Avg")

table.ANN <- array(0, c(4, 4))
colnames(table.ANN) <- c("Recall", "Precision", "F_measure", "Accuracy")
rownames(table.ANN) <- c("1st", "2nd", "3rd", "Avg")

############################################
#################### KNN ###################
############################################
for (i in 1:cv) {
  start_idx <- (i - 1) * num + 1
  end_idx <- i * num
  
  if (i < cv) {
    ts.id <- start_idx:end_idx
  } else {
    ts.id <- start_idx:n
  }
  
  ts.cancer <- cancer_data[ts.id, ]
  tr.cancer <- cancer_data[-ts.id, ]
  
  cancer_knn <- knn(tr.cancer[, -m], ts.cancer[, -m], cl = tr.cancer[, m], k = 3)
  conf.knn <- table(ts.cancer$Class, cancer_knn)
  
  table.KNN[i, 1] <- conf.knn[2, 2] / sum(conf.knn[2, ])
  table.KNN[i, 2] <- conf.knn[2, 2] / sum(conf.knn[, 2])
  table.KNN[i, 3] <- 2 * table.KNN[i, 1] * table.KNN[i, 2] / (table.KNN[i, 1] + table.KNN[i, 2])
  table.KNN[i, 4] <- sum(diag(conf.knn)) / sum(conf.knn)
}

for (metric_idx in 1:4) {
  table.KNN[cv + 1, metric_idx] <- mean(table.KNN[1:cv, metric_idx])
}
print(table.KNN)

############################################
################ Naive Bayes ###############
############################################
for (i in 1:cv) {
  start_idx <- (i - 1) * num + 1
  end_idx <- i * num
  
  if (i < cv) {
    ts.id <- start_idx:end_idx
  } else {
    ts.id <- start_idx:n
  }
  
  ts.cancer <- cancer_data[ts.id, ]
  tr.cancer <- cancer_data[-ts.id, ]
  
  cancer_nb <- NaiveBayes(Class ~ ., data = tr.cancer)
  cancer_nb.pred <- predict(cancer_nb, ts.cancer)$class
  conf.nb <- table(ts.cancer$Class, cancer_nb.pred)
  
  table.NB[i, 1] <- conf.nb[2, 2] / sum(conf.nb[2, ])
  table.NB[i, 2] <- conf.nb[2, 2] / sum(conf.nb[, 2])
  table.NB[i, 3] <- 2 * table.NB[i, 1] * table.NB[i, 2] / (table.NB[i, 1] + table.NB[i, 2])
  table.NB[i, 4] <- sum(diag(conf.nb)) / sum(conf.nb)
}

for (metric_idx in 1:4) {
  table.NB[cv + 1, metric_idx] <- mean(table.NB[1:cv, metric_idx])
}
print(table.NB)

############################################
############ Logistic Regression ###########
############################################
for (i in 1:cv) {
  start_idx <- (i - 1) * num + 1
  end_idx <- i * num
  
  if (i < cv) {
    ts.id <- start_idx:end_idx
  } else {
    ts.id <- start_idx:n
  }
  
  ts.cancer <- cancer_data[ts.id, ]
  tr.cancer <- cancer_data[-ts.id, ]
  
  cancer_lr <- glm(Class ~ ., family = binomial(link = "logit"), data = tr.cancer)
  lr.prob <- predict.glm(cancer_lr, type = "response", newdata = ts.cancer)
  cancer_lr.pred <- ifelse(lr.prob >= 0.5, "malignant", "benign")
  conf.lr <- table(ts.cancer$Class, cancer_lr.pred)
  
  table.LR[i, 1] <- conf.lr[2, 2] / sum(conf.lr[2, ])
  table.LR[i, 2] <- conf.lr[2, 2] / sum(conf.lr[, 2])
  table.LR[i, 3] <- 2 * table.LR[i, 1] * table.LR[i, 2] / (table.LR[i, 1] + table.LR[i, 2])
  table.LR[i, 4] <- sum(diag(conf.lr)) / sum(conf.lr)
}

for (metric_idx in 1:4) {
  table.LR[cv + 1, metric_idx] <- mean(table.LR[1:cv, metric_idx])
}
print(table.LR)

############################################
#################### ANN ###################
############################################
for (i in 1:cv) {
  start_idx <- (i - 1) * num + 1
  end_idx <- i * num
  
  if (i < cv) {
    ts.id <- start_idx:end_idx
  } else {
    ts.id <- start_idx:n
  }
  
  ts.cancer <- cancer_data[ts.id, ]
  tr.cancer <- cancer_data[-ts.id, ]
  
  tr.ann <- data.frame(tr.cancer[, 1:9], Y = ifelse(tr.cancer$Class == "benign", 1, 2))
  ts.ann <- data.frame(ts.cancer[, 1:9], Y = ifelse(ts.cancer$Class == "benign", 1, 2))
  
  colnames(tr.ann) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "Y")
  colnames(ts.ann) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "Y")
  
  trnet <- neuralnet(Y ~ ., hidden = 6, data = tr.ann)
  
  ann.pred.raw <- compute(trnet, ts.ann[, 1:9])$net.result
  ann.pred.num <- round(ann.pred.raw)
  ann.pred.num[ann.pred.num < 1] <- 1
  ann.pred.num[ann.pred.num > 2] <- 2
  
  name <- c("benign", "malignant")
  cancer_ann.pred <- name[ann.pred.num]
  conf.ann <- table(ts.cancer$Class, cancer_ann.pred)
  
  table.ANN[i, 1] <- conf.ann[2, 2] / sum(conf.ann[2, ])
  table.ANN[i, 2] <- conf.ann[2, 2] / sum(conf.ann[, 2])
  table.ANN[i, 3] <- 2 * table.ANN[i, 1] * table.ANN[i, 2] / (table.ANN[i, 1] + table.ANN[i, 2])
  table.ANN[i, 4] <- sum(diag(conf.ann)) / sum(conf.ann)
}

for (metric_idx in 1:4) {
  table.ANN[cv + 1, metric_idx] <- mean(table.ANN[1:cv, metric_idx])
}
print(table.ANN)

############################################
########### Final Avg Comparison ###########
############################################
final.comparison <- rbind(
  KNN = table.KNN["Avg", ],
  Naive_Bayes = table.NB["Avg", ],
  Logistic_Regression = table.LR["Avg", ],
  ANN = table.ANN["Avg", ]
)

print(final.comparison)