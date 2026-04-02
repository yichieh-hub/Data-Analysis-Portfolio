############################
# 1 packages  #
############################
# install.packages("timeDate")
# install.packages("nortest")
# install.packages("car")
# install.packages("cluster")
# install.packages("clusterSim")
# install.packages("NbClust")
# install.packages("e1071")
# install.packages("klaR")
# install.packages("mclust")
# install.packages("isotree")
# install.packages("class")
# install.packages("ROCR")

library(timeDate)
library(nortest)
library(car)
library(cluster)
library(clusterSim)
library(NbClust)
library(e1071)
library(klaR)
library(mclust)
library(isotree)
library(class)
library(ROCR)

#############
# 2. Data   #
#############
gender_data <- read.csv("gender_size.csv", stringsAsFactors = TRUE)
gender_outlier_data <- read.csv("gender_outlier.csv", stringsAsFactors = TRUE)

############################
# 3. Basic Data Structure  #
############################
attributes(gender_data)
str(gender_data)
summary(gender_data)
head(gender_data, 5)

###########################
# 4. Descriptive Analysis #
###########################
colMeans(gender_data[, 1:3])

cor(gender_data$Height, gender_data$Weight)
gender_corr <- cor(gender_data[, 1:3], use = "pairwise")
gender_corr

cov(gender_data$Height, gender_data$Weight)
gender_cov <- cov(gender_data[, 1:3], use = "pairwise")
gender_cov

skewness(gender_data[, 1:3])
kurtosis(gender_data[, 1:3])

####################
# 5. Visualization #
####################
plot(gender_data)

boxplot(gender_data[, 1:3], main = "gender_size")

male_id <- which(gender_data$Gender == "male")
female_id <- which(gender_data$Gender == "female")

boxplot(gender_data[male_id, 1:3], main = "male")
boxplot(gender_data[female_id, 1:3], main = "female")

gender_count <- with(
  gender_data,
  c(sum(Gender == "male"), sum(Gender == "female"))
)

barplot(gender_count, names.arg = c("male", "female"), xlab = "Gender", ylab = "Number")
pie(gender_count, labels = c("male", "female"))

gender_percent <- round(gender_count / sum(gender_count) * 100)
gender_label <- paste(c("male", "female"), gender_percent, "%")
pie(gender_count, gender_label)

plot(
  gender_data$Height[gender_data$Gender == "male"],
  gender_data$Weight[gender_data$Gender == "male"],
  pch = 1, col = "blue",
  xlim = c(min(gender_data$Height), max(gender_data$Height)),
  ylim = c(min(gender_data$Weight), max(gender_data$Weight)),
  main = "Height vs Weight",
  xlab = "Height",
  ylab = "Weight"
)
points(
  gender_data$Height[gender_data$Gender == "female"],
  gender_data$Weight[gender_data$Gender == "female"],
  pch = 2, col = "red"
)
legend("topleft", c("male", "female"), col = c("blue", "red"), pch = c(1, 2))

hist(gender_data$Height, breaks = 20, labels = TRUE, col = "blue", border = "red", main = "Histogram of Height")
hist(gender_data$Height, freq = FALSE, main = "Histogram of Height Density")
lines(density(gender_data$Height))

height_ecdf <- ecdf(gender_data$Height)
plot(height_ecdf, xlab = "Height", main = "Cumulative Frequency of Height")

################################
# 6. Normality / Statistical   #
################################
hist(gender_data$Height, breaks = seq(min(gender_data$Height), max(gender_data$Height), 0.5))
hist(gender_data$Height, breaks = seq(min(gender_data$Height), max(gender_data$Height), 1), prob = TRUE)
qqnorm(gender_data$Height, xlab = "Z-score", ylab = "Height")
qqline(gender_data$Height, col = "red")
curve(
  dnorm(x, mean(gender_data$Height), sd(gender_data$Height)),
  min(gender_data$Height), max(gender_data$Height), col = "red"
)

shapiro.test(gender_data$Height)
ad.test(gender_data$Height)
sf.test(gender_data$Height)
cvm.test(gender_data$Height)

qqplot(gender_data$Height, gender_data$Weight)
qqplot(gender_data$Height, gender_data$Waist)

#################################
# 7. T-test / Variance Test     #
#################################
male_data <- subset(gender_data, Gender == "male")
female_data <- subset(gender_data, Gender == "female")

var.test(male_data$Height, female_data$Height)
t.test(male_data$Height, female_data$Height, var.equal = FALSE)

var.test(male_data$Weight, female_data$Weight)
t.test(male_data$Weight, female_data$Weight, var.equal = FALSE)

var.test(male_data$Waist, female_data$Waist)
t.test(male_data$Waist, female_data$Waist, var.equal = FALSE)

gender_height_split <- split(gender_data$Height, gender_data$Gender)
gender_weight_split <- split(gender_data$Weight, gender_data$Gender)
gender_waist_split <- split(gender_data$Waist, gender_data$Gender)

t.test(gender_height_split$male, mu = 170, alternative = "greater")
t.test(gender_height_split$male, mu = 175, alternative = "greater", conf.level = 0.95)

t.test(gender_weight_split$female, mu = 50, alternative = "less")
t.test(gender_weight_split$female, mu = 50, alternative = "two.sided", conf.level = 0.95)

var.test(gender_waist_split$male, gender_waist_split$female)
t.test(gender_waist_split$male, gender_waist_split$female, alternative = "greater", var.equal = FALSE)
t.test(gender_waist_split$male, gender_waist_split$female, alternative = "less", var.equal = FALSE)

##########################
# 8. Collinearity Check  #
##########################
gender_lm_vif <- lm(Waist ~ Height + Weight, data = gender_data)
vif(gender_lm_vif)

########################
# 9. Outlier Detection #
########################
male_outlier_data <- subset(gender_outlier_data, Gender == "male")
female_outlier_data <- subset(gender_outlier_data, Gender == "female")

male_mean <- colMeans(male_outlier_data[, -4])
male_mean
female_mean <- colMeans(female_outlier_data[, -4])
female_mean

male_cov <- cov(male_outlier_data[, 1:3], use = "pairwise")
male_cov
female_cov <- cov(female_outlier_data[, 1:3], use = "pairwise")
female_cov

male_outlier_data$mahalanobis_distance <- mahalanobis(male_outlier_data[, -4], male_mean, male_cov)
female_outlier_data$mahalanobis_distance <- mahalanobis(female_outlier_data[, -4], female_mean, female_cov)

male_outlier_data$mahalanobis_outlier <- male_outlier_data$mahalanobis_distance > qchisq(df = 3, p = 0.95)
female_outlier_data$mahalanobis_outlier <- female_outlier_data$mahalanobis_distance > qchisq(df = 3, p = 0.95)

which(male_outlier_data$mahalanobis_outlier == TRUE)
which(female_outlier_data$mahalanobis_outlier == TRUE)

male_iso_model <- isolation.forest(male_outlier_data[, -4], ntrees = 100, nthreads = -1)
male_iso_score <- predict(male_iso_model, male_outlier_data[, -4])
male_outlier_data[which.max(male_iso_score), ]
male_outlier_data[which(male_iso_score > 0.6), ]

female_iso_model <- isolation.forest(female_outlier_data[, -4], ntrees = 100, nthreads = -1)
female_iso_score <- predict(female_iso_model, female_outlier_data[, -4])
female_outlier_data[which.max(female_iso_score), ]
female_outlier_data[which(female_iso_score > 0.6), ]

gender_outlier_lm <- lm(Waist ~ Height + Weight, gender_outlier_data[, -4])
summary(gender_outlier_lm)
outlierTest(gender_outlier_lm)
which(as.vector(hatvalues(gender_outlier_lm)) > (2 * 3 / 146))

#################
# 10. Clustering
#################
min_cluster <- 2
max_cluster <- 8
cluster_range <- min_cluster:max_cluster

##########
# K-means
##########
kmeans_index <- array(0, c(max_cluster - min_cluster + 1, 2))
for (cluster_n in cluster_range)
{
  kmeans_fit <- kmeans(gender_data[, -4], centers = cluster_n)
  kmeans_index[cluster_n - (min_cluster - 1), 1] <- kmeans_fit$betweenss / kmeans_fit$tot.withinss
  kmeans_index[cluster_n - (min_cluster - 1), 2] <- index.DB(
    gender_data[, -4],
    kmeans_fit$cluster,
    centrotypes = "centroids",
    p = 2
  )$DB
}

kmeans_best_ratio <- which(kmeans_index[, 1] == max(kmeans_index[, 1])) + (min_cluster - 1)
kmeans_best_db <- which(kmeans_index[, 2] == min(kmeans_index[, 2])) + (min_cluster - 1)

nbclust_kmeans <- NbClust(
  gender_data[, 1:3],
  distance = "euclidean",
  min.nc = min_cluster,
  max.nc = max_cluster,
  method = "kmeans",
  index = "all"
)
kmeans_best_nbclust <- as.numeric(names(sort(table(nbclust_kmeans$Best.nc[1, ]), decreasing = TRUE)[1]))

kmeans_vote <- c(kmeans_best_ratio, kmeans_best_db, kmeans_best_nbclust)
kmeans_final_k <- as.numeric(names(sort(table(kmeans_vote), decreasing = TRUE)[1]))
kmeans_final_k

kmeans_result <- kmeans(gender_data[, -4], centers = kmeans_final_k)
print(kmeans_result)
kmeans_result$centers
table(gender_data$Gender, kmeans_result$cluster)

plot(gender_data[, 1:2], pch = kmeans_result$cluster, col = kmeans_result$cluster)
points(kmeans_result$centers[, 1:2], col = 1:kmeans_final_k, pch = 8)

plot(gender_data[, 2:3], pch = kmeans_result$cluster, col = kmeans_result$cluster)
points(kmeans_result$centers[, 2:3], col = 1:kmeans_final_k, pch = 8)

#############
# K-medoids
#############
pam_index <- array(0, c(max_cluster - min_cluster + 1, 2))
for (cluster_n in cluster_range)
{
  pam_fit <- pam(gender_data[, -4], cluster_n)
  pam_index[cluster_n - (min_cluster - 1), 1] <- index.DB(
    gender_data[, -4],
    pam_fit$clustering,
    centrotypes = "centroids",
    p = 2
  )$DB
  pam_index[cluster_n - (min_cluster - 1), 2] <- index.DB(
    gender_data[, -4],
    pam_fit$clustering,
    dist(gender_data[, -4]),
    centrotypes = "medoids",
    p = 2
  )$DB
}

pam_best_centroid <- which(pam_index[, 1] == min(pam_index[, 1])) + (min_cluster - 1)
pam_best_medoid <- which(pam_index[, 2] == min(pam_index[, 2])) + (min_cluster - 1)

pam_vote <- c(pam_best_centroid, pam_best_medoid)
pam_final_k <- as.numeric(names(sort(table(pam_vote), decreasing = TRUE)[1]))
pam_final_k

pam_result <- pam(gender_data[, -4], pam_final_k)
print(pam_result)
pam_result$medoids
table(gender_data$Gender, pam_result$cluster)

plot(gender_data[, 1:2], pch = pam_result$cluster, col = pam_result$cluster)
points(pam_result$medoids[, 1:2], col = 1:pam_final_k, pch = 8)

plot(gender_data[, 2:3], pch = pam_result$cluster, col = pam_result$cluster)
points(pam_result$medoids[, 2:3], col = 1:pam_final_k, pch = 8)

#################
# Fuzzy C-means
#################
cmeans_index <- array(0, c(max_cluster - min_cluster + 1, 2))
for (cluster_n in cluster_range)
{
  cmeans_fit <- cmeans(
    gender_data[, -4],
    centers = cluster_n,
    m = 2,
    verbose = TRUE,
    method = "cmeans"
  )
  cmeans_index[cluster_n - (min_cluster - 1), 1] <- fclustIndex(
    cmeans_fit,
    gender_data[, -4],
    index = "xie.beni"
  )
  cmeans_index[cluster_n - (min_cluster - 1), 2] <- fclustIndex(
    cmeans_fit,
    gender_data[, -4],
    index = "fukuyama.sugeno"
  )
}

cmeans_best_xb <- which(cmeans_index[, 1] == min(cmeans_index[, 1])) + (min_cluster - 1)
cmeans_best_fs <- which(cmeans_index[, 2] == min(cmeans_index[, 2])) + (min_cluster - 1)

cmeans_vote <- c(cmeans_best_xb, cmeans_best_fs)
cmeans_final_k <- as.numeric(names(sort(table(cmeans_vote), decreasing = TRUE)[1]))
cmeans_final_k

cmeans_result <- cmeans(
  gender_data[, -4],
  centers = cmeans_final_k,
  m = 2,
  iter.max = 200,
  verbose = TRUE,
  method = "cmeans"
)
print(cmeans_result)
table(gender_data$Gender, cmeans_result$cluster)

cmeans_result$centers
plot(gender_data[, 1:2], pch = cmeans_result$cluster, col = cmeans_result$cluster)
points(cmeans_result$centers[, 1:2], col = 1:cmeans_final_k, pch = 8)

plot(gender_data[, 2:3], pch = cmeans_result$cluster, col = cmeans_result$cluster)
points(cmeans_result$centers[, 2:3], col = 1:cmeans_final_k, pch = 8)

#########################
# Hierarchical Clustering
#########################
sample_id <- sample(1:nrow(gender_data), 0.2 * nrow(gender_data))

hierarchical_average <- hclust(dist(gender_data[sample_id, -4]), method = "average")
print(hierarchical_average)
plot(hierarchical_average, labels = gender_data$Gender[sample_id])

hierarchical_index_average_ward <- array(0, c(max_cluster - min_cluster + 1, 2))
for (cluster_n in cluster_range)
{
  hc_fit_average <- hclust(dist(gender_data[, -4]), method = "average")
  hc_cut_average <- cutree(hc_fit_average, k = cluster_n)
  hierarchical_index_average_ward[cluster_n - (min_cluster - 1), 1] <- index.DB(
    gender_data[, -4],
    hc_cut_average,
    centrotypes = "centroids"
  )$DB
  
  hc_fit_ward <- hclust(dist(gender_data[, -4]), method = "ward.D")
  hc_cut_ward <- cutree(hc_fit_ward, k = cluster_n)
  hierarchical_index_average_ward[cluster_n - (min_cluster - 1), 2] <- index.DB(
    gender_data[, -4],
    hc_cut_ward,
    centrotypes = "centroids"
  )$DB
}

hierarchical_index_single_complete <- array(0, c(max_cluster - min_cluster + 1, 2))
for (cluster_n in cluster_range)
{
  hc_fit_single <- hclust(dist(gender_data[, -4]), method = "single")
  hc_cut_single <- cutree(hc_fit_single, k = cluster_n)
  hierarchical_index_single_complete[cluster_n - (min_cluster - 1), 1] <- index.DB(
    gender_data[, -4],
    hc_cut_single,
    centrotypes = "centroids"
  )$DB
  
  hc_fit_complete <- hclust(dist(gender_data[, -4]), method = "complete")
  hc_cut_complete <- cutree(hc_fit_complete, k = cluster_n)
  hierarchical_index_single_complete[cluster_n - (min_cluster - 1), 2] <- index.DB(
    gender_data[, -4],
    hc_cut_complete,
    centrotypes = "centroids"
  )$DB
}

hc_best_average <- which(hierarchical_index_average_ward[, 1] == min(hierarchical_index_average_ward[, 1])) + (min_cluster - 1)
hc_best_ward <- which(hierarchical_index_average_ward[, 2] == min(hierarchical_index_average_ward[, 2])) + (min_cluster - 1)
hc_best_single <- which(hierarchical_index_single_complete[, 1] == min(hierarchical_index_single_complete[, 1])) + (min_cluster - 1)
hc_best_complete <- which(hierarchical_index_single_complete[, 2] == min(hierarchical_index_single_complete[, 2])) + (min_cluster - 1)

hierarchical_vote <- c(hc_best_average, hc_best_ward, hc_best_single, hc_best_complete)
hierarchical_final_k <- as.numeric(names(sort(table(hierarchical_vote), decreasing = TRUE)[1]))
hierarchical_final_k

hierarchical_group <- cutree(hierarchical_average, k = hierarchical_final_k)
print(hierarchical_group)
table(hierarchical_group)
rect.hclust(hierarchical_average, k = hierarchical_final_k, border = "red")

hierarchical_group_1 <- gender_data[sample_id, -4][which(hierarchical_group == 1), ]
hierarchical_group_1

########################
# Gaussian Mixture Model
########################
gmm_fit <- Mclust(gender_data[, -4])
summary(gmm_fit, parameters = TRUE)

gmm_final_k <- gmm_fit$G
gmm_final_k

gmm_result <- Mclust(gender_data[, -4], G = gmm_final_k)
gmm_result$parameters
gmm_result$classification
table(gender_data$Gender, gmm_result$classification)

plot(gender_data[, 1:2], pch = gmm_result$classification, col = gmm_result$classification)
plot(gender_data[, 2:3], pch = gmm_result$classification, col = gmm_result$classification)
plot(gender_data[, 1], gender_data[, 3], pch = gmm_result$classification, col = gmm_result$classification)
plot(gender_data[, 1:3], pch = gmm_result$classification, col = gmm_result$classification)

#######################
# 11. Train/Test Split
#######################
set.seed(123)
test_id <- sample(1:nrow(gender_data), round(0.3 * nrow(gender_data)))
train_gender <- gender_data[-test_id, ]
test_gender <- gender_data[test_id, ]

train_label_num <- ifelse(train_gender$Gender == "male", 1, 0)
test_label_num <- ifelse(test_gender$Gender == "male", 1, 0)

####################
# 12. Classification
####################

##########
# KNN
##########
knn_tuned <- tune.knn(train_gender[, -4], train_gender[, 4], k = c(1, 3, 5, 7, 9))
knn_tuned$best.model
knn_best_k <- knn_tuned$best.parameters$k
knn_best_k

knn_train_pred <- knn(
  train = train_gender[, -4],
  test = train_gender[, -4],
  cl = train_gender[, 4],
  k = knn_best_k,
  prob = TRUE
)
knn_train_table <- table(train_gender$Gender, knn_train_pred)
knn_train_acc <- sum(diag(knn_train_table)) / sum(knn_train_table)
knn_train_table
knn_train_acc
cat("KNN train predictive accuracy", 100 * knn_train_acc, "% \n")

knn_test_pred <- knn(
  train = train_gender[, -4],
  test = test_gender[, -4],
  cl = train_gender[, 4],
  k = knn_best_k,
  prob = TRUE
)
knn_test_table <- table(test_gender$Gender, knn_test_pred)
knn_test_acc <- sum(diag(knn_test_table)) / sum(knn_test_table)
knn_test_table
knn_test_acc
cat("KNN test predictive accuracy", 100 * knn_test_acc, "% \n")

knn_test_prob_attr <- attr(knn_test_pred, "prob")
knn_test_prob_male <- ifelse(knn_test_pred == "male", knn_test_prob_attr, 1 - knn_test_prob_attr)

pred_knn_test <- prediction(knn_test_prob_male, test_label_num)
perf_knn_test <- performance(pred_knn_test, measure = "tpr", x.measure = "fpr")
auc_knn_test <- performance(pred_knn_test, "auc")
auc_knn_test_value <- round(auc_knn_test@y.values[[1]], 3)

plot(perf_knn_test, main = "ROC Curve - KNN (Test)", xlab = "FPR", ylab = "TPR")
abline(0, 1)
text(0.6, 0.2, paste("AUC =", auc_knn_test_value))

################
# Naive Bayes
################
naive_bayes_model_1 <- NaiveBayes(Gender ~ ., train_gender)
names(naive_bayes_model_1)

naive_bayes_model_1$apriori
naive_bayes_model_1$tables

naive_bayes_train_prob <- predict(naive_bayes_model_1, train_gender[, -4], type = "raw")
naive_bayes_train_class <- colnames(naive_bayes_train_prob)[max.col(naive_bayes_train_prob)]
naive_bayes_train_table <- table(train_gender$Gender, naive_bayes_train_class)
naive_bayes_train_acc <- sum(diag(naive_bayes_train_table)) / sum(naive_bayes_train_table)
naive_bayes_train_table
naive_bayes_train_acc
cat("Naive Bayes train predictive accuracy", 100 * naive_bayes_train_acc, "% \n")

naive_bayes_test_prob <- predict(naive_bayes_model_1, test_gender[, -4], type = "raw")
naive_bayes_test_class <- colnames(naive_bayes_test_prob)[max.col(naive_bayes_test_prob)]
naive_bayes_test_table <- table(test_gender$Gender, naive_bayes_test_class)
naive_bayes_test_acc <- sum(diag(naive_bayes_test_table)) / sum(naive_bayes_test_table)
naive_bayes_test_table
naive_bayes_test_acc
cat("Naive Bayes test predictive accuracy", 100 * naive_bayes_test_acc, "% \n")

new_person_1 <- data.frame(Height = 175, Weight = 74, Waist = 32.8)
predict(naive_bayes_model_1, new_person_1)

new_person_2 <- data.frame(Height = 166, Weight = 60, Waist = 27.5)
predict(naive_bayes_model_1, new_person_2)

naive_bayes_model_2 <- NaiveBayes(train_gender[, -4], train_gender[, 4])
naive_bayes_prediction_2 <- predict(naive_bayes_model_2, train_gender[, -4])
table(train_gender[, 4], naive_bayes_prediction_2$class)

pred_nb_test <- prediction(naive_bayes_test_prob[, "male"], test_label_num)
perf_nb_test <- performance(pred_nb_test, measure = "tpr", x.measure = "fpr")
auc_nb_test <- performance(pred_nb_test, "auc")
auc_nb_test_value <- round(auc_nb_test@y.values[[1]], 3)

plot(perf_nb_test, main = "ROC Curve - Naive Bayes (Test)", xlab = "FPR", ylab = "TPR")
abline(0, 1)
text(0.6, 0.2, paste("AUC =", auc_nb_test_value))

#####################
# Logistic Regression
#####################
logistic_model_all <- glm(
  formula = Gender ~ .,
  family = binomial(link = "logit"),
  data = train_gender
)
summary(logistic_model_all)

new_case_1 <- list(Height = 174.8, Weight = 74, Waist = 32.8)
predict.glm(logistic_model_all, type = "response", newdata = new_case_1)

new_case_2 <- list(Height = 165.6, Weight = 60, Waist = 28)
predict.glm(logistic_model_all, type = "response", newdata = new_case_2)

logistic_prob_train <- predict.glm(logistic_model_all, type = "response", newdata = train_gender)
logistic_class_train <- ifelse(logistic_prob_train >= 0.5, "male", "female")
logistic_train_table <- table(train_gender$Gender, logistic_class_train)
logistic_train_acc <- sum(diag(logistic_train_table)) / sum(logistic_train_table)
logistic_train_table
logistic_train_acc
cat("Logistic Regression train predictive accuracy", 100 * logistic_train_acc, "% \n")

logistic_prob_test <- predict.glm(logistic_model_all, type = "response", newdata = test_gender)
logistic_class_test <- ifelse(logistic_prob_test >= 0.5, "male", "female")
logistic_test_table <- table(test_gender$Gender, logistic_class_test)
logistic_test_acc <- sum(diag(logistic_test_table)) / sum(logistic_test_table)
logistic_test_table
logistic_test_acc
cat("Logistic Regression test predictive accuracy", 100 * logistic_test_acc, "% \n")

logistic_model_weight_waist <- glm(
  formula = Gender ~ Weight + Waist,
  family = binomial(link = "logit"),
  data = train_gender
)
summary(logistic_model_weight_waist)
logistic_prob_weight_waist <- predict.glm(logistic_model_weight_waist, type = "response", newdata = test_gender)
logistic_class_weight_waist <- ifelse(logistic_prob_weight_waist > 0.5, "male", "female")
table(test_gender[, 4], logistic_class_weight_waist)

logistic_model_height_weight <- glm(
  Gender ~ Height + Weight,
  data = train_gender,
  family = binomial(link = "logit")
)
summary(logistic_model_height_weight)
logistic_prob_height_weight <- predict.glm(logistic_model_height_weight, type = "response", newdata = test_gender)
logistic_class_height_weight <- ifelse(logistic_prob_height_weight > 0.5, "male", "female")
table(test_gender[, 4], logistic_class_height_weight)

logistic_model_height_waist <- glm(
  Gender ~ Height + Waist,
  data = train_gender,
  family = binomial(link = "logit")
)
summary(logistic_model_height_waist)
logistic_prob_height_waist <- predict.glm(logistic_model_height_waist, type = "response", newdata = test_gender)
logistic_class_height_waist <- ifelse(logistic_prob_height_waist > 0.5, "male", "female")
table(test_gender[, 4], logistic_class_height_waist)

pred_logit_test <- prediction(logistic_prob_test, test_label_num)
perf_logit_test <- performance(pred_logit_test, measure = "tpr", x.measure = "fpr")
auc_logit_test <- performance(pred_logit_test, "auc")
auc_logit_test_value <- round(auc_logit_test@y.values[[1]], 3)

plot(perf_logit_test, main = "ROC Curve - Logistic Regression (Test)", xlab = "FPR", ylab = "TPR")
abline(0, 1)
text(0.6, 0.2, paste("AUC =", auc_logit_test_value))

##########
# SVM
##########
svm_tuned <- tune.svm(
  Gender ~ .,
  data = train_gender,
  gamma = 5^(-3:0),
  cost = 10^(-1:2)
)
summary(svm_tuned)
svm_tuned$best.parameters

svm_best_cost <- svm_tuned$best.parameters$cost
svm_best_gamma <- svm_tuned$best.parameters$gamma

svm_model <- svm(
  Gender ~ .,
  data = train_gender,
  kernel = "radial",
  cost = svm_best_cost,
  gamma = svm_best_gamma,
  probability = TRUE
)
summary(svm_model)

svm_train_pred <- predict(svm_model, train_gender[, -4], probability = TRUE)
svm_train_table <- table(train_gender$Gender, svm_train_pred)
svm_train_acc <- sum(diag(svm_train_table)) / sum(svm_train_table)
svm_train_table
svm_train_acc
cat("SVM train predictive accuracy", 100 * svm_train_acc, "% \n")

svm_test_pred <- predict(svm_model, test_gender[, -4], probability = TRUE)
svm_test_table <- table(test_gender$Gender, svm_test_pred)
svm_test_acc <- sum(diag(svm_test_table)) / sum(svm_test_table)
svm_test_table
svm_test_acc
cat("SVM test predictive accuracy", 100 * svm_test_acc, "% \n")

svm_test_prob <- attr(svm_test_pred, "probabilities")[, "male"]

pred_svm_test <- prediction(svm_test_prob, test_label_num)
perf_svm_test <- performance(pred_svm_test, measure = "tpr", x.measure = "fpr")
auc_svm_test <- performance(pred_svm_test, "auc")
auc_svm_test_value <- round(auc_svm_test@y.values[[1]], 3)

plot(perf_svm_test, main = "ROC Curve - SVM (Test)", xlab = "FPR", ylab = "TPR")
abline(0, 1)
text(0.6, 0.2, paste("AUC =", auc_svm_test_value))

plot(
  svm_model, train_gender, Height ~ Weight,
  fill = FALSE,
  symbolPalette = c("red", "blue"),
  svSymbol = "+",
  dataSymbol = "o"
)
plot(
  svm_model, train_gender, Waist ~ Height,
  fill = FALSE,
  symbolPalette = c("red", "blue"),
  svSymbol = "+",
  dataSymbol = "o"
)
plot(
  svm_model, train_gender, Waist ~ Weight,
  fill = FALSE,
  symbolPalette = c("red", "blue"),
  svSymbol = "+",
  dataSymbol = "o"
)