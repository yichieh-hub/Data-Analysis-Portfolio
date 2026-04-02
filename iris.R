############################
# install packages (commented)
############################
# install.packages("timeDate")
# install.packages("ggplot2")
# install.packages("cluster")
# install.packages("clusterSim")
# install.packages("NbClust")
# install.packages("e1071")
# install.packages("class")
# install.packages("klaR")
# install.packages("randomForest")
# install.packages("psych")
# install.packages("mclust")
# install.packages("nortest")

############################
# libraries
############################
library(timeDate)
library(ggplot2)
library(cluster)
library(clusterSim)
library(NbClust)
library(e1071)
library(class)
library(klaR)
library(randomForest)
library(psych)
library(mclust)
library(nortest)

############################
# Data Preparation
############################
data(iris)

iris_df <- iris
iris_x <- iris_df[, 1:4]
iris_y <- iris_df$Species

set.seed(123)
iris_index <- sample(1:nrow(iris_df), round(0.7 * nrow(iris_df)))

iris_train <- iris_df[iris_index, ]
iris_test  <- iris_df[-iris_index, ]

x_train <- iris_train[, 1:4]
x_test  <- iris_test[, 1:4]
y_train <- iris_train$Species
y_test  <- iris_test$Species

############################################################
################ Descriptive Data Analysis ##################
############################################################
attributes(iris_df)
str(iris_df)
summary(iris_df)
head(iris_df, 5)

iris_var_index <- c(1:4)

colMeans(iris_df[, -5])

cor(iris_df$Sepal.Length, iris_df$Sepal.Width)
iris_corr <- cor(iris_df[, iris_var_index], use = "pairwise")
iris_corr

cov(iris_df$Petal.Length, iris_df$Petal.Width)
iris_cov <- cov(iris_df[, iris_var_index], use = "pairwise")
iris_cov

skewness(iris_df[, 1:4])
kurtosis(iris_df[, 1:4])

############################################################
#################### Visualization ##########################
############################################################
plot(iris_df)

plot(iris_df$Sepal.Length, iris_df$Sepal.Width)

plot(iris_df$Species, iris_df$Sepal.Length, main = "Distribution of Sepal.Length")

iris_box <- boxplot(iris_df[, 1:4], main = "Three species")
iris_box$stats

iris_setosa_id <- which(iris_df[, 5] == "setosa")
boxplot(iris_df[iris_setosa_id, 1:4], main = "setosa")

iris_species_count <- with(
  iris_df,
  c(
    sum(iris_df[, 5] == "setosa"),
    sum(iris_df[, 5] == "versicolor"),
    sum(iris_df[, 5] == "virginica")
  )
)

barplot(
  iris_species_count,
  names.arg = c("setosa", "versicolor", "virginica"),
  xlab = "species",
  ylab = "number"
)

pie(iris_species_count, labels = c("setosa", "versicolor", "virginica"))
pie(
  iris_species_count,
  labels = c(
    sum(iris_df[, 5] == "setosa"),
    sum(iris_df[, 5] == "versicolor"),
    sum(iris_df[, 5] == "virginica")
  )
)

iris_percent <- round(iris_species_count / sum(iris_species_count) * 100)
iris_label <- paste(levels(iris_df$Species), iris_percent, "%")
pie(iris_species_count, iris_label)

plot(
  iris_df$Sepal.Length[iris_df$Species == "setosa"],
  iris_df$Petal.Length[iris_df$Species == "setosa"],
  pch = 1,
  col = "blue",
  xlim = c(3, 8),
  ylim = c(0, 9),
  main = "scatter plot",
  xlab = "SepalLen",
  ylab = "PetalLen"
)
points(
  iris_df$Sepal.Length[iris_df$Species == "versicolor"],
  iris_df$Petal.Length[iris_df$Species == "versicolor"],
  pch = 2,
  col = "red"
)
points(
  iris_df$Sepal.Length[iris_df$Species == "virginica"],
  iris_df$Petal.Length[iris_df$Species == "virginica"],
  pch = 3,
  col = "green"
)
legend(3, 9, c("setosa", "versicolor", "virginica"), col = c(1, 2, 3), pch = c(1, 2, 3))

hist(
  iris_df$Sepal.Length,
  breaks = 20,
  labels = TRUE,
  col = "blue",
  border = "red",
  main = "Histogram of frequency"
)

hist(iris_df$Sepal.Length, freq = FALSE, main = "Histogram of density")
lines(density(iris_df$Sepal.Length))

iris_ecdf <- ecdf(iris_df$Sepal.Length)
plot(iris_ecdf, xlab = "Sepal.Length", main = "Cumulative frequency")

############################################################
#################### Normality Test #########################
############################################################
hist(iris_df$Sepal.Length, breaks = seq(4.0, 8.0, 0.05))
hist(iris_df$Sepal.Length, breaks = seq(4.0, 8.0, 0.25), prob = TRUE)
qqnorm(iris_df$Sepal.Length, xlab = "Z-score", ylab = "Sepal.Length")
qqline(iris_df$Sepal.Length, col = "red")

curve(
  dnorm(x, mean(iris_df$Sepal.Length), sd(iris_df$Sepal.Length)),
  4.0,
  8.0,
  col = "red"
)

shapiro.test(iris_df$Sepal.Length)

qqplot(iris_df$Sepal.Length, iris_df$Sepal.Width)
qqplot(iris_df$Petal.Length, iris_df$Petal.Width)

ad.test(iris_df$Sepal.Width)
sf.test(iris_df$Sepal.Width)
cvm.test(iris_df$Sepal.Width)
shapiro.test(iris_df$Sepal.Width)

############################################################
########################## MDS ##############################
############################################################
iris_dist_mds <- dist(iris_df[, 1:4], method = "euclidean")
iris_mds <- cmdscale(iris_dist_mds, k = 2, eig = TRUE)
iris_species_num <- as.numeric(iris_df$Species)

iris_mds_x <- iris_mds$points[, 1]
iris_mds_y <- iris_mds$points[, 2]

plot(iris_mds_x, iris_mds_y, pch = iris_species_num, col = iris_species_num + 1, main = "MDS for Iris")

iris_mds_plot <- ggplot(data.frame(iris_mds_x, iris_mds_y), aes(iris_mds_x, iris_mds_y))
iris_mds_plot + geom_point(size = 3, aes(colour = factor(iris_species_num), shape = iris_df$Species))

############################################################
######################## Clustering #########################
############################################################
min_nc <- 2
max_nc <- 8

################ K-means ################
iris_km_eval <- array(0, c(max_nc - min_nc + 1, 2))
for (nc in min_nc:max_nc)
{
  iris_fit_km <- kmeans(iris_df[, -5], centers = nc)
  iris_km_eval[nc - (min_nc - 1), 1] <- iris_fit_km$betweenss / iris_fit_km$tot.withinss
  iris_km_eval[nc - (min_nc - 1), 2] <- index.DB(
    iris_df[, -5],
    iris_fit_km$cluster,
    centrotypes = "centroids",
    p = 2
  )$DB
}
which(iris_km_eval[, 1] == max(iris_km_eval[, 1]))
which(iris_km_eval[, 2] == min(iris_km_eval[, 2]))

iris_nbclust_kmeans <- NbClust(
  iris_df[, 1:4],
  distance = "euclidean",
  min.nc = 2,
  max.nc = 8,
  method = "kmeans",
  index = "all"
)
iris_kmeans_best_k <- as.numeric(iris_nbclust_kmeans$Best.nc[1, "Number_clusters"])
iris_kmeans_best_k

set.seed(123)
iris_kmeans_best <- kmeans(iris_df[, 1:4], centers = iris_kmeans_best_k)
print(iris_kmeans_best)
iris_kmeans_best$centers
plot(iris_df)

table(iris_df$Species, iris_kmeans_best$cluster)

plot(iris_df[, 1:2], pch = iris_kmeans_best$cluster, col = iris_kmeans_best$cluster)
points(iris_kmeans_best$centers[, 1:2], col = 1:iris_kmeans_best_k, pch = 8)

plot(iris_df[, 3:4], pch = iris_kmeans_best$cluster, col = iris_kmeans_best$cluster)
points(iris_kmeans_best$centers[, 3:4], col = 1:iris_kmeans_best_k, pch = 8)

################ K-medoids ################
iris_pm_eval <- array(0, c(max_nc - min_nc + 1, 2))
for (nc in min_nc:max_nc)
{
  iris_fit_pm <- pam(iris_df[, -5], nc)
  iris_pm_eval[nc - (min_nc - 1), 1] <- index.DB(
    iris_df[, -5],
    iris_fit_pm$clustering,
    centrotypes = "centroids",
    p = 2
  )$DB
  
  iris_pm_eval[nc - (min_nc - 1), 2] <- index.DB(
    iris_df[, -5],
    iris_fit_pm$clustering,
    dist(iris_df[, -5]),
    centrotypes = "medoids",
    p = 2
  )$DB
}
which(iris_pm_eval[, 1] == min(iris_pm_eval[, 1]))
which(iris_pm_eval[, 2] == min(iris_pm_eval[, 2]))

iris_pam_best_k <- which(iris_pm_eval[, 2] == min(iris_pm_eval[, 2]))[1] + min_nc - 1
iris_pam_best_k

iris_pam_best <- pam(iris_df[, -5], iris_pam_best_k)
print(iris_pam_best)
iris_pam_best$medoids
table(iris_df$Species, iris_pam_best$cluster)

plot(iris_df[, 1:2], pch = iris_pam_best$cluster, col = iris_pam_best$cluster)
points(iris_pam_best$medoids[, 1:2], col = 1:iris_pam_best_k, pch = 8)

plot(iris_df[, 3:4], pch = iris_pam_best$cluster, col = iris_pam_best$cluster)
points(iris_pam_best$medoids[, 3:4], col = 1:iris_pam_best_k, pch = 8)

################ Fuzzy C-means ################
iris_cm_eval <- array(0, c(max_nc - min_nc + 1, 2))
for (nc in min_nc:max_nc)
{
  iris_fit_cm <- cmeans(
    iris_df[, -5],
    centers = nc,
    m = 2,
    verbose = TRUE,
    method = "cmeans"
  )
  iris_cm_eval[nc - (min_nc - 1), 1] <- fclustIndex(
    iris_fit_cm,
    iris_df[, -5],
    index = "xie.beni"
  )
  iris_cm_eval[nc - (min_nc - 1), 2] <- fclustIndex(
    iris_fit_cm,
    iris_df[, -5],
    index = "fukuyama.sugeno"
  )
}
which(iris_cm_eval[, 1] == min(iris_cm_eval[, 1]))
which(iris_cm_eval[, 2] == min(iris_cm_eval[, 2]))

iris_cmeans_best_k <- which(iris_cm_eval[, 1] == min(iris_cm_eval[, 1]))[1] + min_nc - 1
iris_cmeans_best_k

iris_cmeans_best <- cmeans(
  iris_df[, -5],
  centers = iris_cmeans_best_k,
  m = 2,
  iter.max = 200,
  verbose = TRUE,
  method = "cmeans"
)
print(iris_cmeans_best)
table(iris_df$Species, iris_cmeans_best$cluster)

iris_cmeans_best$centers
plot(iris_df[, 1:2], pch = iris_cmeans_best$cluster, col = iris_cmeans_best$cluster)
points(iris_cmeans_best$centers[, 1:2], col = 1:iris_cmeans_best_k, pch = 8)

plot(iris_df[, 3:4], pch = iris_cmeans_best$cluster, col = iris_cmeans_best$cluster)
points(iris_cmeans_best$centers[, 3:4], col = 1:iris_cmeans_best_k, pch = 8)

################ Hierarchical clustering ################
set.seed(123)
iris_hc_id <- sample(1:nrow(iris_df), 0.2 * nrow(iris_df))

iris_hclust_avg_demo <- hclust(dist(iris_df[iris_hc_id, -5]), method = "average")
print(iris_hclust_avg_demo)
plot(iris_hclust_avg_demo, labels = iris_df$Species[iris_hc_id])

iris_hc_group_demo <- cutree(iris_hclust_avg_demo, k = 3)
print(iris_hc_group_demo)
table(iris_hc_group_demo)
rect.hclust(iris_hclust_avg_demo, k = 3, border = "red")

iris_hc_test <- iris_df[iris_hc_id, -5]
iris_hc_group1 <- iris_hc_test[which(iris_hc_group_demo == 1), ]
iris_hc_group1

iris_hc_eval_avg_ward <- array(0, c(max_nc - min_nc + 1, 2))
for (nc in min_nc:max_nc)
{
  iris_fit_hc1 <- hclust(dist(iris_df[, -5]), method = "average")
  iris_cut_1 <- cutree(iris_fit_hc1, k = nc)
  iris_hc_eval_avg_ward[nc - (min_nc - 1), 1] <- index.DB(
    iris_df[, -5],
    iris_cut_1,
    centrotypes = "centroids"
  )$DB
  
  iris_fit_hc2 <- hclust(dist(iris_df[, -5]), method = "ward.D")
  iris_cut_2 <- cutree(iris_fit_hc2, k = nc)
  iris_hc_eval_avg_ward[nc - (min_nc - 1), 2] <- index.DB(
    iris_df[, -5],
    iris_cut_2,
    centrotypes = "centroids"
  )$DB
}
which(iris_hc_eval_avg_ward[, 1] == min(iris_hc_eval_avg_ward[, 1]))
which(iris_hc_eval_avg_ward[, 2] == min(iris_hc_eval_avg_ward[, 2]))

iris_hclust_best_k <- which(iris_hc_eval_avg_ward[, 2] == min(iris_hc_eval_avg_ward[, 2]))[1] + min_nc - 1
iris_hclust_best_k

iris_hclust_best <- hclust(dist(iris_df[, -5]), method = "ward.D")
plot(iris_hclust_best, labels = iris_df$Species)
rect.hclust(iris_hclust_best, k = iris_hclust_best_k, border = "red")
iris_hclust_best_group <- cutree(iris_hclust_best, k = iris_hclust_best_k)
table(iris_df$Species, iris_hclust_best_group)

iris_hc_eval_single_complete <- array(0, c(max_nc - min_nc + 1, 2))
for (nc in min_nc:max_nc)
{
  iris_fit_hc3 <- hclust(dist(iris_df[, -5]), method = "single")
  iris_cut_3 <- cutree(iris_fit_hc3, k = nc)
  iris_hc_eval_single_complete[nc - (min_nc - 1), 1] <- index.DB(
    iris_df[, -5],
    iris_cut_3,
    centrotypes = "centroids"
  )$DB
  
  iris_fit_hc4 <- hclust(dist(iris_df[, -5]), method = "complete")
  iris_cut_4 <- cutree(iris_fit_hc4, k = nc)
  iris_hc_eval_single_complete[nc - (min_nc - 1), 2] <- index.DB(
    iris_df[, -5],
    iris_cut_4,
    centrotypes = "centroids"
  )$DB
}
which(iris_hc_eval_single_complete[, 1] == min(iris_hc_eval_single_complete[, 1]))
which(iris_hc_eval_single_complete[, 2] == min(iris_hc_eval_single_complete[, 2]))

################ Gaussian Mixture Model ################
iris_gmm_best <- Mclust(iris_df[, -5])
summary(iris_gmm_best, parameters = TRUE)
iris_gmm_best$parameters
iris_gmm_best$classification
table(iris_df[, 5], iris_gmm_best$classification)

iris_gmm_best$BIC

iris_bic <- mclustBIC(iris_df[, -5], G = seq(from = 1, to = 9, by = 1))
iris_bic
plot(iris_bic)

iris_bic_summary <- summary(iris_bic, data = iris_df[, -5])
iris_bic_summary
names(iris_bic_summary)
iris_bic_summary$classification

mclust2Dplot(iris_df[, 3:4], classification = iris_bic_summary$classification)

iris_density_1 <- densityMclust(iris_df[, 3:4])
plot(iris_density_1, iris_df[, 3:4], col = "blue", nlevels = 15)

iris_density_2 <- densityMclust(iris_df[, 3:4])
plot(iris_density_2, iris_df[, 3:4], type = "persp", col = "red")

############################################################
################## Basic Classification ####################
############################################################

################ KNN ################
iris_knn_tuned <- tune.knn(x_train, y_train, k = c(1, 3, 5, 7, 9))
iris_knn_tuned$best.model
iris_knn_best_k <- iris_knn_tuned$best.parameters$k
iris_knn_best_k

iris_knn_train <- knn(x_train, x_train, cl = y_train, k = iris_knn_best_k)
iris_knn_test  <- knn(x_train, x_test,  cl = y_train, k = iris_knn_best_k)

table(y_train, iris_knn_train)
iris_knn_train_acc <- sum(as.numeric(iris_knn_train == y_train)) / nrow(iris_train)
iris_knn_train_acc

table(y_test, iris_knn_test)
iris_knn_test_acc <- sum(as.numeric(iris_knn_test == y_test)) / nrow(iris_test)
iris_knn_test_acc

iris_knn_result <- table(y_test, iris_knn_test)
sum(diag(iris_knn_result)) / sum(iris_knn_result)

################ Naive Bayes ################
iris_nb_model <- NaiveBayes(Species ~ Petal.Length + Petal.Width, data = iris_train)
names(iris_nb_model)
iris_nb_model$apriori
iris_nb_model$tables

iris_nb_train <- predict(iris_nb_model, iris_train)
table(y_train, iris_nb_train$class)
iris_nb_train_acc <- sum(as.numeric(iris_nb_train$class == y_train)) / nrow(iris_train)
iris_nb_train_acc

iris_nb_test <- predict(iris_nb_model, iris_test)
table(y_test, iris_nb_test$class)
iris_nb_test_acc <- sum(as.numeric(iris_nb_test$class == y_test)) / nrow(iris_test)
iris_nb_test_acc

############################################################
##################### Random Forest #########################
############################################################
set.seed(123)
iris_rf_tuned <- tuneRF(
  x = x_train,
  y = y_train,
  ntreeTry = 30,
  stepFactor = 1.5,
  improve = 0.01,
  trace = TRUE,
  plot = TRUE,
  doBest = FALSE
)
iris_rf_tuned

iris_rf_best_mtry <- iris_rf_tuned[which.min(iris_rf_tuned[, 2]), 1]
iris_rf_best_mtry

set.seed(123)
iris_rf_model <- randomForest(
  Species ~ .,
  data = iris_train,
  ntree = 30,
  mtry = iris_rf_best_mtry,
  importance = TRUE,
  proximity = TRUE,
  na.action = na.omit
)
importance(iris_rf_model)
iris_rf_model$confusion
MDSplot(iris_rf_model, iris_train$Species, pch = as.numeric(iris_train$Species))

plot(iris_rf_model)
print(iris_rf_model)

iris_rf_train_pred <- predict(iris_rf_model, newdata = iris_train)
table(y_train, iris_rf_train_pred)
iris_rf_train_acc <- sum(as.numeric(iris_rf_train_pred == y_train)) / nrow(iris_train)
iris_rf_train_acc

iris_rf_test_pred <- predict(iris_rf_model, newdata = iris_test)
table(y_test, iris_rf_test_pred)
iris_rf_test_acc <- sum(as.numeric(iris_rf_test_pred == y_test)) / nrow(iris_test)
iris_rf_test_acc

############################################################
########################### PCA #############################
############################################################

################ PCA by prcomp ################
iris_pca <- prcomp(
  ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
  data = iris_df,
  center = TRUE,
  scale = TRUE
)
summary(iris_pca)

plot(iris_pca, type = "line", main = "Scree Plot for iris flower")
iris_pca$sdev^2

iris_loading <- iris_pca$rotation
iris_loading
biplot(iris_pca, choices = 1:2)

iris_sorted_loading_pc1 <- iris_loading[order(iris_loading[, 1]), 1]
dotchart(
  iris_sorted_loading_pc1,
  main = "Loading Plot for PC1",
  xlab = "Variable Loadings",
  cex = 1.5,
  col = "red"
)

iris_sorted_loading_pc2 <- iris_loading[order(iris_loading[, 2]), 2]
dotchart(
  iris_sorted_loading_pc2,
  main = "Loading Plot for PC2",
  xlab = "Variable Loadings",
  cex = 1.5,
  col = "blue"
)

iris_sorted_loading_pc3 <- iris_loading[order(iris_loading[, 3]), 3]
dotchart(
  iris_sorted_loading_pc3,
  main = "Loading Plot for PC3",
  xlab = "Variable Loadings",
  cex = 1.5,
  col = "green"
)

################ PCA by principal ################
iris_rpca <- principal(iris_df[, -5], nfactors = 2, score = TRUE)
print(iris_rpca)
iris_rpca$loadings

iris_rpca_df <- data.frame(cbind(iris_rpca$scores, iris_df$Species))
colnames(iris_rpca_df) <- c("RC1", "RC2", "Species")
iris_rpca_df$Species <- as.factor(iris_rpca_df$Species)

plot(iris_rpca_df$RC1, iris_rpca_df$RC2, pch = iris_rpca_df$Species, col = iris_rpca_df$Species)

################ PCA + K-means ################
iris_pca_km_eval <- array(0, c(max_nc - min_nc + 1, 2))
for (nc in min_nc:max_nc)
{
  iris_pca_fit_km <- kmeans(iris_rpca_df[, 1:2], centers = nc)
  iris_pca_km_eval[nc - (min_nc - 1), 1] <- iris_pca_fit_km$betweenss / iris_pca_fit_km$tot.withinss
  iris_pca_km_eval[nc - (min_nc - 1), 2] <- index.DB(
    iris_rpca_df[, 1:2],
    iris_pca_fit_km$cluster,
    centrotypes = "centroids",
    p = 2
  )$DB
}
which(iris_pca_km_eval[, 1] == max(iris_pca_km_eval[, 1]))
which(iris_pca_km_eval[, 2] == min(iris_pca_km_eval[, 2]))

iris_pca_kmeans_best_k <- which(iris_pca_km_eval[, 2] == min(iris_pca_km_eval[, 2]))[1] + min_nc - 1
iris_pca_kmeans_best_k

set.seed(123)
iris_pca_kmeans <- kmeans(iris_rpca_df[, 1:2], centers = iris_pca_kmeans_best_k)
print(iris_pca_kmeans)
iris_pca_kmeans$centers
iris_pca_kmeans$cluster

table(iris_df$Species, iris_pca_kmeans$cluster)

plot(iris_rpca_df[, 1:2], pch = iris_pca_kmeans$cluster, col = iris_pca_kmeans$cluster)
points(iris_pca_kmeans$centers, col = 1:iris_pca_kmeans_best_k, pch = 8)

################ PCA train / test split ################
iris_train_pca <- predict(iris_pca, newdata = iris_train[, -5])
iris_test_pca  <- predict(iris_pca, newdata = iris_test[, -5])

iris_train_pca_df <- data.frame(PC1 = iris_train_pca[, 1], PC2 = iris_train_pca[, 2], Species = y_train)
iris_test_pca_df  <- data.frame(PC1 = iris_test_pca[, 1],  PC2 = iris_test_pca[, 2],  Species = y_test)

################ PCA + KNN Classification ################
iris_pca_knn_tuned <- tune.knn(
  iris_train_pca_df[, 1:2],
  iris_train_pca_df[, 3],
  k = c(1, 3, 5, 7, 9)
)
iris_pca_knn_tuned$best.model
iris_pca_knn_best_k <- iris_pca_knn_tuned$best.parameters$k
iris_pca_knn_best_k

iris_pca_knn_train <- knn(
  iris_train_pca_df[, 1:2],
  iris_train_pca_df[, 1:2],
  cl = iris_train_pca_df[, 3],
  k = iris_pca_knn_best_k
)
table(iris_train_pca_df[, 3], iris_pca_knn_train)
iris_pca_knn_train_acc <- sum(as.numeric(iris_pca_knn_train == iris_train_pca_df[, 3])) / nrow(iris_train_pca_df)
iris_pca_knn_train_acc

iris_pca_knn_test <- knn(
  iris_train_pca_df[, 1:2],
  iris_test_pca_df[, 1:2],
  cl = iris_train_pca_df[, 3],
  k = iris_pca_knn_best_k
)
table(iris_test_pca_df[, 3], iris_pca_knn_test)
iris_pca_knn_test_acc <- sum(as.numeric(iris_pca_knn_test == iris_test_pca_df[, 3])) / nrow(iris_test_pca_df)
iris_pca_knn_test_acc

################ PCA + Naive Bayes Classification ################
iris_pca_nb_model <- NaiveBayes(Species ~ ., data = iris_train_pca_df)

iris_pca_nb_train <- predict(iris_pca_nb_model, iris_train_pca_df[, -3])
table(iris_train_pca_df[, 3], iris_pca_nb_train$class)
iris_pca_nb_train_acc <- sum(as.numeric(iris_pca_nb_train$class == iris_train_pca_df$Species)) / nrow(iris_train_pca_df)
iris_pca_nb_train_acc

iris_pca_nb_test <- predict(iris_pca_nb_model, iris_test_pca_df[, -3])
table(iris_test_pca_df[, 3], iris_pca_nb_test$class)
iris_pca_nb_test_acc <- sum(as.numeric(iris_pca_nb_test$class == iris_test_pca_df$Species)) / nrow(iris_test_pca_df)
iris_pca_nb_test_acc

############################################################
############################ SVM ############################
############################################################

################ Multi-SVM parameter selection ################
iris_svm_x <- subset(iris_df, select = -Species)
iris_svm_y <- iris_df$Species

iris_svm_type <- c("C-classification", "nu-classification")
iris_svm_kernel <- c("linear", "polynomial", "radial", "sigmoid")

iris_fit_svm <- array(0, dim = c(150, 2, 4))
iris_svm_error <- matrix(0, 2, 4)
iris_svm_z <- as.integer(iris_svm_y)

for (i in 1:2)
{
  for (j in 1:4)
  {
    iris_svm_temp_model <- svm(
      iris_svm_x,
      iris_svm_y,
      type = iris_svm_type[i],
      kernel = iris_svm_kernel[j]
    )
    iris_fit_svm[, i, j] <- predict(iris_svm_temp_model, iris_svm_x)
    iris_svm_error[i, j] <- sum(iris_fit_svm[, i, j] != iris_svm_z)
  }
}
dimnames(iris_svm_error) <- list(iris_svm_type, iris_svm_kernel)
iris_svm_error

iris_svm_demo_model <- svm(
  iris_svm_x,
  iris_svm_y,
  kernel = "radial",
  cost = 10,
  nu = 0.5,
  na.action = na.omit
)
predict(iris_svm_demo_model, iris_svm_x[95:105, ], decision.values = TRUE)

################ SVM classification with tuning ################
iris_svm_tuned <- tune.svm(
  Species ~ ., 
  data = iris_train,
  gamma = 10^(-3:1),
  cost = 10^(-1:2)
)
summary(iris_svm_tuned)
iris_svm_tuned$best.parameters

iris_svm_model <- iris_svm_tuned$best.model

iris_svm_train_pred <- predict(iris_svm_model, iris_train[, -5])
table(y_train, iris_svm_train_pred)
iris_svm_train_acc <- sum(as.numeric(iris_svm_train_pred == y_train)) / nrow(iris_train)
iris_svm_train_acc

iris_svm_test_pred <- predict(iris_svm_model, iris_test[, -5])
table(y_test, iris_svm_test_pred)
iris_svm_test_acc <- sum(as.numeric(iris_svm_test_pred == y_test)) / nrow(iris_test)
iris_svm_test_acc