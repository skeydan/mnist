library(dplyr)
library(ggplot2)
library(purrr)
library(tensorflow)
library(MASS)
library(dplyr)
library(caret)
library(kernlab)
library(class)
library(gridExtra)

#######################################
# Linearly separable data illustrated #
#######################################


# the separating hyperplane is: 2x + 5y -3 = 0

x <- runif(20, min=0, max=1)
y <- runif(20, min=0, max=1)
z <- as_vector(map2(x, y, function(x,y) {2*x + 5*y - 3}))
# positive sign is one class, negative the other
z_sgn <- sign(z)
z_random <- sample(z_sgn)

df <- data_frame(x,y,z,z_sgn, z_random)
df
g1 <- ggplot(df, aes(x,y)) + geom_point(aes(color=factor(z_sgn))) + 
  geom_abline(slope=-0.4, intercept=0.6, linetype = 'dashed') +
  coord_fixed() + theme (legend.position = "none") 
g2 <- ggplot(df, aes(x,y)) + geom_point(aes(color=factor(z_random))) +
  coord_fixed() + theme (legend.position = "none") 
grid.arrange(g1,g2,ncol=2)

#######################################
#                Get data             #
#######################################


datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

X_train <- mnist$train$images
y_train <- mnist$train$labels

y_train <- apply(y_train, 1, function(r) { which.max(r) - 1 })
y_train[1:10]

X_test <- mnist$test$images
y_test <- mnist$test$labels

y_test <- apply(y_test, 1, function(r) { which.max(r) - 1 })
y_test[1:10]


#######################################
#                LDA                  #
#######################################

# https://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Classification/SVM
# http://www.ats.ucla.edu/stat/r/dae/mlogit.htm
# http://cbio.mines-paristech.fr/~pchiche/teaching/mlbio/mlbio_2012.R


lda(X_train, y_train)

table(X_train[, 1])

z = nearZeroVar(X_train, saveMetrics = TRUE)
nrow(z[z$zeroVar==TRUE,])
z$zeroVar==TRUE
X_train <- X_train[,z$zeroVar==FALSE]
ncol(X_train)

X_test <- X_test[,z$zeroVar==FALSE]
ncol(X_test)

lda_fit <- lda(X_train, y_train)

lda_pred <- predict(lda_fit, X_test[1:10,])
lda_pred

lda_pred <- predict(lda_fit, X_test)
ct <- table(lda_pred$class, y_test)
# percent correct for each category 
diag(prop.table(ct, 1))
# total percent correct
sum(diag(prop.table(ct)))


#######################################
#          Linear SVM                 #
#######################################

#X_train = X_train[1:10000,]
#y_train = y_train[1:10000]
svm_fit_linear <- ksvm(x = X_train, y = y_train, type='C-svc',
                       kernel='vanilladot', C=1, scale=FALSE)

# Predictions for test set
svm_pred <- predict(svm_fit_linear, X_test[1:10,])
svm_pred
svm_pred <- predict(svm_fit_linear, X_test)

ct <- table(svm_pred, y_test)
ct
# percent correct for each category 
diag(prop.table(ct, 1))
# total percent correct
sum(diag(prop.table(ct))) #0.9393

# regularization
svm_fit_linear <- ksvm(x = X_train, y = y_train, type='C-svc',
                       kernel='vanilladot', C=10, scale=FALSE)

# Predictions for test set
svm_pred <- predict(svm_fit_linear, X_test[1:10,])
svm_pred
svm_pred <- predict(svm_fit_linear, X_test)

ct <- table(svm_pred, y_test)
ct
# percent correct for each category 
diag(prop.table(ct, 1))
# total percent correct
sum(diag(prop.table(ct))) 


#######################################
#          Nonlinear SVM              #
#######################################

svm_fit_rbf <- ksvm(x = X_train, y = y_train, type='C-svc',
                    kernel='rbf', C=1, scale=FALSE)

# Predictions for test set
svm_pred <- predict(svm_fit_rbf, X_test[1:10,])
svm_pred
svm_pred <- predict(svm_fit_rbf, X_test)

ct <- table(svm_pred, y_test)
ct
# percent correct for each category 
diag(prop.table(ct, 1))
# total percent correct
sum(diag(prop.table(ct))) 



