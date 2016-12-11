library(dplyr)
library(ggplot2)
library(purrr)
library(tensorflow)
library(MASS)
library(dplyr)
library(caret)
require( 'kernlab' )

#######################################
# Linearly separable data illustrated #
#######################################


# the separator line is: 2x + 5y -2 = 0

x <- runif(10, min=0, max=1)
y <- runif(10, min=0, max=1)
z <- as_vector(map2(x, y, function(x,y) {2*x + 5*y - 2}))
# positive sign is one class, negative the other
z_sgn <- sign(z)

df <- data_frame(x,y,z,z_sgn)

ggplot(df, aes(x,y)) + geom_point(aes(color=factor(z_sgn))) + 
  geom_abline(slope=-0.4, intercept=0.4, linetype = 'dashed')


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
# http://sebastianraschka.com/Articles/2014_python_lda.html
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

lda_pred <- predict(lda_fit, X_test[1,])
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
#linear.svm <- ksvm( y ~ ., data=linear.train, type='C-svc', kernel='vanilladot',
                    C=100, scale=c() )

# Plot the model
#plot( linear.svm, data=linear.train )

# Predictions for test set
#linear.prediction <- predict( linear.svm, linear.test )

# Prediction scores
#linear.prediction.score <- predict( linear.svm, linear.test, type='decision' )
