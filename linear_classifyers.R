library(dplyr)
library(ggplot2)
library(purrr)
library(tensorflow)
library(MASS)
library(dplyr)
library(caret)

# Linear separable data

# the separator line is: 2x + 5y -2 = 0

x <- runif(10, min=0, max=1)
y <- runif(10, min=0, max=1)
z <- as_vector(map2(x, y, function(x,y) {2*x + 5*y - 2}))
# positive sign is one class, negative the other
z_sgn <- sign(z)

df <- data_frame(x,y,z,z_sgn)

ggplot(df, aes(x,y)) + geom_point(aes(color=factor(z_sgn))) + 
  geom_abline(slope=-0.4, intercept=0.4, linetype = 'dashed')


# http://sebastianraschka.com/Articles/2014_python_lda.html
# https://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Classification/SVM
# https://tgmstat.wordpress.com/2014/03/06/near-zero-variance-predictors/
# http://www.ats.ucla.edu/stat/r/dae/mlogit.htm
# http://cbio.mines-paristech.fr/~pchiche/teaching/mlbio/mlbio_2012.R


datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

X_train <- mnist$train$images
y_train <- mnist$train$labels

y_train <- apply(y_train, 1, function(r) { which.max(r) - 1 })


