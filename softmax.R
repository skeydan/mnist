library(tensorflow)
library(glmnet)
library(dplyr)

datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

X_train <- mnist$train$images
y_train <- mnist$train$labels

#fit = glmnet(X_train, y_train, family = "multinomial", type.multinomial = "grouped")
