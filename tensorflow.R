# http://yann.lecun.com/exdb/mnist/

library(tensorflow)
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

sess <- tf$InteractiveSession()

# images are 55000 * 784
x <- tf$placeholder(tf$float32, shape(NULL, 784L))
# labels are 55000 * 10
y_ <- tf$placeholder(tf$float32, shape(NULL, 10L))

##########################################
# simple neural net without hidden units #
##########################################

# weights are 784 * 10
W <- tf$Variable(tf$zeros(shape(784L, 10L)))
# bias is 10 * 1
b <- tf$Variable(tf$zeros(shape(10L)))

sess$run(tf$initialize_all_variables())

# specify activation function
y <- tf$nn$softmax(tf$matmul(x,W) + b)
# specify loss function
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y), reduction_indices=1L))
# specify optimization method and step size
optimizer <- tf$train$GradientDescentOptimizer(0.5)
train_step <- optimizer$minimize(cross_entropy)

for (i in 1:1000) {
  batches <- mnist$train$next_batch(100L)
  batch_xs <- batches[[1]]
  batch_ys <- batches[[2]]
  sess$run(train_step,
           feed_dict = dict(x = batch_xs, y_ = batch_ys))
}

correct_prediction <- tf$equal(tf$argmax(y, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

# actually evaluate training accuracy
accuracy$eval(feed_dict=dict(x = mnist$train$images, y_ = mnist$train$labels))
# and test accuracy
accuracy$eval(feed_dict=dict(x = mnist$test$images, y_ = mnist$test$labels))


##########################################
#               convnet                  #
##########################################

#  template to initialize weights with a small amount of noise for symmetry breaking
#  and to prevent 0 gradients
weight_variable <- function(shape) {
  initial <- tf$truncated_normal(shape, stddev=0.1)
  tf$Variable(initial)
}

#  template to initialize bias to small positive value to avoid "dead neurons" 
# (as could happen with ReLU)
bias_variable <- function(shape) {
  initial <- tf$constant(0.1, shape=shape)
  tf$Variable(initial)
}

# template to define convolutional layer
# tf$nn$conv2d parameters: input tensor, kernel tensor, strides, padding
# input tensor has shape [batch, in_height, in_width, in_channels] (NHWC)
# kernel tensor has shape [filter_height, filter_width, in_channels, out_channels]
# actions:
## - flatten the filter to a 2-D matrix with shape [filter_height * filter_width * in_channels, output_channels].
## - extract image patches from the input tensor to form a virtual tensor of shape [batch, out_height, out_width, filter_height * filter_width * in_channels].
## - for each patch, right-multiply the filter matrix and the image patch vector.

conv2d <- function(x, W) {
  tf$nn$conv2d(x, W, strides=c(1L, 1L, 1L, 1L), padding='SAME')
}

# template to define max pooling over 2x2 regions
max_pool_2x2 <- function(x) {
  tf$nn$max_pool(
    x, 
    ksize=c(1L, 2L, 2L, 1L),
    strides=c(1L, 2L, 2L, 1L), 
    padding='SAME')
}

#####################################
# conv layer 1: convolution and ReLU #
#####################################

# compute 32 feature maps for each 5x5 patch
# we have just 1 channel
# so shape is: patch size (height, width), number of input channels, number of output channels
W_conv1 <- weight_variable(shape(5L, 5L, 1L, 32L))
# shape for bias: number of output channels
b_conv1 <- bias_variable(shape(32L))

# reshape x from 2d to 4d tensor
# <?>, image width, height, number of color channels
x_image <- tf$reshape(x, shape(-1L, 28L, 28L, 1L))
# perform convolution and ReLU activation
# shape is ?, 28, 28, 32
h_conv1 <- tf$nn$relu(conv2d(x_image, W_conv1) + b_conv1)


####################
# max pool layer 1 #
####################

# shape is ?, 14, 14, 32
h_pool1 <- max_pool_2x2(h_conv1)


######################################
# conv layer 2: convolution and ReLU #
######################################

# next feature map is 5*5, takes 32 channels, produces 64 channels - size weights accordingly
W_conv2 <- weight_variable(shape = shape(5L, 5L, 32L, 64L))
b_conv2 <- bias_variable(shape = shape(64L))
# shape is ?, 14, 14, 64
h_conv2 <- tf$nn$relu(conv2d(h_pool1, W_conv2) + b_conv2)


####################
# max pool layer 2 #
####################

# shape is ?, 7, 7, 64
h_pool2 <- max_pool_2x2(h_conv2)


#########################################
# densely connected layer (with ReLU)   #
# bring together all feature maps       #
#########################################

# weights shape: 3136, 1024 (fully connected)
W_fc1 <- weight_variable(shape(7L * 7L * 64L, 1024L))
b_fc1 <- bias_variable(shape(1024L))
# new shape: ?, 3136
h_pool2_flat <- tf$reshape(h_pool2, shape(-1L, 7L * 7L * 64L))
# matrix multiply and ReLU
# shape: ?, 1024
h_fc1 <- tf$nn$relu(tf$matmul(h_pool2_flat, W_fc1) + b_fc1)


################
# dropout      #
################

keep_prob <- tf$placeholder(tf$float32)
# shape: ?, 1024
h_fc1_drop <- tf$nn$dropout(h_fc1, keep_prob)


################
# softmax layer#
################

W_fc2 <- weight_variable(shape(1024L, 10L))
b_fc2 <- bias_variable(shape(10L))
# shape: ?, 10
y_conv <- tf$nn$softmax(tf$matmul(h_fc1_drop, W_fc2) + b_fc2)


################
# train        #
################


cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y_conv), reduction_indices=1L))
train_step <- tf$train$AdamOptimizer(1e-4)$minimize(cross_entropy)
correct_prediction <- tf$equal(tf$argmax(y_conv, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
sess$run(tf$initialize_all_variables())

for (i in 1:20000) {
  batch <- mnist$train$next_batch(50L)
  if (i %% 100 == 0) {
    train_accuracy <- accuracy$eval(feed_dict = dict(
      x = batch[[1]], y_ = batch[[2]], keep_prob = 1.0))
    cat(sprintf("step %d, training accuracy %g\n", i, train_accuracy))
  }
  train_step$run(feed_dict = dict(
    x = batch[[1]], y_ = batch[[2]], keep_prob = 0.5))
}

################
# accuracy     #
################

train_accuracy <- accuracy$eval(feed_dict = dict(
  x = mnist$train$images, y_ = mnist$train$labels, keep_prob = 1.0))
cat(sprintf("train accuracy %g", train_accuracy))

test_accuracy <- accuracy$eval(feed_dict = dict(
  x = mnist$test$images, y_ = mnist$test$labels, keep_prob = 1.0))
cat(sprintf("test accuracy %g", test_accuracy))
