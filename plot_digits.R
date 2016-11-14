library(tensorflow)
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

training_set <- mnist$train
images <- training_set$images
labels <- training_set$labels

label_1 <- labels[1,]
image_1 <- images[1,]

grayscale <- colorRampPalette(c('white','black'))
par(mar=c(1,1,1,1), mfrow=c(8,8),pty='s',xaxt='n',yaxt='n')

for(i in 1:60)
{
  z<-array(images[i,],dim=c(28,28))
  z<-z[,28:1] ##right side up
  image(1:28,1:28,z,main=which.max(labels[i,])-1,col=grayscale(256), , xlab="", ylab="")
}
