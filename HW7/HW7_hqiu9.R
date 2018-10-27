setwd('d:/my_repo/CS498_AML/HW7')
library(glmnet)
library(data.table)
library(doParallel)
registerDoParallel(cores = 6)

blog.train <- read.csv('BlogFeedback/blogData_train.csv', header = FALSE)
blog.train.xmat <- as.matrix(blog.train[, -c(281)])
blog.train.ymat <- as.matrix(blog.train[, 281])
dmodel <- cv.glmnet(blog.train.xmat, blog.train.ymat, family='poisson', parallel = TRUE)
plot(dmodel)



tumor_feature <- fread('http://genomics-pubs.princeton.edu/oncology/affydata/I2000.html')
tumor_Y <- read.csv('tissue.txt', header = FALSE)
tumor_feature <- tumor_feature[,-c(1,2,3)]
tumor_feature$V4 <- substring(tumor_feature$V4, 10)
tumor_feature$V65 <- substr(tumor_feature$V65, 1, 
                            nchar(tumor_feature$V65)-11)
tumor_feature[1, 62] = 7.4720100e+003
tumor <- as.matrix(sapply(tumor_feature, as.numeric))
tumor <- t(tumor)

tumor.xmat <- as.matrix(tumor)
tumor.ymat = as.matrix(apply(tumor_Y, 1, FUN=function(x){if (x > 0) {x = 0} else {x = 1}}))
tumor.glm <- cv.glmnet(tumor.xmat, tumor.ymat, family='binomial', type.measure = 'class', parallel = TRUE)
plot(tumor.glm)
