#Part 1D
library(klaR)
library(caret)
Diabete_data <- read.csv(file = "pima-indians-diabetes.csv", header = TRUE, sep = ",")
path = '~/svm_light/'
accuracy = double(10)

itr = 1
while (itr < 11) {

  train_Index <- createDataPartition(Diabete_data$X1, p = 0.2, list = FALSE, times = 1)
  train_feature <- Diabete_data[train_Index, 1:8]
  train_label <- Diabete_data$X1[train_Index]
  test_data <- Diabete_data[-train_Index, 1:8]
  test_Y <- Diabete_data$X1[-train_Index]

  svm_result <- svmlight(train_feature, train_label, pathsvm = path)
  prediction <- predict(svm_result, test_data)
  
  valid_num <- 0
  for (i in 1:length(test_Y)) {
    if (prediction[["class"]][i] == test_Y[i]) {
      valid_num <- valid_num + 1
    } 
  }
  accuracy[itr] = valid_num / length(test_Y)
  itr <- itr + 1
}
print(accuracy)