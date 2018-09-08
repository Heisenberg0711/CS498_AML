#Read in the data
library(caret)
Diabete_data <- read.csv(file = "pima-indians-diabetes.csv", header = TRUE, sep = ",")

#Split the data 10 times
accuracy = double(length = 10)
itr = 1
while (itr < 11) {

train_Index <- createDataPartition(Diabete_data$X1, p = 0.2, list = FALSE, times = 1)

diabete_testY <- Diabete_data$X1[-train_Index]
diabete_trainY <- Diabete_data$X1[train_Index]

p1y = sum(diabete_trainY == 1) / length(diabete_trainY)
p0y = sum(diabete_trainY == 0) / length(diabete_trainY)

location_Y1 <- which(diabete_trainY %in% c(1))
location_Y0 <- which(diabete_trainY %in% c(0))

#Initialize the X vectors for std.distribution
N = length(Diabete_data) - 1
X1_Means = integer(N) 
X1_Stdev = integer(N)
X0_Means = integer(N) 
X0_Stdev = integer(N)

#Get the mean and std for Gaussian distribution
  for (property in 1:N) {
       curr_prop = Diabete_data[[property]][train_Index]
       
       negative_attr <- curr_prop[location_Y0]
       positive_attr <- curr_prop[location_Y1]
       
       X0_Means[property] = mean(negative_attr)
       X0_Stdev[property] = sd(negative_attr)
       
       X1_Means[property] = mean(positive_attr)
       X1_Stdev[property] = sd(positive_attr)
  } 
  

#Begin the validation and calculate the error rate
result = integer(length(diabete_testY))

  for (entry in 1:length(diabete_testY)) {
    prob0 = double(length = N)
    prob1 = double(length = N)
      for (property in 1:N) {
        curr_prop = (Diabete_data[[property]][-train_Index])[entry]
        prob0[property] <- dnorm(curr_prop, mean = X0_Means[property], sd = X0_Stdev[property], log = TRUE)
        prob1[property] <- dnorm(curr_prop, mean = X1_Means[property], sd = X1_Stdev[property], log = TRUE)
      }
      if (sum(prob0) + log(p0y) >= sum(prob1) + log(p1y)) {
        result[entry] <- 0 
      } else {
        result[entry] <- 1
      }
  }

    valid = 0
      for (i in 1: length(diabete_testY)) {
        if (diabete_testY[i] == result[i]) {
          valid <- valid + 1
        }
      }
    accuracy[itr] = valid / length(diabete_testY)

  itr <- itr + 1
}


