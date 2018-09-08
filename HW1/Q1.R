#Read in the data
library(caret)
Diabete_data <- read.csv(file = "pima-indians-diabetes.csv", header = TRUE, sep = ",")

# itr = 1
# #while (itr < 10) {

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


  for (property in 1:N) {
       curr_prop = Diabete_data[[1]][train_Index]
       positive_attr <- curr_prop[location_Y1]
       negative_attr <- curr_prop[location_Y0]
       
       print(positive_attr)
       print(negative_attr)
       # print(train_Index)
       # print(curr_prop)
       X1_Means[1] = mean(curr_prop)
       X1_Stdev[1] = sd(curr_prop)
       
       
  } 







# itr <- itr + 1
# }


