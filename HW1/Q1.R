#Read in the data
Diabete_data <- read.csv(file = "pima-indians-diabetes.csv", header = TRUE, sep = ",")
classes = Diabete_data$X1
N = length(classes)
test_threshold = as.integer(N * 0.2)  #The size of the test section 

itr = 1
while (itr < 10) {
test_data = Diabete_data[sample(nrow(Diabete_data), test_threshold), ]





Num_positive = sum(classes == 1)
Num_negative = N - Num_positive


print(test_threshold)
p1y = Num_positive / N
p2y = Num_negative / N




itr <- itr + 1
}


