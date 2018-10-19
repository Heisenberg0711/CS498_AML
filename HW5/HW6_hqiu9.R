library(MASS)
#Import the data as a DataFrame
housing <- read.csv("housing.data", header=FALSE, sep="")

#Part A
#Regress housing price against other variables
housingLM1 <- lm(housing$V14 ~ housing$V1 + housing$V2 + housing$V3 + housing$V4 + housing$V5 + housing$V6 + housing$V7 + housing$V8 + housing$V9 + housing$V10 + housing$V11 + housing$V12 + housing$V13)

#Produce diagnostic plots
plot(housingLM1, id.n = 10)


#Part B
#Remove the outliers from the DataFrame and produce new plots
housing_rm <- housing[-c(369, 371, 372, 373, 365, 370, 413),]
housingLM2 <- lm(housing_rm$V14 ~ housing_rm$V1 + housing_rm$V2 + housing_rm$V3 + housing_rm$V4 + housing_rm$V5 + housing_rm$V6 + housing_rm$V7 + housing_rm$V8 + housing_rm$V9 + housing_rm$V10 + housing_rm$V11 + housing_rm$V12 + housing_rm$V13)
plot(housingLM2, id.n = 2)


#Part C
box_transform <- boxcox(housingLM2)
best_lambda <- box_transform$x[which(box_transform$y == max(box_transform$y))]

#Part D
housing_box <- (housing_rm$V14 ** best_lambda - 1) / best_lambda
housingLM3 <- lm(housing_box ~ housing_rm$V1 + housing_rm$V2 + housing_rm$V3 + housing_rm$V4 + housing_rm$V5 + housing_rm$V6 + housing_rm$V7 + housing_rm$V8 + housing_rm$V9 + housing_rm$V10 + housing_rm$V11 + housing_rm$V12 + housing_rm$V13)

#plot(housingLM3)
trans <- (housingLM3$fitted.values * best_lambda + 1) ** (1 / best_lambda)
plot(housing_rm$V14, trans, col = 'blue', pch = 1, ylab = "Fitted House Price", xlab = "Real House Price")
title("Fitted House Price vs. Real House Price")
#legend("topright", legend = c("Fitted Price", "True Price"), pch = c(1,4), col = c("blue","red"))




