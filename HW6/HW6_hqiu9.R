library(MASS)
#Import the data as a DataFrame
housing_data <- read.csv("housing.data", header=FALSE, sep="")

#Part A
#Regress housing price against other variables
housing.lm <- lm(housing_data$V14 ~ housing_data$V1 + housing_data$V2 + housing_data$V3 + housing_data$V4 + 
              housing_data$V5 + housing_data$V6 + housing_data$V7 + housing_data$V8 + housing_data$V9 + 
              housing_data$V10 + housing_data$V11 + housing_data$V12 + housing_data$V13)

housing.stdres <- rstandard(housing.lm)  #Calculate the standard residuals 
plot(housing.stdres, housing.lm$fitted.values, ylab = "Fitted Values", xlab = "Standardized Residual")
title("Standardized Residual vs.Fitted Values (Without Transform)")

#Produce diagnostic plots
plot(housing.lm, id.n = 10)

#Part B
#Remove the outliers from the DataFrame and produce new plots
housing_rm_data <- housing_data[-c(369, 371, 372, 373, 365, 370, 413),]
housing_rm.lm <- lm(housing_rm_data$V14 ~ housing_rm_data$V1 + housing_rm_data$V2 + housing_rm_data$V3 + housing_rm_data$V4 + 
                    housing_rm_data$V5 + housing_rm_data$V6 + housing_rm_data$V7 + housing_rm_data$V8 + housing_rm_data$V9 + 
                    housing_rm_data$V10 + housing_rm_data$V11 + housing_rm_data$V12 + housing_rm_data$V13)  
plot(housing_rm.lm, id.n = 2)


#Part C
box_transform <- boxcox(housing_rm.lm)
best_lambda <- box_transform$x[which(box_transform$y == max(box_transform$y))]

#Part D
housing_box_price <- (housing_rm_data$V14 ** best_lambda - 1) / best_lambda
housing_box.lm <- lm(housing_box_price ~ housing_rm_data$V1 + housing_rm_data$V2 + housing_rm_data$V3 + housing_rm_data$V4 + 
                 housing_rm_data$V5 + housing_rm_data$V6 + housing_rm_data$V7 + housing_rm_data$V8 + housing_rm_data$V9 + 
                 housing_rm_data$V10 + housing_rm_data$V11 + housing_rm_data$V12 + housing_rm_data$V13)

housing_box.stdres <- rstandard(housing_box.lm)
plot(housing_box.stdres, housing_box.lm$fitted.values, ylab = "Fitted Values", xlab = "Standardized Residual")
title("Standardized Residual vs.Fitted Values (After Transform)")




#plot(housingLM3)
trans <- (housingLM3$fitted.values * best_lambda + 1) ** (1 / best_lambda)
plot(housing_rm$V14, trans, col = 'blue', pch = 1, ylab = "Fitted House Price", xlab = "Real House Price")
title("Fitted House Price vs. Real House Price")
#legend("topright", legend = c("Fitted Price", "True Price"), pch = c(1,4), col = c("blue","red"))



