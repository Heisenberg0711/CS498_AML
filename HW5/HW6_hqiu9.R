library(MASS)
#Import the data as a DataFrame
housing <- read.csv("housing.data", header=FALSE, sep="")

#Regress housing price against other variables
housingLM1 <- lm(housing$V14 ~ housing$V1 + housing$V2 + housing$V3 + housing$V4 + housing$V5 + housing$V6 + housing$V7 + housing$V8 + housing$V9 + housing$V10 + housing$V11 + housing$V12 + housing$V13)

#Produce 4 diagnostic plots in 2 by 2 layout 
old_par <- par(mfrow = c(2,2))
plot(housingLM1, id.n = 6)
par(old_par)

#Remove the outliers from the DataFrame and produce new plots
housing_rm <- housing[-c(369, 372, 373, 365, 370, 413),]
housingLM2 <- lm(housing_rm$V14 ~ housing_rm$V1 + housing_rm$V2 + housing_rm$V3 + housing_rm$V4 + housing_rm$V5 + housing_rm$V6 + housing_rm$V7 + housing_rm$V8 + housing_rm$V9 + housing_rm$V10 + housing_rm$V11 + housing_rm$V12 + housing_rm$V13)

old_par_rm <- par(mfrow = c(2,2))
plot(housingLM2, id.n = 6)
par(old_par_rm)

boxcox(housingLM2)
