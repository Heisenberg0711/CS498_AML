regnum <- 1:10
print(regnum);

marks <- c(12, 45, 67, 54, 34, 68, 88, 33, 22, 25) # This is of length 10
conduct <- c("A", "B", "A", "A", "C", "B", "B", "A", "B", "A")
df <- data.frame(regnum, marks, conduct)
df$conduct
