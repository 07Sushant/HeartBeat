# Load necessary libraries

install.packages("readr")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("caret")
install.packages("class")  # Included in the base R package
install.packages("e1071")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("corrplot")

library(readr)
library(corrplot)
library(dplyr)
library(ggplot2)
library(caret)
library(class)
library(e1071)
library(rpart)
library(rpart.plot)

# Load the dataset
heart_data <- read_csv("C:/Local volume/Programming/Machine Learning/Dashboard/heart.csv")

# Initial exploration
str(heart_data)
summary(heart_data)
head(heart_data, 5)

# Basic visualization of data distribution
ggplot(heart_data, aes(x = age)) + geom_histogram(bins = 10, fill = "blue", color = "black") +
  ggtitle("Distribution of Age")

# Correlation matrix visualization
correlations <- cor(heart_data[, sapply(heart_data, is.numeric)])
corrplot(correlations, method = "circle")




# Data Preprocessing

# Check for missing values
sum(is.na(heart_data))

# Impute missing values with the median
heart_data <- heart_data %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))



