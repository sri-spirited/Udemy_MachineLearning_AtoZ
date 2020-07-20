# Data Preprocessing Template

# Importing the dataset
setwd("C:/Users/sridevi.tolety/StudyReferences/MachineLearning_AtoZ_Udemy/Part 1 - Data Preprocessing")
dataset = read.csv('Data.csv')

# Taking care of missing data
dataset$Age <- ifelse(is.na(dataset$Age), 
                            mean(dataset$Age, na.rm = T), 
                            dataset$Age)
dataset$Salary <- ifelse(is.na(dataset$Salary), 
                      mean(dataset$Salary, na.rm = T), 
                      dataset$Salary)


# Encoding categorical data
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3))

dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0, 1))



# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Feature Scaling
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])

