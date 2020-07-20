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



# Simple Linear Regression ------------------------------------------------

dataset = read.csv('C:/Users/sridevi.tolety/Documents/StudyReferences/MachineLearning_AtoZ_Udemy/Salary_Data.csv')

library(caTools)
set.seed(123)

split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

regModel = lm(Salary~YearsExperience, data = training_set)

y_pred = predict(regModel, newdata=test_set)

# Visualizing the training set results 
library(ggplot2)

ggplot() + geom_point(aes(x=training_set$YearsExperience, y=training_set$Salary), color = 'red') + 
  geom_line(aes(x=training_set$YearsExperience, y = predict(regModel, newdata=training_set)), color ='blue') + 
  ggtitle('Salary vs Years of Experience') +
  xlab('Years of Experience') +
  ylab('Salary')

# Visualizing the training set results 
library(ggplot2)

ggplot() + geom_point(aes(x=training_set$YearsExperience, y=training_set$Salary), color = 'red') + 
  geom_line(aes(x=training_set$YearsExperience, y = predict(regModel, newdata=training_set)), color ='blue') + 
  ggtitle('Salary vs Years of Experience') +
  xlab('Years of Experience') +
  ylab('Salary')


# Multiple linear regression ----------------------------------------------

# Importing the dataset
setwd("~/StudyReferences/MachineLearning_AtoZ_Udemy/Part 2 - Regression/Section 5 - Multiple Linear Regression")
dataset = read.csv('50_Startups.csv')

# Taking care of missing data
# dataset$Age <- ifelse(is.na(dataset$Age), 
                      mean(dataset$Age, na.rm = T), 
                      dataset$Age)
# dataset$Salary <- ifelse(is.na(dataset$Salary), 
                         mean(dataset$Salary, na.rm = T), 
                         dataset$Salary)


# Encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c('California', 'Florida', 'New York'),
                         labels = c(2, 3, 1))

# Splitting into test and train set 
library(caTools)
set.seed(123)            
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

# Feature scaling 
regressor = lm(training_set$Profit~., data = training_set)
summary(regressor)
# Only RD Spends are significant
regressor = lm(Profit~R.D.Spend, data = training_set)
summary(regressor)

# Predicting test set results 
test_set$y_pred = predict(regressor, newdata = test_set)

# Building the optimal model usingbackward elimination 
backwardElimination <- function(x, sl){
  numVars = length(x)
  for (i in c(1:numVars)) {
    print(i)
    regressor <- lm(formula = Profit ~., data = x)
    maxVar <- max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if(maxVar>sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"]==maxVar)
      print(j)
      x = x[,-j]
      print(dim(x))
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}
sl = 0.05
backwardElimination(training_set, sl)


# Polynomial Linear Regression --------------------------------------------

dataset = read.csv("~/StudyReferences/MachineLearning_AtoZ_Udemy/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv")

dataset = dataset[2:3]

lin_reg = lm(Salary~Level, data=dataset)
summary(lin_reg)

dataset$Level2 <- dataset$Level^2
dataset$Level3 <- dataset$Level^3
dataset$Level4 <- dataset$Level^4
poly_reg_2 = lm(Salary~., data=dataset)
summary(poly_reg_2)

library(ggplot2)

ggplot(data = dataset) + geom_point(aes(x = Level, y = Salary), color = 'red') +
  geom_line(aes(x = Level, y = predict(lin_reg, newdata = dataset)), color='blue') +
  ggtitle('Linear regression') + 
  xlab('Level') + 
  ylab('Salary')

ggplot(data = dataset) + geom_point(aes(x = Level, y = Salary), color = 'red') + 
  geom_line(aes(x = Level, y = predict(poly_reg_2, newdata = dataset)), color = 'blue') +
  ggtitle('Polynomial regression') + 
  xlab("Level") +
  ylab('Salary')

# Predicitng salary for a new employee 
y_pred = predict(lin_reg, newdata = data.frame(Level=6.5)) # 330379

y_pred = predict(poly_reg_2, newdata = data.frame(Level=6.5, Level2=6.5^2, Level3=6.5^3, Level4=6.5^4)) # 158862



# SUPPORT VECTOR MACHINE --------------------------------------------------
dataset = read.csv("~/StudyReferences/MachineLearning_AtoZ_Udemy/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv")

dataset = dataset[2:3]

library(e1071)

regressor_svr = svm(Salary~., data = dataset, type = 'eps-regression')

(y_pred = predict(regressor_svr, newdata=data.frame(Level=6.5))) # 177,861

ggplot(data = dataset) + geom_point(aes(x = Level, y = Salary), color = 'red') + 
  geom_line(aes(x = Level, y = predict(regressor_svr, newdata = dataset)), color = 'blue') +
  ggtitle('SVR regression') + 
  xlab("Level") +
  ylab('Salary')



# DECISION TREE REGRESSION  -----------------------------------------------
setwd("~/StudyReferences/MachineLearning_AtoZ_Udemy/Part 2 - Regression/Section 8 - Decision Tree Regression")

dataset <- read.csv('Position_Salaries.csv')
library(rpart)
library(ggplot2)

X= dataset[,c(2)]
y = dataset[,c(3)]
y = dataset[2:3]

regressor_tree = rpart(Salary~Level, data=dataset, control = rpart.control(minsplit = 1))

(y_pred = predict(regressor_tree, newdata = data.frame(Level=6.5))) # 250,500

X_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)

ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = 'red') +
  geom_line(aes(x = X_grid, y = predict(regressor_tree, newdata = data.frame(Level=X_grid))), color = 'blue') +
  ggtitle('Decision tree regression') + 
  xlab("Level") +
  ylab('Salary')


# RANDOM FOREST REGRESSION  -----------------------------------------------
setwd("~/StudyReferences/MachineLearning_AtoZ_Udemy/Part 2 - Regression/Section 9 - Random Forest Regression")

dataset <- read.csv('Position_Salaries.csv')
library(randomForest)
library(ggplot2)

X= dataset[,c(2)]
y = dataset[2:3]

regressor_rf = randomForest(Salary~Level, ntree=10, data=dataset)

(y_pred = predict(regressor_rf, newdata = data.frame(Level=6.5))) # 170,600

X_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)

ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = 'red') +
  geom_line(aes(x = X_grid, y = predict(regressor_rf, newdata = data.frame(Level=X_grid))), color = 'blue') +
  ggtitle('Decision tree regression') + 
  xlab("Level") +
  ylab('Salary')

# Logistic Regression -----------------------------------------------------

# Classification template

# Importing the dataset
setwd("~/StudyReferences/MachineLearning_AtoZ_Udemy/Part 3 - Classification/Section 14 - Logistic Regression")
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting classifier to the Training set
# Create your classifier here
lr_classifier = glm(formula = Purchased~., 
                    family = binomial,
                    data = training_set)

# Predicting the Test set results
prob_pred = predict(lr_classifier, type = 'response', newdata = test_set[-3])

y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)
cm

(57+26)/nrow(test_set)# Accuracy 0.83 

# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(lr_classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Classifier (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'Classifier (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# KNN Classification  -----------------------------------------------------

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting classifier to the Training set
# Create your classifier here
library(class)

#Fitting knn to training set and fitting to test set 
y_pred = knn(train = training_set[,-3], test = test_set[,-3], cl = training_set[,3],k = 5)


# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = knn(train = training_set[,-3], test = X_grid, cl = training_set[,3],k = 5)
plot(set[, -3],
     main = 'Classifier (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'Classifier (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# SUPPORT VECTOR MACHINE --------------------------------------------------

dataset <- read.csv('Social_Network_Ads.csv')
dataset <- dataset[3:5]

dataset$Purchased <- factor(dataset$Purchased, levels = c(0,1))

library(caTools)
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = 0.75)

training_set <- subset(dataset, split == TRUE) 
test_set <- subset(dataset, split==FALSE)

training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting SVM to the Training set
library(e1071)
svm_classifier = svm(formula = Purchased ~ .,
                     data = training_set,
                     type = 'C-classification',
                     kernel = 'linear')
# Predicting the Test set results
y_pred = predict(svm_classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)
cm


# Kernel SVM --------------------------------------------------------------

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting Kernel SVM to the Training set
# install.packages('e1071')
library(e1071)
kernelsvm_classifier = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicting the Test set results
y_pred = predict(kernelsvm_classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)


# Naive Bayes  ------------------------------------------------------------

nb_classifier = naiveBayes(x=training_set[-3], y= training_set$Purchased)
y_pred = predict(nb_classifier, newdata = test_set[-3])


# Decision Tree Classifier  -----------------------------------------------

# Decision Tree Classification

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting Decision Tree Classification to the Training set
# install.packages('rpart')
library(rpart)
dt_classifier = rpart(formula = Purchased ~ .,
                      data = training_set)

# Predicting the Test set results
y_pred = predict(dt_classifier, newdata = test_set[-3], type = 'class')

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)


# Random Forest Classifier ------------------------------------------------
library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set[-3],
                          y = training_set$Purchased,
                          ntree = 500)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])



# K-Means Clustering ------------------------------------------------------

dataset = read.csv('Mall_Customers.csv')
X = dataset[,c(4,5)]

#Using the elbow method to find the optimal number of clusters 
set.seed(6)
wcss <- vector()

for (i in 1:10){
  k <- kmeans(X, i)
  wcss[i] <- sum(k$withinss)
}

plot(1:10, wcss, type='b', main=paste('Clusters of clients'), xlab='Number of clusters', ylab='WCSS')

# Applying kmeans to the mall dataset 
set.seed(29)
km <- kmeans(X, 5, iter.max = 300)

#Visualizing the clusters 
library(cluster)
clusplot(X, km$cluster, shade = TRUE, color = TRUE, labels = 2, plotchar = FALSE, span = TRUE, main = paste('Cluster plot'), xlab='Annual income', ylab='Spending score')


# Hierarchical clustering -------------------------------------------------

dataset <- read.csv('Mall_Customers.csv')
X = dataset[,c(4,5)]

# Using the dendogram to find optimal number of clusters 
dendogram <- hclust(dist(X, method = 'euclidean'), method = 'ward.D')
plot(dendogram,
     main = paste("Dendogram"),
     ylab = 'Euclidean distances',
     xlab = 'Customers')

# Fitting hierarchical clustering to the mall dataset 
hc =hclust(dist(X, method = 'euclidean'), method = 'ward.D') 
y_hc = cutree(hc, 5)


# Association Rule Learning  ----------------------------------------------
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
# Each product : 1 column
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep=',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN=10)

# Training apriori on the dataset 
rules = apriori(data=dataset, 
                parameter = list(support=(7*3/7500), confidence=0.8)) # This support sets a minimum support for at least 3 purchases a day, 7 times a wekk divided by total number of transactions 


rules = apriori(data=dataset, 
                parameter = list(support=(7*3/7500), confidence=0.4)) 

# Visualizing the results 
inspect(sort(rules, by='lift')[1:10])

# Reduce the confidence further 
rules = apriori(data=dataset, 
                parameter = list(support=(7*3/7500), confidence=0.2)) 
inspect(sort(rules, by='lift')[1:10])
# Suppose we want items that are purchased at least three times a day.
# Thats 3*7 times a week
# Support = 3*7 / 7500 = 0.0037333 ~ 0.004 
rules = apriori(data=dataset, 
                parameter = list(support=(0.004), confidence=0.2)) 
inspect(sort(rules, by='lift')[1:10])


# Eclat -------------------------------------------------------------------

# Each product : 1 column
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep=',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN=10)

# Training Eclat on the dataset 
rules = eclat(data=dataset, 
                parameter = list(support=0.004, minlen = 2))

# Visualizing the results 
inspect(sort(rules, by='support')[1:10])



# Reinforcement Learning  -------------------------------------------------

##########################################
#   Upper Confidence Bound algorithm
##########################################

# Random Selection

# Importing the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
N = 10000
d = 10
ads_selected = integer(0)
total_reward = 0
for (n in 1:N) {
  ad = sample(1:10, 1)
  ads_selected = append(ads_selected, ad)
  reward = dataset[n, ad]
  total_reward = total_reward + reward
}

# Visualising the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')


# Upper Confidence Bound

# Importing the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
N = 10000
d = 10
ads_selected = integer(0)
numbers_of_selections = integer(d)
sums_of_rewards = integer(d)
total_reward = 0
for (n in 1:N) {
  ad = 0
  max_upper_bound = 0
  for (i in 1:d) {
    if (numbers_of_selections[i] > 0) {
      average_reward = sums_of_rewards[i] / numbers_of_selections[i]
      delta_i = sqrt(3/2 * log(n) / numbers_of_selections[i])
      upper_bound = average_reward + delta_i
    } else {
      upper_bound = 1e400
    }
    if (upper_bound > max_upper_bound) {
      max_upper_bound = upper_bound
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  numbers_of_selections[ad] = numbers_of_selections[ad] + 1
  reward = dataset[n, ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_r
  eward = total_reward + reward
}

# Visualising the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')

# 

# k-Fold Cross Validation -------------------------------------------------


# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting Kernel SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Applying k-Fold Cross Validation
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Purchased, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = Purchased ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
  y_pred = predict(classifier, newdata = test_fold[-3])
  cm = table(test_fold[, 3], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))

# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Kernel SVM (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'Kernel SVM (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# XGBoost -----------------------------------------------------------------

# XGBoost

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

# Encoding the categorical variables as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting XGBoost to the Training set
# install.packages('xgboost')
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[-11]), label = training_set$Exited, nrounds = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = as.matrix(test_set[-11]))
y_pred = (y_pred >= 0.5)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)

# Applying k-Fold Cross Validation
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Exited, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = xgboost(data = as.matrix(training_set[-11]), label = training_set$Exited, nrounds = 10)
  y_pred = predict(classifier, newdata = as.matrix(test_fold[-11]))
  y_pred = (y_pred >= 0.5)
  cm = table(test_fold[, 11], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))


# Natural Language Processing  --------------------------------------------

dataset_original = read.delim(file='Restaurant_Reviews.tsv', quot='',stringsAsFactors = F)
# Cleaning the texts
install.packages('tm')
install.packages('SnowballC')


# Cleaning the texts
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)

corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))

corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

as.character(corpus[[841]])

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)



# Aritificial Neural Network ----------------------------------------------

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

# Encoding the categorical variables as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Fitting ANN to the Training set
# install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
model = h2o.deeplearning(y = 'Exited',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',
                         hidden = c(5,5),
                         epochs = 100,
                         train_samples_per_iteration = -2)

# Predicting the Test set results
y_pred = h2o.predict(model, newdata = as.h2o(test_set[-11]))
y_pred = (y_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)
cm

h2o.shutdown()


# PCA ---------------------------------------------------------------------

# PCA

# Importing the dataset
dataset = read.csv('Wine.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-14] = scale(training_set[-14])
test_set[-14] = scale(test_set[-14])

# Applying PCA
# install.packages('caret')
library(caret)
# install.packages('e1071')
library(e1071)
pca = preProcess(x = training_set[-14], method = 'pca', pcaComp = 2)
training_set = predict(pca, training_set)
training_set = training_set[c(2, 3, 1)]
test_set = predict(pca, test_set)
test_set = test_set[c(2, 3, 1)]

# Fitting SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Customer_Segment ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'SVM (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))


# LDA ---------------------------------------------------------------------

# LDA

# Importing the dataset
dataset = read.csv('Wine.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-14] = scale(training_set[-14])
test_set[-14] = scale(test_set[-14])

# Applying LDA
library(MASS)
lda = lda(formula = Customer_Segment ~ ., data = training_set)
training_set = as.data.frame(predict(lda, training_set))
training_set = training_set[c(5, 6, 1)]
test_set = as.data.frame(predict(lda, test_set))
test_set = test_set[c(5, 6, 1)]

# Fitting SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = class ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('x.LD1', 'x.LD2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Training set)',
     xlab = 'LD1', ylab = 'LD2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('x.LD1', 'x.LD2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'SVM (Test set)',
     xlab = 'LD1', ylab = 'LD2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))


# Kernel PCA --------------------------------------------------------------

# Kernel PCA

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[, 1:2] = scale(training_set[, 1:2])
test_set[, 1:2] = scale(test_set[, 1:2])

# Applying Kernel PCA
# install.packages('kernlab')
library(kernlab)
kpca = kpca(~., data = training_set[-3], kernel = 'rbfdot', features = 2)
training_set_pca = as.data.frame(predict(kpca, training_set))
training_set_pca$Purchased = training_set$Purchased
test_set_pca = as.data.frame(predict(kpca, test_set))
test_set_pca$Purchased = test_set$Purchased

# Fitting Logistic Regression to the Training set
classifier = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training_set_pca)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set_pca[-3])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set_pca[, 3], y_pred)

# Visualising the Training set results
install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
# install.packages('ElemStatLearn')
library(ElemStatLearn)
set = test_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# Grid Search

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting Kernel SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Applying k-Fold Cross Validation
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Purchased, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = Purchased ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
  y_pred = predict(classifier, newdata = test_fold[-3])
  cm = table(test_fold[, 3], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))

# Applying Grid Search to find the best parameters
# install.packages('caret')
library(caret)
classifier = train(form = Purchased ~ ., data = training_set, method = 'svmRadial')
classifier
classifier$bestTune

# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Kernel SVM (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'Kernel SVM (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


