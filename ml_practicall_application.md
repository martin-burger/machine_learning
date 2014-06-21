HAR Data Analysis
========================================================

## 01. Overview

The aim of this study is to learn a model which predicts the manner of excercise people did. The model will be trained with data collected by devices tracking their movements.

## 02. Loading packages and data, split the data

First, the needed packages are loaded, then the data is splited up into training and test data.


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(lubridate)
library(ggplot2)

data <- read.csv("pml-training.csv", na.string = c("NA", ""))

set.seed(123)

inTrain <- createDataPartition(y = data$classe, p = 0.6, list = FALSE)
train.feature <- data[inTrain, -ncol(data)]
train.outcome <- data[inTrain, ncol(data), drop = FALSE]
test.feature <- data[-inTrain, -ncol(data)]
test.outcome <- data[-inTrain, ncol(data), drop = FALSE]
```

## 03. Preprocessing and data cleaning

Deleting Index, user name and one of the two time rows, as long we need just one.


```r
train.feature <- train.feature[, -c(1, 2, 5)]
test.feature <- test.feature[, -c(1, 2, 5)]
```

Splitting up the time row to have the possibilty to learn for hours, weeks and day's
seperate weights.


```r
time.feature <- function(data) {
  
  data[, 1] <- as.POSIXct(data[, 1], origin = "1970-01-01", tz = "UTC")
  data$wday <- wday(data[, 1])
  data$hour <- hour(data[, 1])
  data$minute <- minute(data[, 1])
  data[, -1]

  }

train.feature <- time.feature(train.feature)
test.feature <- time.feature(test.feature)
```
As there are many features with less then 95% of data, we assume that they are not contributing much information to our model.

```r
filter.feature <- function(data, threshold = .95) {
  p.num <- function(x) {
    sum(is.na(x)) / length(x)
  }
  less <- apply(data, 2, p.num)
  which(less >= threshold)
}

less.feature <- filter.feature(train.feature)

train.feature <- train.feature[, -less.feature]

#less.feature <- filter.feature(test.feature)

test.feature <- test.feature[, -less.feature]

 train.feature[, 2] <- as.numeric(train.feature[, 2])

 test.feature[, 2] <- as.numeric(test.feature[, 2])
```

## 04. Train the model with random forest


```r
set.seed(555)
boot.control <- trainControl(number = 5)
rf.model <- train(train.feature, train.outcome$classe, method = "rf", trControl = boot.control)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

Test the performance on the test set of the training data


```r
rf.pred <- predict(rf.model, test.feature)
confusionMatrix(rf.pred, test.outcome$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    1    0    0    0
##          B    0 1516    2    0    0
##          C    0    1 1366    2    0
##          D    0    0    0 1284    2
##          E    0    0    0    0 1440
## 
## Overall Statistics
##                                     
##                Accuracy : 0.999     
##                  95% CI : (0.998, 1)
##     No Information Rate : 0.284     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 0.999     
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.999    0.999    0.998    0.999
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    0.999    0.998    0.998    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.285    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    0.999    0.999    0.999    0.999
```

Loading the original test data which was not included in the training process


```r
test <- read.csv("pml-testing.csv", na.string = c("NA", ""))

test <- test[, -ncol(test)]

test <- test[, -c(1, 2, 5)]

test <- time.feature(test)

test <- test[, -less.feature]

 test[, 2] <- as.numeric(test[, 2])

rf.pred <- predict(rf.model, test)
```


