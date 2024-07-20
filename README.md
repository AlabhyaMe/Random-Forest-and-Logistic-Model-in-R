# Random-Forest-and-Logistic-Model-in-R
In this project, I will use two machine learning models, Random Forest and Logistic Regression, to predict heart disease. This is a simple model where I use all available variables to make the prediction. I won’t delve too deeply into the problem beyond the estimation part.

## Dependencies
Please install the following packages
library(ggplot2)
library(cowplot)
library(randomForest)
library(tidymodels)
library(ROCR)

## Download the data directly from UCI repository.
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
data <- read.csv(url, header=FALSE)

## Follow my blog for more detailed explanation 
https://rpubs.com/Alabhya/Logit_rf OR
https://alabhya.medium.com/using-random-forest-and-logistic-regression-2f8366b5e9a4

