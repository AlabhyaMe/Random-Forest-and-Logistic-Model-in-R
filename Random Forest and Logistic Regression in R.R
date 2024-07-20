library(ggplot2)
library(cowplot)
library(randomForest)
library(tidymodels)
library(ROCR)

rm(list=ls())
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
data <- read.csv(url, header=FALSE)
###########################################################
#Data Cleaning
colnames(data) <- c(
  "age",
  "sex",# 0 = female, 1 = male
  "cp", # chest pain 
  # 1 = typical angina, 
  # 2 = atypical angina, 
  # 3 = non-anginal pain, 
  # 4 = asymptomatic
  "trestbps", # resting blood pressure (in mm Hg)
  "chol", # serum cholestoral in mg/dl
  "fbs",  # fasting blood sugar if less than 120 mg/dl, 1 = TRUE, 0 = FALSE
  "restecg", # resting electrocardiographic results
  # 1 = normal
  # 2 = having ST-T wave abnormality
  # 3 = showing probable or definite left ventricular hypertrophy
  "thalach", # maximum heart rate achieved
  "exang",   # exercise induced angina, 1 = yes, 0 = no
  "oldpeak", # ST depression induced by exercise relative to rest
  "slope", # the slope of the peak exercise ST segment 
  # 1 = upsloping 
  # 2 = flat 
  # 3 = downsloping 
  "ca", # number of major vessels (0-3) colored by fluoroscopy
  "thal", # this is short of thalium heart scan
  # 3 = normal (no cold spots)
  # 6 = fixed defect (cold spots during rest and exercise)
  # 7 = reversible defect (when cold spots only appear during exercise)
  "hd" # (the predicted attribute) - diagnosis of heart disease 
  # 0 if less than or equal to 50% diameter narrowing
  # 1 if greater than 50% diameter narrowing
)


str(data) 
data[data == "?"] <- NA #some values are ? which we will make NA


data[data$sex == 0,]$sex <- "F"
data[data$sex == 1,]$sex <- "M"
data$sex <- as.factor(data$sex)

#changing int variables as factor
data$cp <- as.factor(data$cp)
data$fbs <- as.factor(data$fbs)
data$restecg <- as.factor(data$restecg)
data$exang <- as.factor(data$exang)
data$slope <- as.factor(data$slope)

data$ca <- as.integer(data$ca) # since this column had "?"s in it (which
data$ca <- as.factor(data$ca)  # ...then convert the integers to factor levels

data$thal <- as.integer(data$thal) # "thal" also had "?"s in it.
data$thal <- as.factor(data$thal)

## This next line replaces 0 and 1 with "Healthy" and "Unhealthy"
data$hd <- ifelse(test=data$hd == 0, yes="Healthy", no="Unhealthy")
data$hd <- as.factor(data$hd) # Now convert to a factor
head(data,10)

########################################################

set.seed(2059)
df <- data %>% drop_na()
split_data <- initial_split(df,strata = hd)
train <- training(split_data)
test <- testing(split_data)
table(df$hd)
print(160/297)
print(137/297)
#####################################################
#Random Forest
set.seed(2059)
ntree_values <- seq(100,1000,by=100)
tree <- NA
iter <- NA
optimal <-Inf

for(i in 1:10){
  for(j in seq_along(ntree_values)){
    temp <- randomForest(hd~.,data=train, mtry=i,ntree=ntree_values[j])
    
    optimal_temp <- temp$err.rate[nrow(temp$err.rate),1]
    if (optimal_temp < optimal){
      tree <- ntree_values[j]
      iter <- i
      optimal = optimal_temp
    }
  }
}

paste('Best Tree is',tree)
paste('Best ntry is',iter)

rf_model <- randomForest(hd~.,data=train,iter=iter,ntree=tree)
rf_model
head(rf_model$err.rate)

error_data <- as.data.frame(rf_model$err.rate)

error_data <- error_data %>% pivot_longer(
  cols = c(OOB, Healthy, Unhealthy),
  names_to = 'Type',
  values_to = 'Value'
)
error_data$Tree <- rep(seq(1,tree),3)
head(error_data)

rf <- randomForest(hd~.,data=train,iter=iter,ntree=1000)

error_df <- as.data.frame(rf$err.rate)

error_df <- error_df %>% pivot_longer(
  cols = c(OOB, Healthy, Unhealthy),
  names_to = 'Type',
  values_to = 'Value'
)
error_df$Tree <- rep(seq(1,1000),3)
ggplot(data = error_df, aes(x = Tree, y = Value)) +
  geom_line(aes(color = Type)) + theme_classic()

importance(rf_model)

varImpPlot(rf_model) 
rf_predictions <- predict(rf_model,test,type="response") 
table(rf_predictions,test$hd)
confusionMatrix <- table(rf_predictions,test$hd)
sum(diag(confusionMatrix)) / sum(confusionMatrix)
#####################################################################
#Logistic Model
lr <- glm(data=train, hd~., family = binomial)
summary(lr)

lr_predictions <-predict(lr,test,type = 'response')
ROCRpred <- prediction(lr_predictions,test$hd)
ROCRperf <- performance(ROCRpred, "tpr",'fpr')
plot(ROCRperf,colorize=T,print.cutoffs.at=seq(0,1,0.1), text.adj=c(-0.5,2))

threshold <- seq(0.1,1,0.1)
a<-c()
t=c()
for(i in threshold){
  lr_pred <- ifelse(lr_predictions <=i,"Pred_Healthy","Pred_Unhealthy")
  confusionmatrix <- table(lr_pred, test$hd)
  acc <- sum(diag(confusionmatrix)) / sum(confusionmatrix)
  a <- c(a,acc)
  t<- c(t,i)
}
print(a)

print(t)
print(max(a))
lr_pred <- ifelse(lr_predictions <=0.7,"Pred_Healthy","Pred_Unhealthy")
confusionmatrix <- table(lr_pred, test$hd)
confusionmatrix

lr_pred <- ifelse(lr_predictions <=0.5,"Pred_Healthy","Pred_Unhealthy")
confusionmatrix <- table(lr_pred, test$hd)
confusionmatrix