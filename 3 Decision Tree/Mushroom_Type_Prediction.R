library(dplyr)

df_mt = read.csv('F:/Imarticus/Projects/DecisionTree_R/Dataset/mushrooms.csv')

str(df_mt)
head(df_mt)

summary(df_mt)

## Value counts on each and every variable
for (col in names(df_mt)) {
  print(paste('Column: ', col))
  print(table(df_mt[[col]]))
}

## Converting all character to factor variables
df_mt[sapply(df_mt, is.character)] <- lapply(df_mt[sapply(df_mt, is.character)], 
                                       as.factor)
## Another way
for (col in names(df_mt)) {
  if (is.character(df_mt[[col]])){
    df_mt[[col]] = as.factor(df_mt[[col]])
  }
}


paste0(round(prop.table(table(df_mt$class))*100,2),'%')
barplot(table(df_mt$class), xlab = "Class", ylab="Count",
        main = 'Count of each Target Class',
        names.arg = c('Edible','Poissionous'))

sam_rows = sample(1:nrow(df_mt), nrow(df_mt) *.75)
length(sam_rows)
df_train_sam = df_mt[sam_rows,]
df_test_sam = df_mt[-sam_rows,]

## Checking whether the train and test split follows the main df proportion
round(prop.table(table(df_mt$class))*100,2)
round(prop.table(table(df_train_sam$class))*100,2)
round(prop.table(table(df_test_sam$class))*100,2)

library(caTools)
## creating train and test sets
dt = sample.split(df_mt$class, SplitRatio = .75)

sample(dt, 10)

## Splitting the dataset as Train and Test
df_mt_train = subset(df_mt, dt == TRUE)
df_mt_test = subset(df_mt, dt == FALSE)

dim(df_mt_train)
dim(df_mt_test)

## Checking whether the train and test split follows the main df proportion
round(prop.table(table(df_mt$class))*100,2)
round(prop.table(table(df_mt_train$class))*100,2)
round(prop.table(table(df_mt_test$class))*100,2)

## Building the tree model
library(rpart)
library(rpart.plot)

mt_fit <- rpart(class ~ . , data = df_mt_train, method = 'class')

mt_fit <- rpart(class~., data = df_mt_train, method = 'class', 
                control=rpart.control(minisplit=100, maxdepth = 3,
                                      minibucket = round(minisplit/3),
                                      cp=0.001))

mt_fit <- rpart(class~., data = df_mt_train, method = 'class', 
                parms = list(split = "gini"))

rpart.plot(mt_fit)
print(mt_fit)

## Predicting unseen data
df_mt_test$pre_class <-predict(mt_fit, select(df_mt_test, -class), type = 'class')

## Evaluating the performance
table_test_matches = table(df_mt_test$class, df_mt_test$pre_class)
table_test_matches

library(caret)

confusionMatrix(as.factor(df_mt_test$pre_class), df_mt_test$class)
## Accuracy of test
accuracy_Test <- sum(diag(table_test_matches)) / sum(table_test_matches)
accuracy_Test

## Evaluating using confusion matrix
library(caret)
confusionMatrix(data = factor(df_mt_test$pre_class), reference = df_mt_test$class)

df_mt_test$class_number = ifelse(df_mt_test$class == 'e', 1, 0)
df_mt_test$predclass_number = ifelse(df_mt_test$pre_class == 'e', 1, 0)

# Plot ROC curve
library(ROCR)
pred <- prediction(df_mt_test$predclass_number,df_mt_test$class_number)
perf <- performance(pred,"tpr","fpr")
plot(perf)

library(ModelMetrics)
library(pROC)
## AUC
auc(roc(df_mt_test$predclass_number,df_mt_test$class_number))
