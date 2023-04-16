library(caret)

# Load the iris dataset
data(iris)

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
train <- iris[trainIndex, ]
test <- iris[-trainIndex, ]

# Train a classification model using k-Nearest Neighbors
model <- train(Species ~ ., data = train, method = "knn")

# Use the model to make predictions on the test set
predictions <- predict(model, test)

# Evaluate the accuracy of the model
confusionMatrix(predictions, test$Species)

