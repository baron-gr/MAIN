# Create some example actual and predicted class labels
actual <- factor(c("A", "B", "C", "D"))
predicted <- factor(c("A", "B", "C", "D"))

# Create a confusion matrix
confusion_matrix <- table(actual, predicted)

# Print the confusion matrix
confusion_matrix
