import joblib
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

# Load the saved model and the preprocessing file
random_forest = joblib.load("randomForest.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Read the csv file for testing with new data
test_data = pd.read_csv("test.csv")

# Removed the columns that are not needed for inputing into the model
inputed_data = test_data.drop(columns=["Date", "RainTomorrow"])

# The real value of Y
y_test = test_data["RainTomorrow"]

# Preprocessing the new data
inputed_data = preprocessor.transform(inputed_data)

# Predict the y with the model and the data we inputed
predicted = random_forest.predict(inputed_data)

predicted_values = []

# Changing the 0 into No and, 1 into Yes
for value in predicted:
    if value == 0:
        predicted_values.append("No")
    else:
        predicted_values.append("Yes")

# Displaying the Confusion matrix
cm = confusion_matrix(y_test, predicted_values)
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.show()