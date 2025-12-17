import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import joblib

data = pd.read_csv("weatherAUS.csv")

# Checking for missing values
missing_values = data.isnull().sum()

# print(missing_values[missing_values > 0])

df = data.copy()

# Handle missing data in RainToday and RainTomorrow
categorical_features = df.select_dtypes(include=["object"]).columns

# Removing the rows that have a NaN
for column in categorical_features:
    # Handle it by replacing the missing value with the mode of the column
    df[column] = df[column].fillna(df[column].mode()[0])

# Handling missing data for numeric features
num_features = df.select_dtypes(include=["int64", "float64"]).columns

# Handle it by replacing the missing value with the median of the column
df[num_features] = df[num_features].fillna(df[num_features].median())

# Checking for missing values again
missing_values = df.isnull().sum()

# print(missing_values[missing_values > 0])

# Visualizing the count of Yes and No
# plt.figure(figsize=(14, 6))
# sns.countplot(x=df["RainTomorrow"])
# plt.title(f"Count plot of RainTomorrow")
# plt.show()

# Removing useless data for classification
df = df.drop(columns=["Date"])

# Seperating features with number and text
num_features = df.select_dtypes(include=["int64", "float64"]).columns

categorical_features = df.select_dtypes(include=["object"]).columns

# Visualizing the box plot of each numeric column to check for outliers
# for feature in num_features:
#     plt.figure(figsize=(14, 6))
#     sns.boxplot(x=df[feature])
#     plt.title(f"Box plot of {feature}")
#     plt.show()

# Removing the outliers using the IQR Method
for column in num_features:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1


    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index

    df = df.drop(outliers)

# Visualizing the box plot of each numeric column to check if the outliers has been reduce/gone
# for feature in num_features:
#     plt.figure(figsize=(14, 6))
#     sns.boxplot(x=df[feature])
#     plt.title(f"Box plot of {feature}")
#     plt.show()

# Visualizing the histogram of each numeric column
# plt.figure(figsize=(24, 20))
# for i, feature in enumerate(num_features, 1):
#     plt.subplot(4, 5, i)
#     sns.histplot(df[feature], kde=True)
#     plt.title(feature)
# plt.tight_layout
# plt.show()

# Seperating the category for One Hot Encoder
one_hot = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm"]

# # Seperating the category for Label Encoder
label = ["RainToday", "RainTomorrow"]

label_encoder = LabelEncoder()

# # Encoding using the Label Encoder
for col in label:
    df[col] = label_encoder.fit_transform(df[col])

# Checking to see if the Encoding was successful
# print(df.to_string())

# preprocessing the data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), num_features),
        ("one_hot", OneHotEncoder(handle_unknown="ignore"), one_hot),
    ]
)

"""
Seperating the data
x = the value that can help to predict by the model
y = the value that want to be predicted by the model
"""
x = df.drop(columns=["RainTomorrow"])
y = df["RainTomorrow"]

# Splitting x and y to train and test the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train_p = preprocessor.fit_transform(x_train)
x_test_p  = preprocessor.transform(x_test)

# Checking the length of the variable
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# Training the model using Random Forest Classification
random_forest = RandomForestClassifier(class_weight="balanced", n_estimators=300, max_depth=6, min_samples_leaf=20).fit(x_train_p, y_train)

# Predicting the results using the model
y_pred = random_forest.predict(x_test_p)

# Summary preformance of the model
summary_df = classification_report(y_test, y_pred)
f1ScoreYES = f1_score(y_test, y_pred)
f1ScoreNO = f1_score(y_test, y_pred, pos_label=0)

# Printing the summary performance of the model
print(summary_df)
print(f"F1 Score (Yes): {f1ScoreYES}")
print(f"F1 Score (No): {f1ScoreNO}")

# Displaying the confusion matrix of the model
confusionMatrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusionMatrix, display_labels=["No", "Yes"]).plot(cmap="Blues")
plt.show()

# Saving the model and the preprocessing into a file
joblib.dump(random_forest, "randomForest.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")