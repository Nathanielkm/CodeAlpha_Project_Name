import pandas as pd
from pyexpat import features

#load dataset
training_data = pd.read_csv(r"C:\Users\natha\Downloads\Documents\Titanic Dataset\train.csv")
testing_data = pd.read_csv(r"C:\Users\natha\Downloads\Documents\Titanic Dataset\train.csv")

#show the first rows
print(training_data.head())

#Checking for missing values
print(training_data.isnull())

#Summary statistics
print(training_data.describe())

#Check survival rates based on gender
print(training_data.groupby("Sex")["Survived"].mean())

#Checking survival rates based on class
print(training_data.groupby("Pclass")["Survived"].mean())

#Filling in missing Age values with median
training_data["Age"].fillna(training_data["Age"].median())
testing_data["Age"].fillna(testing_data["Age"].median())

# Fill missing Embarked with most common value
training_data["Embarked"].fillna(training_data["Embarked"].mode()[0])
testing_data["Embarked"].fillna(testing_data["Embarked"].mode()[0])

# Convert categorical columns (Sex and Embarked) into numeric values
training_data["Sex"] = training_data["Sex"].map({"male": 0, "female": 1})
testing_data["Sex"] = testing_data["Sex"].map({"male": 0, "female": 1})

training_data["Embarked"] = training_data["Embarked"].map({"C": 0, "Q": 1, "S": 2})
testing_data["Embarked"] = testing_data["Embarked"].map({"C": 0, "Q": 1, "S": 2})

# Drop unnecessary columns
training_data.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
testing_data.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Select features for training
features = ["Pclass","Sex","Age","SibSp","Parch","Fare", "Embarked"]
x = training_data[features]
y = training_data["Survived"]

x_text= testing_data[features]