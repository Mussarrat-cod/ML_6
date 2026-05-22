import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Load dataset (fix path)
data = pd.read_csv(r"D:\Arr\ML-6\heartdisease.csv")

# Features (X) and Target (Y)
X = data.drop('heartdisease', axis=1)
Y = data['heartdisease']

# Split data (train + test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=42)

# Create model
model = GaussianNB()

# Train model
model.fit(X_train, Y_train)

# Predict
Y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))

# Test with custom input
sample = pd.DataFrame([[2,1,1,0,1,1]], 
                      columns=X.columns)  
result = model.predict(sample)

print("Prediction (0=No Disease, 1=Disease):", result[0])