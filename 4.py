import numpy as np
import pandas as pd

# ================= PERCEPTRON =================
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])  # XOR

learning_rate = 0.1
weights = np.random.rand(2)
bias = np.random.rand(1)

for _ in range(1000):
    output = np.dot(X, weights) + bias
    output = np.where(output > 0, 1, 0)

    error = y - output
    weights += learning_rate * np.dot(X.T, error)
    bias += learning_rate * np.sum(error)

print("Perceptron Output:", output)



# ================= DECISION TREE =================
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data = {
    "Customer_Segment": ["Gold", "Silver", "Bronze", "Gold", "Silver", "Bronze"],
    "Product_Category": ["Electronics", "Clothing", "Food", "Home Decor", "Clothing", "Electronics"],
    "Purchase_Amount": [50, 15, 80, 30, 20, 10],
    "Promotion_Used": ["Yes", "No", "No", "Yes", "Yes", "No"],
    "Sales_Target": ["High", "Medium", "Low", "Medium", "Medium", "Low"]
}

df = pd.DataFrame(data)

# Encoding
for col in ["Customer_Segment","Product_Category","Promotion_Used"]:
    df[col] = pd.Categorical(df[col]).codes

X = df[["Customer_Segment","Product_Category","Purchase_Amount","Promotion_Used"]]
y = df["Sales_Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print(df)