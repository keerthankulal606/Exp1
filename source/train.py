import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (Sepal/Petal length and width)
y = iris.target  # Target labels (Species: Setosa, Versicolor, Virginica)

# 2. Split the data into Training set (80%) and Testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize the Logistic Regression model
# We increase max_iter to 200 to ensure the model converges properly
model = LogisticRegression(max_iter=200)

# 4. Train (fit) the model on the training data
model.fit(X_train, y_train)

# 5. Make predictions on the testing data
y_pred = model.predict(X_test)

# 6. Evaluate how well the model performed
accuracy = accuracy_score(y_test, y_pred)
print(f"=== Model Training Complete ===")
print(f"Accuracy Score: {accuracy * 100:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
