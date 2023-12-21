from sklearn.datasets import load_breast_cancer,load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from IB3 import IB3  
from sklearn.preprocessing import LabelEncoder


# # Load the Breast Cancer Wisconsin dataset
# breast_cancer = load_breast_cancer()
# X_bc = breast_cancer.data
# y_bc = breast_cancer.target

# # Convert target labels to integers
# label_encoder = LabelEncoder()
# y_bc = label_encoder.fit_transform(y_bc)

# # Split the data into training and testing sets
# X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size=0.2, random_state=42)

# # Instantiate and fit the IB3 classifier
# ib3_classifier_bc = IB3()
# ib3_classifier_bc.fit(X_train_bc, y_train_bc)

# # Make predictions on the test set
# y_pred_bc = ib3_classifier_bc.predict(X_test_bc)

# # Evaluate the accuracy of the classifier
# accuracy_bc = accuracy_score(y_test_bc, y_pred_bc)
# print(f'Accuracy on Breast Cancer Wisconsin dataset: {accuracy_bc}')


#Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert target labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

# Instantiate and fit the IB3 classifier
ib3_classifier = IB3()
ib3_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ib3_classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
