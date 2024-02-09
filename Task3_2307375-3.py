# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical computations
import seaborn as sns  # For data visualization
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.preprocessing import StandardScaler  # For standardizing features
from sklearn.neural_network import MLPClassifier  # Multi-layer Perceptron Classifier
from sklearn.naive_bayes import GaussianNB  # Gaussian Naive Bayes Classifier
from sklearn.metrics import (
    roc_curve, roc_auc_score, classification_report, accuracy_score
)  # Metrics for model evaluation
from matplotlib.colors import Normalize, ListedColormap  # For colormap and normalization
from sklearn.linear_model import LogisticRegression  # Logistic Regression Classifier
from sklearn.metrics import roc_curve, auc  # Receiver Operating Characteristic (ROC) curve

# For Suppressing warnings
import warnings
warnings.filterwarnings('ignore')


# Loading dataset
data = pd.read_csv('C:/Users/Faraz Yusuf Khan/Desktop/Data/nba_rookie_data.csv')

# Missing value analysis
missing_values = data.isnull().sum()
data = data.dropna()

# Data types analysis
data_types = data.dtypes

# Create a correlation matrix, excluding the first column (player names)
correlation_matrix = data.iloc[:, 1:].corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
# Save the plot as an image (e.g., as a PNG file)
plt.savefig('correlation_matrix.png')
plt.show()

# Select input features and target variable
X = data.iloc[:, [3, 5]] #points per game and field goal attempts

y = data.iloc[:, -1]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Make Models
#Neural Network
model = MLPClassifier(hidden_layer_sizes=(10, 10), activation='logistic', max_iter=2000, random_state=42)
#Gaussian Naïve Bayes
gnb_model = GaussianNB()
#Logistic Regression
logreg_model = LogisticRegression(random_state=42)

# Train the models
model.fit(X_train, y_train)     
gnb_model.fit(X_train, y_train) 
logreg_model.fit(X_train, y_train) 

# Make predictions
y_pred = model.predict(X_test)
y_pred_gnb = gnb_model.predict(X_test)
y_pred_logreg = logreg_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy for Artificial Neural Network (ANN):", accuracy)
print('Number of mislabeled points for ANN out of a total %d points : %d'
% (X_test.shape[0], (y_test != model.predict(X_test)).sum()))
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print("Gaussian Naive Bayes Accuracy:", accuracy_gnb)
print('Number of mislabeled points (GNB) out of a total %d points : %d'
% (X.shape[0], (y != gnb_model.predict(X)).sum()))
print('Logitic Regression %.2f' % logreg_model.score(X, y))
print('Number of mislabeled points out of a total %d points : %d'
% (X.shape[0], (y != logreg_model.predict(X)).sum()))


# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Visualize the Neural Network model
x_min, x_max = X.iloc[:, 0].min() - 0.1, X.iloc[:, 0].max() + 0.1
y_min, y_max = X.iloc[:, 1].min() - 0.1, X.iloc[:, 1].max() + 0.1
h = 0.02  # Step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.column_stack([xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Contour plot of the model using the 'viridis' color map
contour = ax1.contourf(xx, yy, Z, cmap='viridis', alpha=0.6)

# Scatter plot for actual and misclassified values
scatter = ax1.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap='viridis', marker='o', edgecolor='k', s=100, label='Predicted')
misclassified_points = X_test[y_test != y_pred]
misclassified_scatter = ax1.scatter(misclassified_points.iloc[:, 0], misclassified_points.iloc[:, 1], c='red', marker='*', s=150, label='Misclassified')

# Set axis labels and title
ax1.set_xlabel('Points per Game')
ax1.set_ylabel('Field Goal Attempts')
ax1.set_title('Neural Network Decision Boundary')

# Create a legend for both "Predicted" and "Misclassified" points
legend1 = ax1.legend(handles=[scatter, misclassified_scatter], labels=['Predicted', 'Misclassified'], title="Classes")
ax1.add_artist(legend1)

# Add colorbar for the contour plot
cbar1 = fig.colorbar(contour, ax=ax1)
cbar1.set_label('Class Probability')

# Visualize the Logistic Regression model
# Scatter plot for actual values
ax2.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual')

# Scatter plot for predicted values
ax2.scatter(X_test.iloc[:, 0], logreg_model.predict(X_test), color='red', marker='*', label='Predicted')

# Scatter plot for predicted probabilities
ax2.scatter(X_test.iloc[:, 0], logreg_model.predict_proba(X_test)[:, 1], color='green', marker='.', label='Predicted Probability')

ax2.set_xlabel('Points per Game')
ax2.set_ylabel('Outcome')
ax2.legend()
ax2.set_title('Logistic Regression Model')

# Save the combined plot as an image
fig.savefig('Combined_Plot.png')

# Show the combined plot
plt.show()

#############################################
### Complex Models featuring all features ###
#############################################

missing_values = data.isnull().sum()

# Data cleaning and preprocessing
# Handle missing values
data = data.dropna()

# Select input features and target variable
X = data.iloc[:, 1:20]  #  columns 1 to 19 are input features
y = data.iloc[:, -1]    #  the last column is target variable

# Create histograms for all columns
plt.figure(figsize=(19, 18))
for i, column in enumerate(X.columns):
    plt.subplot(4, 5, i + 1)
    sns.histplot(data=X, x=column, kde=True, color='skyblue')
    plt.title(f'Hist. of {column}')
plt.tight_layout()

# Save the plot as an image
plt.savefig('NBA_RookieData_histograms.png')

# Show the plot
plt.show()

# Apply logarithmic transformation to all columns
X_log = np.log1p(X)  

# Create and print histograms for log-transformed data
plt.figure(figsize=(19, 18))
for i, column in enumerate(X_log.columns):
    plt.subplot(4, 5, i + 1)
    sns.histplot(data=X_log, x=column, kde=True, color='skyblue')
    plt.title(f'Hist. of Log({column})')
plt.tight_layout()
plt.savefig('NBA_RookieData_LOGhistograms.png')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_log, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Create Models

#Artificial Neural Network
model = MLPClassifier(hidden_layer_sizes=(10, 10), activation='logistic', max_iter=2000, random_state=42)
#Gaussian Naïve Bayes
gnb_model = GaussianNB()
#Logistic Regression
logreg_model = LogisticRegression(random_state=42)

# Train the models
model.fit(X_train, y_train)
gnb_model.fit(X_train, y_train)
logreg_model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_gnb = gnb_model.predict(X_test)
y_pred_logreg = logreg_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy for All Features Artificial Neural Network:", accuracy)
print('Number of mislabeled points for ANN out of a total %d points : %d'
% (X_test.shape[0], (y_test != model.predict(X_test)).sum()))

accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print("Gaussian Naive Bayes Accuracy for All Features:", accuracy_gnb)
print('Number of mislabeled points (GNB) out of a total %d points : %d'
% (X.shape[0], (y != gnb_model.predict(X)).sum()))

accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f'Accuracy for ALL feature Logistic Regression): {accuracy_logreg:.2f}')
print('Number of mislabeled points out of a total %d points : %d'
% (X.shape[0], (y != logreg_model.predict(X)).sum()))

# ROC Curve for Logistic Regression
y_prob_logreg = logreg_model.predict_proba(X_test)[:, 1]
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_prob_logreg)
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)

plt.figure(figsize=(8, 6))
plt.plot(fpr_logreg, tpr_logreg, color='darkorange', lw=2, label=f'Logistic Regression (AUC = {roc_auc_logreg:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.legend(loc='lower right')
plt.show()























