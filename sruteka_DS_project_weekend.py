"""Linear Regression"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("AB_NYC_2019.csv")
print(df.head())

# PREPROCESSING

find_null = df.isnull().sum()
print(find_null)
df.dropna(inplace=True)

x = df[['host_id','latitude','longitude']]
y = df['price']

print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

# LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
from sklearn import metrics
model = LinearRegression()
model.fit(x_train,y_train)

prediction = model.predict(x_test)
print(metrics.mean_squared_error(y_test,prediction))
print(metrics.r2_score(y_test,prediction))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, prediction, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # Diagonal line
plt.title('Actual Prices vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid()
plt.show()

# CONCLUSION: LINEAR REGRESSION IS NOT A SUITABLE MODEL
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

""" Polynomial Regression"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load Data
df = pd.read_csv("AB_NYC_2019.csv")

# Preprocessing
df.dropna(inplace=True)

x = df[['longitude']].values
y = df['price'].values

# Split the dataset into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 1: Polynomial Feature Transformation
poly_reg = PolynomialFeatures(degree=4)
x_train_poly = poly_reg.fit_transform(x_train)
x_test_poly = poly_reg.transform(x_test)  # Transform test data using the same polynomial transformation

# Step 2: Fit Polynomial Regression Model
line_reg2 = LinearRegression()
line_reg2.fit(x_train_poly, y_train)

# Step 3: Make predictions on the test set
y_pred = line_reg2.predict(x_test_poly)

# Step 4: Evaluate the model
print("Mean Squared Error (MSE):", metrics.mean_squared_error(y_test, y_pred))
print("R-squared (R2):", metrics.r2_score(y_test, y_pred))

# Step 5: Plot Actual Data Points and Polynomial Regression Line (for training data)
x_grid = np.arange(min(x_train), max(x_train), 0.1)  # Create a grid for plotting
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x_train, y_train, color='red', label='Actual Prices (Train)'),plt.plot(x_grid, line_reg2.predict(
poly_reg.fit_transform(x_grid)), color='blue', label='Polynomial Regression (Train)')

plt.title('Polynomial Regression (Using Longitude as Feature)')
plt.xlabel('Longitude')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
""" Support Vector Regression"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

#Load Data
df = pd.read_csv("AB_NYC_2019.csv")

#Preprocessing
df.dropna(inplace=True)

#Feature Scaling (important for SVR)
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

#Split the dataset into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

#Step 1: Train the SVR Model
svr_model = SVR(kernel='rbf')
svr_model.fit(x_train, y_train.ravel())

#Step 2: Predict on the test set
y_pred = svr_model.predict(x_test)

#Step 3: Evaluate the model
y_pred_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1))  # Rescale the predictions back
y_test_rescaled = scaler_y.inverse_transform(y_test)  # Rescale the true values back

print("Mean Squared Error (MSE):", metrics.mean_squared_error(y_test_rescaled, y_pred_rescaled))
print("R-squared (R2):", metrics.r2_score(y_test_rescaled, y_pred_rescaled))

#Step 4: Visualize the SVR predictions

#Plot 1: Scatter plot of Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test_rescaled, y_pred_rescaled, alpha=0.5, color='purple')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # Diagonal line
plt.title('SVR: Actual Prices vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()

#Plot 2: Actual Data Points and SVR Prediction Line (for training data)
x_grid = np.arange(min(x_scaled), max(x_scaled), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x_scaled, scaler_y.inverse_transform(y_scaled), color='red', label='Actual Prices')
plt.plot(x_grid, scaler_y.inverse_transform(svr_model.predict(x_grid).reshape(-1, 1)), color='blue',
         label='SVR Prediction')

plt.title('SVR (Using Longitude as Feature)')
plt.xlabel('Longitude (scaled)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

#Plot 3: Residual Plot (to visualize prediction errors)
residuals = y_test_rescaled - y_pred_rescaled
plt.scatter(y_pred_rescaled, residuals, color='green')
plt.hlines(y=0, xmin=y_pred_rescaled.min(), xmax=y_pred_rescaled.max(), color='red')
plt.title('Residuals Plot')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()
# ----------------------------------------------------------------------------------------------------------------------------------------------

"""Decision Tree Regression"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score

#Load Data
df = pd.read_csv("AB_NYC_2019.csv")

#Preprocessing
df.dropna(inplace=True)

#Choose a single feature for Decision Tree Regression
x = df[['longitude']].values
y = df['price'].values

#Split the dataset into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Step 1: Train the Decision Tree Regressor Model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(x_train, y_train)

#Step 2: Predict on the test set
y_pred = dt_model.predict(x_test)

#Step 3: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

#Step 4: Visualize the Results

#Plot 1: Scatter plot of Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='purple')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # Diagonal line
plt.title('Decision Tree: Actual Prices vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()

#Plot 2: Decision Tree Regression Prediction Line (for training data)
x_grid = np.arange(min(x), max(x), 0.01)  # Smaller steps for clearer visualization
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color='red', label='Actual Prices')
plt.plot(x_grid, dt_model.predict(x_grid), color='blue', label='Decision Tree Prediction')

plt.title('Decision Tree Regression (Using Longitude as Feature)')
plt.xlabel('Longitude')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

#Plot 3: Visualizing the Decision Tree Structure
plt.figure(figsize=(16, 10))
plot_tree(dt_model, filled=True, feature_names=['longitude'], rounded=True)
plt.title('Decision Tree Structure')
plt.show()
# ----------------------------------------------------------------------------------------------------------------------------------------------------

"""Random Forest Regressor"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Data
df = pd.read_csv("AB_NYC_2019.csv")

# Preprocessing
df.dropna(inplace=True)

# Choose a single feature for Random Forest Regression
x = df[['longitude']].values
y = df['price'].values

# Split the dataset into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 1: Train the Random Forest Regressor Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Step 2: Predict on the test set
y_pred = rf_model.predict(x_test)

# Step 3: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Step 4: Visualize the Results

# Plot 1: Scatter plot of Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # Diagonal line
plt.title('Random Forest: Actual Prices vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()

# Plot 2: Random Forest Regression Prediction Line (for training data)
x_grid = np.arange(min(x), max(x), 0.01)  # Smaller steps for smoother curve
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color='red', label='Actual Prices')
plt.plot(x_grid, rf_model.predict(x_grid), color='blue', label='Random Forest Prediction')

plt.title('Random Forest Regression (Using Longitude as Feature)')
plt.xlabel('Longitude')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Plot 3: Visualizing Feature Importance (Random Forest Specific)
# Since we only have one feature, we’ll just plot it for demonstration purposes.
# Usually, you'd use this when you have multiple features.

importances = rf_model.feature_importances_
plt.figure(figsize=(8, 5))
plt.barh(['Longitude'], importances, color='orange')
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""Classification"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("AB_NYC_2019.csv")

# Preprocessing (dropping missing values)
df.dropna(inplace=True)

# For classification, let's create a new binary target variable (e.g., expensive or not)
# We'll classify whether the price is above or below the median price
median_price = df['price'].median()
df['price_category'] = np.where(df['price'] > median_price, 1, 0)  # 1: Expensive, 0: Cheap

# Features and target variable
x = df[['longitude', 'latitude', 'minimum_nights', 'number_of_reviews',
        'calculated_host_listings_count']]
y = df['price_category']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Function to visualize confusion matrix
def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Cheap', 'Expensive'], rotation=45)
    plt.yticks(tick_marks, ['Cheap', 'Expensive'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Function to plot ROC Curve
def plot_roc_curve(y_test, y_proba, title):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


# ================== Classification Models ==================

#1. Logistic Regression
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(random_state=42)
log_model.fit(x_train, y_train)
y_pred = log_model.predict(x_test)
y_proba = log_model.predict_proba(x_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, title="Logistic Regression - Confusion Matrix")

# ROC Curve
plot_roc_curve(y_test, y_proba, title="Logistic Regression - ROC Curve")

# 2. K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)
y_pred = knn_model.predict(x_test)
y_proba = knn_model.predict_proba(x_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, title="KNN - Confusion Matrix")

# ROC Curve
plot_roc_curve(y_test, y_proba, title="KNN - ROC Curve")

# 3. Support Vector Machine (SVM)
from sklearn.svm import SVC

svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(x_train, y_train)
y_pred = svm_model.predict(x_test)
y_proba = svm_model.predict_proba(x_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, title="SVM - Confusion Matrix")

# ROC Curve
plot_roc_curve(y_test, y_proba, title="SVM - ROC Curve")

# 4. Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(x_train, y_train)
y_pred = tree_model.predict(x_test)
y_proba = tree_model.predict_proba(x_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, title="Decision Tree - Confusion Matrix")

#ROC Curve
plot_roc_curve(y_test, y_proba, title="Decision Tree - ROC Curve")

#5. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
y_pred = rf_model.predict(x_test)
y_proba = rf_model.predict_proba(x_test)[:, 1]

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, title="Random Forest - Confusion Matrix")

#ROC Curve
plot_roc_curve(y_test, y_proba, title="Random Forest - ROC Curve")

#6. Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
y_pred = nb_model.predict(x_test)
y_proba = nb_model.predict_proba(x_test)[:, 1]

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, title="Naive Bayes - Confusion Matrix")

#ROC Curve
plot_roc_curve(y_test, y_proba, title="Naive Bayes - ROC Curve")

#7. Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(x_train, y_train)
y_pred = gb_model.predict(x_test)
y_proba = gb_model.predict_proba(x_test)[:, 1]

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, title="Gradient Boosting - Confusion Matrix")

#ROC Curve
plot_roc_curve(y_test, y_proba, title="Gradient Boosting - ROC Curve")

# -------------------------------------------------------------------------------------------------------------------------------------------------------

""" Clustering"""

# K Means

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("AB_NYC_2019.csv")

# Preprocessing
df.dropna(inplace=True)

# Selecting features for clustering
features = df[['latitude', 'longitude', 'number_of_reviews', 'minimum_nights', 'price']]

# Scaling the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Applying K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(features_scaled)

# Visualizing K-Means Clustering
plt.figure(figsize=(10, 6))
plt.scatter(df['longitude'], df['latitude'], c=df['kmeans_cluster'], cmap='viridis', alpha=0.5)
plt.title('K-Means Clustering of Airbnb Listings')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='K-Means Cluster')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------
""" Hierarchical Clustering"""

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Applying Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=5)
df['hierarchical_cluster'] = hierarchical.fit_predict(features_scaled)

# Visualizing Hierarchical Clustering
plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(sch.linkage(features_scaled, method='ward'))
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distances')
plt.show()

# Scatter Plot for Hierarchical Clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['longitude'], df['latitude'], c=df['hierarchical_cluster'], cmap='plasma', alpha=0.5)
plt.title('Hierarchical Clustering of Airbnb Listings')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Hierarchical Cluster')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------------

""" DBScan Clustering"""

from sklearn.cluster import DBSCAN

# Applying DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=10)
df['dbscan_cluster'] = dbscan.fit_predict(features_scaled)

# Visualizing DBSCAN Clustering
plt.figure(figsize=(10, 6))
plt.scatter(df['longitude'], df['latitude'], c=df['dbscan_cluster'], cmap='coolwarm', alpha=0.5)
plt.title('DBSCAN Clustering of Airbnb Listings')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='DBSCAN Cluster')
plt.show()

























