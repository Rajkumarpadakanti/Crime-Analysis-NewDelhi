

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Setup
sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv('Downloads/crime rate prediction/data/Updated_New_Delhi_Crime_Dataset.csv')  # Ensure correct path

# Feature Engineering
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df_cleaned = df.drop(columns=['Crime_ID', 'Date', 'Time'])

#  One-hot Encoding
df_encoded = pd.get_dummies(df_cleaned, columns=[
    'Crime_Type', 'Location', 'Status', 'Victim_Gender', 'Suspect_Gender'
])

# KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df_encoded['Crime_Cluster'] = kmeans.fit_predict(df_encoded)

# Merge cluster back for insights
df_with_clusters = df.copy()
df_with_clusters['Crime_Cluster'] = df_encoded['Crime_Cluster']

#  Cluster by Location
plt.figure(figsize=(14, 8))
location_cluster_summary = df_with_clusters.groupby(['Crime_Cluster', 'Location']).size().reset_index(name='Count')
sns.barplot(data=location_cluster_summary, x='Location', y='Count', hue='Crime_Cluster', palette='Set2')
plt.title("Crime Clusters by Location in New Delhi")
plt.xlabel("Location")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#  Train-Test Split
X = df_encoded.drop(columns='Crime_Cluster')
y = df_encoded['Crime_Cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print("🔹 Random Forest Accuracy:", acc_rf)
print(classification_report(y_test, y_pred_rf))

# Logistic Regression
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)
print("🔹 Logistic Regression Accuracy:", acc_lr)
print(classification_report(y_test, y_pred_lr))

# Support Vector Machine
sv = svm.LinearSVC(max_iter=20000)
sv.fit(X_train_scaled, y_train)
y_pred_svm = sv.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, y_pred_svm)
print("🔹 SVM Accuracy:", acc_svm)
print(classification_report(y_test, y_pred_svm))

# Accuracy Comparison
models = ['Random Forest', 'Logistic Regression', 'SVM']
scores = [acc_rf, acc_lr, acc_svm]

plt.figure(figsize=(8, 6))
sns.barplot(x=models, y=scores, palette='Set2')
plt.ylabel('Accuracy')
plt.title('Model Comparison: RF vs LR vs SVM')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

#  Top Crime Locations
location_counts = df['Location'].value_counts().reset_index()
location_counts.columns = ['Location', 'Total_Crimes']
plt.figure(figsize=(12, 6))
sns.barplot(data=location_counts.head(10), x='Location', y='Total_Crimes', palette='rocket')
plt.title("Top 10 Crime-Prone Locations in New Delhi")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Crime by Year and Month
yearly_trend = df['Year'].value_counts().sort_index()
monthly_trend = df['Month'].value_counts().sort_index()

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
yearly_trend.plot(kind='bar', color='skyblue')
plt.title("Crimes per Year")
plt.xlabel("Year")
plt.ylabel("Number of Crimes")

plt.subplot(1, 2, 2)
monthly_trend.plot(kind='bar', color='orange')
plt.title("Crimes per Month")
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.tight_layout()
plt.show()

# Top Crime Types
crime_types = df['Crime_Type'].value_counts().reset_index()
crime_types.columns = ['Crime_Type', 'Total_Crimes']
plt.figure(figsize=(8, 8))
plt.pie(crime_types['Total_Crimes'].head(6),
        labels=crime_types['Crime_Type'].head(6),
        autopct='%1.1f%%', startangle=140)
plt.title("Most Common Crime Types in New Delhi")
plt.axis('equal')
plt.tight_layout()
plt.show()
