import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('blood.csv')
data.columns = data.columns.str.lower().str.strip()

# Feature selection
X = data[['recency', 'frequency', 'monetary', 'time', 'class']]
if 'recommendation' in data.columns:
    y = data['recommendation']
else:
    y = ((data['recency'] > 6) & (data['monetary'] > 500)).astype(int)  

# Define numerical and categorical features
numerical_features = ['recency', 'frequency', 'monetary', 'time']
categorical_features = ['class']

# Preprocessing pipelines
numerical_pipeline = Pipeline(steps=[('scaler', StandardScaler())])
categorical_pipeline = Pipeline(steps=[('encoder', OneHotEncoder(drop='first'))])
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
model_pipeline.fit(X_train, y_train)

# Predictions
y_pred = model_pipeline.predict(X_test)

# Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualizations
sns.set_style("whitegrid")

# Count plot of recommendations
plt.figure(figsize=(8, 5))
sns.countplot(x=y, hue=y, palette='coolwarm', legend=False)
plt.title("Distribution of Healthcare Recommendations")
plt.xlabel("Recommendation Type")
plt.ylabel("Count")
plt.xticks(rotation=15)
plt.show()

# Scatter plot of recency vs. frequency
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X['recency'], y=X['frequency'], hue=y, palette='coolwarm', s=100)
plt.title("Recency vs. Frequency of Blood Donations")
plt.xlabel("Recency")
plt.ylabel("Frequency")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Pair plot
sns.pairplot(X, hue=y, palette='coolwarm')
plt.show()

# Feature distributions
for col in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(X[col], kde=True, bins=30, color='blue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

# ROC Curve (if binary classification)
if len(set(y_test)) == 2:
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
