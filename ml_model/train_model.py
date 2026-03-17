import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Get the correct path for the CSV file ─────────────────────────────────
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Path to the CSV file in project root
CSV_PATH = os.path.join(PROJECT_ROOT, 'HR_Employee_Attrition_Dataset.csv')

print(f"📊 Loading HR dataset from: {CSV_PATH}")

try:
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Attrition distribution:\n{df['Attrition'].value_counts()}")
except FileNotFoundError:
    print(f"❌ Error: Could not find the CSV file at {CSV_PATH}")
    print("Please make sure 'HR_Employee_Attrition_Dataset.csv' is in the project root folder.")
    exit(1)

# ─── Data Cleaning ─────────────────────────────────────────────────────────
# Remove EmployeeNumber if it exists (it's just an ID)
if 'EmployeeNumber' in df.columns:
    df = df.drop('EmployeeNumber', axis=1)

# Check for any missing values
print(f"\n🔍 Checking for missing values...")
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print("⚠️ Missing values found:")
    print(missing_values[missing_values > 0])
    # Fill missing values with appropriate defaults
    df = df.fillna({
        'MonthlyIncome': df['MonthlyIncome'].median(),
        'YearsAtCompany': 0,
        'TotalWorkingYears': 0,
    })
else:
    print("✅ No missing values found")

# ─── Preprocessing ──────────────────────────────────────────────────────────
print(f"\n🔧 Encoding categorical variables...")
label_encoders = {}
categorical_cols = ['BusinessTravel', 'Department', 'EducationField',
                    'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"  ✅ Encoded: {col}")

target_le = LabelEncoder()
df['Attrition'] = target_le.fit_transform(df['Attrition'])  # No=0, Yes=1
label_encoders['Attrition'] = target_le
print(f"  ✅ Encoded: Attrition (0=Stay, 1=Leave)")

# ─── Feature Engineering ────────────────────────────────────────────────────
print(f"\n🛠️ Creating derived features...")

# Create derived features safely (handling potential NaN values)
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5])
df['AgeGroup'] = df['AgeGroup'].cat.add_categories([0]).fillna(0).astype(int)

df['TenureGroup'] = pd.cut(df['YearsAtCompany'], bins=[-1, 2, 5, 10, 20, 100], labels=[1, 2, 3, 4, 5])
df['TenureGroup'] = df['TenureGroup'].cat.add_categories([0]).fillna(0).astype(int)

df['IncomePerYear'] = df['MonthlyIncome'] * 12

# Calculate satisfaction scores safely
satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 
                     'RelationshipSatisfaction', 'WorkLifeBalance']
for col in satisfaction_cols:
    if col not in df.columns:
        df[col] = 3  # Default value

df['SatisfactionScore'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + 
                           df['RelationshipSatisfaction'] + df['WorkLifeBalance']) / 4

# Create overtime impact safely
if 'OverTime' in df.columns and 'JobSatisfaction' in df.columns:
    df['OverTime_Impact'] = (df['OverTime'] == 1).astype(int) * df['JobSatisfaction']
else:
    df['OverTime_Impact'] = 0

# Create promotion delay safely
df['PromotionDelay'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
df['PromotionDelay'] = df['PromotionDelay'].fillna(0)

print(f"  ✅ Created {len(df.columns) - 32} new features")

# Get all features (excluding target)
feature_cols = [c for c in df.columns if c != 'Attrition']
X = df[feature_cols]
y = df['Attrition']

print(f"\n🔍 Total features before selection: {len(feature_cols)}")
print(f"📊 Training samples: {len(X)}")

# ─── Feature Importance Analysis ────────────────────────────────────────────
print("\n🔍 Analyzing feature importance...")

# Train initial model to get feature importance
initial_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
initial_model.fit(X, y)

# Get feature importances
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': initial_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 15 Most Important Features:")
print("=" * 60)
for idx, row in importances.head(15).iterrows():
    print(f"{row['feature'][:25]:<25} : {row['importance']:.4f}")

# Select top features (reducing to 15 most important)
selector = SelectFromModel(initial_model, threshold='median', max_features=15)
X_selected = selector.fit_transform(X, y)
selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]

print(f"\n✅ Selected {len(selected_features)} most important features:")
print("=" * 60)
for i, feat in enumerate(sorted(selected_features, key=lambda x: importances[importances['feature']==x]['importance'].values[0] if len(importances[importances['feature']==x])>0 else 0, reverse=True), 1):
    importance_val = importances[importances['feature']==feat]['importance'].values[0] if len(importances[importances['feature']==feat])>0 else 0
    print(f"  {i:2d}. {feat:<25} : {importance_val:.4f}")

# ─── Train/Test Split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X[selected_features], y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Training set size: {len(X_train)} samples")
print(f"📊 Test set size: {len(X_test)} samples")
print(f"📊 Training set distribution: {np.bincount(y_train)}")
print(f"📊 Test set distribution: {np.bincount(y_test)}")

# ─── Train Optimized Random Forest ─────────────────────────────────────────
print("\n🌲 Training optimized Random Forest model...")

# Calculate class weights automatically
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

model = RandomForestClassifier(
    n_estimators=200,  # Reduced for faster training
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    class_weight=class_weight_dict,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Cross-validation
try:
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"✅ Cross-validation F1 Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
except Exception as e:
    print(f"⚠️ Cross-validation skipped: {e}")

# Final evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2%}")

print("\n📊 Classification Report:")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=['Stay', 'Leave']))

# Feature importance of final model
final_importances = pd.Series(model.feature_importances_, index=selected_features)
print("\n📊 Final Feature Importances (Selected Features):")
print("=" * 60)
for name, importance in final_importances.sort_values(ascending=False).items():
    print(f"{name[:25]:<25} : {importance:.4f}")

# ─── Save model and artifacts ──────────────────────────────────────────────
save_dir = SCRIPT_DIR  # Save in ml_model folder

print(f"\n💾 Saving model artifacts to: {save_dir}")

# Save model
with open(os.path.join(save_dir, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)
print(f"  ✅ model.pkl saved ({os.path.getsize(os.path.join(save_dir, 'model.pkl')) / 1024:.1f} KB)")

# Save encoders
with open(os.path.join(save_dir, 'encoders.pkl'), 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"  ✅ encoders.pkl saved")

# Save feature columns (selected features only)
with open(os.path.join(save_dir, 'feature_cols.pkl'), 'wb') as f:
    pickle.dump(selected_features, f)
print(f"  ✅ feature_cols.pkl saved ({len(selected_features)} features)")

# Save feature importance for frontend
feature_importance_dict = dict(zip(selected_features, model.feature_importances_))
with open(os.path.join(save_dir, 'feature_importance.pkl'), 'wb') as f:
    pickle.dump(feature_importance_dict, f)
print(f"  ✅ feature_importance.pkl saved")

# Also save the list of original columns for reference
original_cols = [c for c in df.columns if c != 'Attrition']
with open(os.path.join(save_dir, 'original_cols.pkl'), 'wb') as f:
    pickle.dump(original_cols, f)

# Save a summary text file
with open(os.path.join(save_dir, 'model_summary.txt'), 'w') as f:
    f.write("Employee Attrition Predictor - Model Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Training Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset Shape: {df.shape}\n")
    f.write(f"Attrition Distribution: {df['Attrition'].value_counts().to_dict()}\n\n")
    f.write(f"Model Type: Random Forest Classifier\n")
    f.write(f"Number of Features: {len(selected_features)}\n")
    f.write(f"Accuracy: {accuracy:.2%}\n\n")
    f.write("Top 10 Features:\n")
    for name, imp in final_importances.sort_values(ascending=False).head(10).items():
        f.write(f"  {name}: {imp:.4f}\n")

print(f"  ✅ model_summary.txt saved")

print(f"\n🎉 Model training complete!")
print(f"📁 All files saved in: {save_dir}")
print(f"📊 Model uses {len(selected_features)} features with {accuracy:.2%} accuracy")