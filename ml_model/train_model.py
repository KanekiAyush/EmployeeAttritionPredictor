import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Path setup ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(PROJECT_ROOT, 'HR_Employee_Attrition_Dataset.csv')

print(f"📊 Loading HR dataset from: {CSV_PATH}")

try:
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Dataset loaded! Shape: {df.shape}")
    print(f"Attrition distribution:\n{df['Attrition'].value_counts()}")
except FileNotFoundError:
    print(f"❌ CSV not found at {CSV_PATH}")
    exit(1)

# ─── Data Cleaning ─────────────────────────────────────────────────────
if 'EmployeeNumber' in df.columns:
    df = df.drop('EmployeeNumber', axis=1)

print("\n🔍 Checking missing values...")
if df.isnull().sum().sum() > 0:
    df = df.fillna({
        'MonthlyIncome': df['MonthlyIncome'].median(),
        'YearsAtCompany': 0,
        'TotalWorkingYears': 0,
    })
else:
    print("✅ No missing values")

# ─── Encoding ──────────────────────────────────────────────────────────
print("\n🔧 Encoding categorical variables...")
label_encoders = {}
categorical_cols = ['BusinessTravel', 'Department', 'EducationField',
                    'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

target_le = LabelEncoder()
df['Attrition'] = target_le.fit_transform(df['Attrition'])
label_encoders['Attrition'] = target_le
print("  ✅ All categoricals encoded")

# ─── Feature Engineering (same as before) ──────────────────────────────
print("\n🛠️ Creating derived features...")
df['AgeGroup'] = pd.cut(df['Age'], bins=[0,25,35,45,55,100], labels=[1,2,3,4,5])
df['AgeGroup'] = df['AgeGroup'].cat.add_categories([0]).fillna(0).astype(int)
df['TenureGroup'] = pd.cut(df['YearsAtCompany'], bins=[-1,2,5,10,20,100], labels=[1,2,3,4,5])
df['TenureGroup'] = df['TenureGroup'].cat.add_categories([0]).fillna(0).astype(int)
df['IncomePerYear'] = df['MonthlyIncome'] * 12

satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
for col in satisfaction_cols:
    if col not in df.columns:
        df[col] = 3
df['SatisfactionScore'] = df[satisfaction_cols].mean(axis=1)

if 'OverTime' in df.columns and 'JobSatisfaction' in df.columns:
    df['OverTime_Impact'] = (df['OverTime'] == 1).astype(int) * df['JobSatisfaction']
else:
    df['OverTime_Impact'] = 0

df['PromotionDelay'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
df['PromotionDelay'] = df['PromotionDelay'].fillna(0)
print("  ✅ Derived features added")

feature_cols = [c for c in df.columns if c != 'Attrition']
X = df[feature_cols]
y = df['Attrition']

print(f"\n🔍 Total features: {len(feature_cols)}")

# ─── Feature Selection using a shallow tree ───────────────────────────
print("\n🔍 Selecting top features...")
temp_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
temp_tree.fit(X, y)

selector = SelectFromModel(temp_tree, threshold='median', max_features=15)
X_selected = selector.fit_transform(X, y)
selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
print(f"✅ Selected {len(selected_features)} features")

# ─── Train/Test Split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X[selected_features], y, test_size=0.2, random_state=42, stratify=y
)
print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

# ─── Train Base Decision Tree (pruned, but raw probabilities will be extreme) ──
print("\n🌲 Training base Decision Tree...")
base_tree = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=6,
    class_weight='balanced',
    random_state=42
)
base_tree.fit(X_train, y_train)

# ─── Calibrate probabilities using Platt Scaling (sigmoid) ────────────
print("\n🎯 Calibrating probabilities for realistic outputs...")
calibrated_model = CalibratedClassifierCV(base_tree, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)

# ─── Evaluation ───────────────────────────────────────────────────────
y_pred = calibrated_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2%}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stay', 'Leave']))

# Check probability spread on test set
test_probs = calibrated_model.predict_proba(X_test)[:, 1]
print(f"\n📈 Probability statistics (Leave):")
print(f"   Min: {test_probs.min():.3f}, Max: {test_probs.max():.3f}, Mean: {test_probs.mean():.3f}")
print(f"   Std: {test_probs.std():.3f}")   # Should be > 0.15, not near 0

# Feature importance (from base tree, calibrated model doesn't have feature_importances_)
final_importances = pd.Series(base_tree.feature_importances_, index=selected_features)

# ─── Save artifacts (compatible with existing views.py) ───────────────
save_dir = SCRIPT_DIR
print(f"\n💾 Saving model artifacts to: {save_dir}")

# Save the calibrated model (still has predict_proba)
with open(os.path.join(save_dir, 'model.pkl'), 'wb') as f:
    pickle.dump(calibrated_model, f)
print("  ✅ model.pkl saved (Calibrated Decision Tree)")

with open(os.path.join(save_dir, 'encoders.pkl'), 'wb') as f:
    pickle.dump(label_encoders, f)
print("  ✅ encoders.pkl saved")

with open(os.path.join(save_dir, 'feature_cols.pkl'), 'wb') as f:
    pickle.dump(selected_features, f)
print(f"  ✅ feature_cols.pkl saved ({len(selected_features)} features)")

# Feature importance (used by dashboard)
feature_importance_dict = dict(zip(selected_features, base_tree.feature_importances_))
with open(os.path.join(save_dir, 'feature_importance.pkl'), 'wb') as f:
    pickle.dump(feature_importance_dict, f)
print("  ✅ feature_importance.pkl saved")

# Original columns (for reference)
original_cols = [c for c in df.columns if c != 'Attrition']
with open(os.path.join(save_dir, 'original_cols.pkl'), 'wb') as f:
    pickle.dump(original_cols, f)

# Summary text
with open(os.path.join(save_dir, 'model_summary.txt'), 'w') as f:
    f.write("Employee Attrition Predictor - Calibrated Decision Tree\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Training Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset Shape: {df.shape}\n")
    f.write(f"Attrition Distribution: {df['Attrition'].value_counts().to_dict()}\n")
    f.write(f"Model Type: DecisionTreeClassifier + CalibratedClassifierCV (sigmoid)\n")
    f.write(f"Number of Features: {len(selected_features)}\n")
    f.write(f"Accuracy: {accuracy:.2%}\n")
    f.write(f"Probability range on test set: {test_probs.min():.2f}% - {test_probs.max():.2f}%\n\n")
    f.write("Top 10 Features:\n")
    for name, imp in final_importances.sort_values(ascending=False).head(10).items():
        f.write(f"  {name}: {imp:.4f}\n")

print("  ✅ model_summary.txt saved")
print("\n🎉 Calibrated Decision Tree training complete!")
print("📊 Probabilities will now be realistic and varied.")