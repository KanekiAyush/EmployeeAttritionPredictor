import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# ─── Generate synthetic IBM-style HR dataset ───────────────────────────────
np.random.seed(42)
n = 1470

departments   = ['Sales', 'Research & Development', 'Human Resources']
job_roles     = ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                 'Manufacturing Director', 'Healthcare Representative', 'Manager',
                 'Sales Representative', 'Research Director', 'Human Resources']
edu_fields    = ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources']
marital       = ['Single', 'Married', 'Divorced']
overtime_vals = ['Yes', 'No']
genders       = ['Male', 'Female']

data = {
    'Age':                        np.random.randint(18, 60, n),
    'BusinessTravel':             np.random.choice(['Travel_Rarely','Travel_Frequently','Non-Travel'], n),
    'DailyRate':                  np.random.randint(100, 1500, n),
    'Department':                 np.random.choice(departments, n),
    'DistanceFromHome':           np.random.randint(1, 30, n),
    'Education':                  np.random.randint(1, 5, n),
    'EducationField':             np.random.choice(edu_fields, n),
    'EnvironmentSatisfaction':    np.random.randint(1, 5, n),
    'Gender':                     np.random.choice(genders, n),
    'HourlyRate':                 np.random.randint(30, 100, n),
    'JobInvolvement':             np.random.randint(1, 5, n),
    'JobLevel':                   np.random.randint(1, 6, n),
    'JobRole':                    np.random.choice(job_roles, n),
    'JobSatisfaction':            np.random.randint(1, 5, n),
    'MaritalStatus':              np.random.choice(marital, n),
    'MonthlyIncome':              np.random.randint(1000, 20000, n),
    'MonthlyRate':                np.random.randint(2000, 27000, n),
    'NumCompaniesWorked':         np.random.randint(0, 10, n),
    'OverTime':                   np.random.choice(overtime_vals, n),
    'PercentSalaryHike':          np.random.randint(11, 25, n),
    'PerformanceRating':          np.random.randint(3, 5, n),
    'RelationshipSatisfaction':   np.random.randint(1, 5, n),
    'StockOptionLevel':           np.random.randint(0, 4, n),
    'TotalWorkingYears':          np.random.randint(0, 40, n),
    'TrainingTimesLastYear':      np.random.randint(0, 7, n),
    'WorkLifeBalance':            np.random.randint(1, 5, n),
    'YearsAtCompany':             np.random.randint(0, 40, n),
    'YearsInCurrentRole':         np.random.randint(0, 18, n),
    'YearsSinceLastPromotion':    np.random.randint(0, 15, n),
    'YearsWithCurrManager':       np.random.randint(0, 17, n),
}

df = pd.DataFrame(data)

# Realistic attrition probability based on key factors
attrition_prob = (
    0.05
    + 0.25 * (df['OverTime'] == 'Yes')
    + 0.10 * (df['JobSatisfaction'] <= 2)
    + 0.10 * (df['WorkLifeBalance'] <= 2)
    + 0.08 * (df['MonthlyIncome'] < 3000)
    + 0.08 * (df['YearsAtCompany'] < 2)
    + 0.06 * (df['DistanceFromHome'] > 20)
    + 0.05 * (df['JobInvolvement'] <= 2)
    - 0.05 * (df['StockOptionLevel'] >= 2)
    - 0.04 * (df['JobLevel'] >= 3)
)
attrition_prob = attrition_prob.clip(0.02, 0.95)
df['Attrition'] = np.where(np.random.rand(n) < attrition_prob, 'Yes', 'No')

# ─── Preprocessing ──────────────────────────────────────────────────────────
label_encoders = {}
categorical_cols = ['BusinessTravel', 'Department', 'EducationField',
                    'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

target_le = LabelEncoder()
df['Attrition'] = target_le.fit_transform(df['Attrition'])  # No=0, Yes=1
label_encoders['Attrition'] = target_le

feature_cols = [c for c in df.columns if c != 'Attrition']
X = df[feature_cols]
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─── Train Random Forest ────────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=200, max_depth=10,
                               random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Stay', 'Leave']))

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols)
top10 = importances.nlargest(10)
print("\nTop 10 Important Features:")
print(top10)

# ─── Save model & encoders ──────────────────────────────────────────────────
save_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(save_dir, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)
with open(os.path.join(save_dir, 'encoders.pkl'), 'wb') as f:
    pickle.dump(label_encoders, f)
with open(os.path.join(save_dir, 'feature_cols.pkl'), 'wb') as f:
    pickle.dump(feature_cols, f)

print("\n✅ model.pkl, encoders.pkl, feature_cols.pkl saved!")
