import os, pickle, json
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse

# ─── Load model artifacts — path relative to manage.py (BASE_DIR) ────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR   = os.path.join(BASE_DIR, 'ml_model')

with open(os.path.join(ML_DIR, 'model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)
with open(os.path.join(ML_DIR, 'encoders.pkl'), 'rb') as f:
    ENCODERS = pickle.load(f)
with open(os.path.join(ML_DIR, 'feature_cols.pkl'), 'rb') as f:
    FEATURE_COLS = pickle.load(f)

CATEGORICAL = ['BusinessTravel', 'Department', 'EducationField',
               'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

# Top 10 feature importances
importances = dict(zip(FEATURE_COLS, MODEL.feature_importances_))
TOP_FEATURES = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]


def dashboard(request):
    dept_attrition         = {'Sales': 21, 'Research & Development': 14, 'Human Resources': 19}
    age_attrition          = {'18-25': 35, '26-30': 28, '31-35': 18, '36-40': 12, '41-50': 10, '51+': 8}
    satisfaction_attrition = {'Low (1)': 42, 'Medium (2)': 22, 'High (3)': 13, 'Very High (4)': 8}
    overtime_attrition     = {'With Overtime': 31, 'Without Overtime': 10}

    context = {
        'dept_attrition':         json.dumps(dept_attrition),
        'age_attrition':          json.dumps(age_attrition),
        'satisfaction_attrition': json.dumps(satisfaction_attrition),
        'overtime_attrition':     json.dumps(overtime_attrition),
        'top_features':           TOP_FEATURES,
        'total_employees':        1470,
        'attrition_rate':         16.1,
        'avg_tenure':             7.0,
        'high_risk_count':        237,
    }
    return render(request, 'predictor/dashboard.html', context)


def predict(request):
    context = {
        'departments': ['Sales', 'Research & Development', 'Human Resources'],
        'job_roles':   ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                        'Manufacturing Director', 'Healthcare Representative', 'Manager',
                        'Sales Representative', 'Research Director', 'Human Resources'],
        'edu_fields':  ['Life Sciences', 'Other', 'Medical', 'Marketing',
                        'Technical Degree', 'Human Resources'],
        'marital':     ['Single', 'Married', 'Divorced'],
        'travel':      ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
        'genders':     ['Male', 'Female'],
    }
    return render(request, 'predictor/predict.html', context)


def predict_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        raw = {
            'Age':                      int(request.POST.get('Age', 30)),
            'BusinessTravel':           request.POST.get('BusinessTravel', 'Travel_Rarely'),
            'DailyRate':                int(request.POST.get('DailyRate', 800)),
            'Department':               request.POST.get('Department', 'Sales'),
            'DistanceFromHome':         int(request.POST.get('DistanceFromHome', 5)),
            'Education':                int(request.POST.get('Education', 3)),
            'EducationField':           request.POST.get('EducationField', 'Life Sciences'),
            'EnvironmentSatisfaction':  int(request.POST.get('EnvironmentSatisfaction', 3)),
            'Gender':                   request.POST.get('Gender', 'Male'),
            'HourlyRate':               int(request.POST.get('HourlyRate', 65)),
            'JobInvolvement':           int(request.POST.get('JobInvolvement', 3)),
            'JobLevel':                 int(request.POST.get('JobLevel', 2)),
            'JobRole':                  request.POST.get('JobRole', 'Sales Executive'),
            'JobSatisfaction':          int(request.POST.get('JobSatisfaction', 3)),
            'MaritalStatus':            request.POST.get('MaritalStatus', 'Single'),
            'MonthlyIncome':            int(request.POST.get('MonthlyIncome', 5000)),
            'MonthlyRate':              int(request.POST.get('MonthlyRate', 14000)),
            'NumCompaniesWorked':       int(request.POST.get('NumCompaniesWorked', 2)),
            'OverTime':                 request.POST.get('OverTime', 'No'),
            'PercentSalaryHike':        int(request.POST.get('PercentSalaryHike', 14)),
            'PerformanceRating':        int(request.POST.get('PerformanceRating', 3)),
            'RelationshipSatisfaction': int(request.POST.get('RelationshipSatisfaction', 3)),
            'StockOptionLevel':         int(request.POST.get('StockOptionLevel', 1)),
            'TotalWorkingYears':        int(request.POST.get('TotalWorkingYears', 8)),
            'TrainingTimesLastYear':    int(request.POST.get('TrainingTimesLastYear', 3)),
            'WorkLifeBalance':          int(request.POST.get('WorkLifeBalance', 3)),
            'YearsAtCompany':           int(request.POST.get('YearsAtCompany', 5)),
            'YearsInCurrentRole':       int(request.POST.get('YearsInCurrentRole', 3)),
            'YearsSinceLastPromotion':  int(request.POST.get('YearsSinceLastPromotion', 1)),
            'YearsWithCurrManager':     int(request.POST.get('YearsWithCurrManager', 3)),
        }

        # Encode categoricals
        for col in CATEGORICAL:
            le  = ENCODERS[col]
            val = raw[col]
            raw[col] = int(le.transform([val])[0]) if val in le.classes_ else 0

        # Feature vector in correct order
        X    = np.array([[raw[col] for col in FEATURE_COLS]])
        prob = MODEL.predict_proba(X)[0]

        classes   = list(ENCODERS['Attrition'].classes_)   # ['No', 'Yes']
        leave_idx = classes.index('Yes')
        stay_idx  = classes.index('No')

        leave_prob = float(prob[leave_idx]) * 100
        stay_prob  = float(prob[stay_idx])  * 100

        risk = 'HIGH' if leave_prob >= 70 else ('MEDIUM' if leave_prob >= 40 else 'LOW')

        # Key risk factors
        risk_factors = []
        if request.POST.get('OverTime') == 'Yes':
            risk_factors.append('Working Overtime')
        if int(request.POST.get('JobSatisfaction', 3)) <= 2:
            risk_factors.append('Low Job Satisfaction')
        if int(request.POST.get('WorkLifeBalance', 3)) <= 2:
            risk_factors.append('Poor Work-Life Balance')
        if int(request.POST.get('MonthlyIncome', 5000)) < 3000:
            risk_factors.append('Below-average Income')
        if int(request.POST.get('YearsAtCompany', 5)) <= 2:
            risk_factors.append('Short Tenure')
        if int(request.POST.get('DistanceFromHome', 5)) > 20:
            risk_factors.append('Long Commute Distance')
        if int(request.POST.get('JobInvolvement', 3)) <= 2:
            risk_factors.append('Low Job Involvement')
        if int(request.POST.get('EnvironmentSatisfaction', 3)) <= 2:
            risk_factors.append('Poor Environment Satisfaction')

        return JsonResponse({
            'prediction':        'Yes' if leave_prob >= 50 else 'No',
            'leave_probability': round(leave_prob, 1),
            'stay_probability':  round(stay_prob, 1),
            'risk_level':        risk,
            'risk_factors':      risk_factors,
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
