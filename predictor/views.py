import os, pickle, json
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse

# ─── Load model artifacts ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(BASE_DIR, 'ml_model')

with open(os.path.join(ML_DIR, 'model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)
with open(os.path.join(ML_DIR, 'encoders.pkl'), 'rb') as f:
    ENCODERS = pickle.load(f)
with open(os.path.join(ML_DIR, 'feature_cols.pkl'), 'rb') as f:
    FEATURE_COLS = pickle.load(f)
with open(os.path.join(ML_DIR, 'feature_importance.pkl'), 'rb') as f:
    FEATURE_IMPORTANCE = pickle.load(f)

# Categorical columns for encoding
CATEGORICAL = ['BusinessTravel', 'Department', 'EducationField',
               'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

# Risk factor mapping for user-friendly messages
RISK_FACTORS_MAP = {
    'OverTime': ('Working Overtime', lambda x: x == 'Yes'),
    'JobSatisfaction': ('Low Job Satisfaction', lambda x: int(x) <= 2),
    'WorkLifeBalance': ('Poor Work-Life Balance', lambda x: int(x) <= 2),
    'MonthlyIncome': ('Below-average Income', lambda x: int(x) < 4000),
    'YearsAtCompany': ('Short Tenure (< 2 years)', lambda x: int(x) < 2),
    'DistanceFromHome': ('Long Commute (> 15 km)', lambda x: int(x) > 15),
    'JobInvolvement': ('Low Job Involvement', lambda x: int(x) <= 2),
    'EnvironmentSatisfaction': ('Poor Work Environment', lambda x: int(x) <= 2),
    'StockOptionLevel': ('No Stock Options', lambda x: int(x) == 0),
    'YearsSinceLastPromotion': ('No Recent Promotion (> 3 years)', lambda x: int(x) > 3),
    'NumCompaniesWorked': ('Frequent Job Changes', lambda x: int(x) > 5),
    'TrainingTimesLastYear': ('Low Training Opportunities', lambda x: int(x) < 2),
}

# Recommendation engine based on risk factors
RECOMMENDATIONS = {
    'Working Overtime': 'Consider reviewing workload distribution and hiring additional staff.',
    'Low Job Satisfaction': 'Schedule a one-on-one meeting to discuss career growth and concerns.',
    'Poor Work-Life Balance': 'Offer flexible working hours or remote work options.',
    'Below-average Income': 'Review compensation package and consider market adjustment.',
    'Short Tenure (< 2 years)': 'Assign a mentor and create a 90-day development plan.',
    'Long Commute (> 15 km)': 'Explore remote work options or flexible hours.',
    'Low Job Involvement': 'Involve employee in more engaging projects and team activities.',
    'Poor Work Environment': 'Conduct workplace assessment and address reported issues.',
    'No Stock Options': 'Consider including in next equity grant cycle.',
    'No Recent Promotion (> 3 years)': 'Review career progression path and create growth plan.',
    'Frequent Job Changes': 'Conduct stay interviews to understand retention factors.',
    'Low Training Opportunities': 'Allocate budget for professional development courses.',
}


def dashboard(request):
    """Dashboard view with analytics"""
    # Calculate some real metrics from the model if possible
    dept_attrition = {'Sales': 21, 'Research & Development': 14, 'Human Resources': 19}
    age_attrition = {'18-25': 35, '26-30': 28, '31-35': 18, '36-40': 12, '41-50': 10, '51+': 8}
    satisfaction_attrition = {'Low (1)': 42, 'Medium (2)': 22, 'High (3)': 13, 'Very High (4)': 8}
    overtime_attrition = {'With Overtime': 31, 'Without Overtime': 10}

    # Get top features for display
    top_features = sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1], reverse=True)[:10]

    context = {
        'dept_attrition': json.dumps(dept_attrition),
        'age_attrition': json.dumps(age_attrition),
        'satisfaction_attrition': json.dumps(satisfaction_attrition),
        'overtime_attrition': json.dumps(overtime_attrition),
        'top_features': top_features,
        'total_employees': 1470,
        'attrition_rate': 16.1,
        'avg_tenure': 7.0,
        'high_risk_count': 237,
        'model_features': FEATURE_COLS,
        'feature_count': len(FEATURE_COLS),
    }
    return render(request, 'predictor/dashboard.html', context)


def predict(request):
    """Prediction form view"""
    context = {
        'departments': ['Sales', 'Research & Development', 'Human Resources'],
        'job_roles': ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                      'Manufacturing Director', 'Healthcare Representative', 'Manager',
                      'Sales Representative', 'Research Director', 'Human Resources'],
        'edu_fields': ['Life Sciences', 'Other', 'Medical', 'Marketing',
                       'Technical Degree', 'Human Resources'],
        'marital': ['Single', 'Married', 'Divorced'],
        'travel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
        'genders': ['Male', 'Female'],
        'feature_count': len(FEATURE_COLS),
        'top_features': sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1], reverse=True)[:5],
    }
    return render(request, 'predictor/predict.html', context)


def predict_api(request):
    """API endpoint for predictions"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        # Collect and validate input data
        raw = {}
        for col in FEATURE_COLS:
            if col in CATEGORICAL:
                raw[col] = request.POST.get(col, '')
            else:
                try:
                    raw[col] = int(request.POST.get(col, 0))
                except (ValueError, TypeError):
                    raw[col] = 0

        # Encode categoricals
        for col in CATEGORICAL:
            if col in raw and raw[col]:
                le = ENCODERS[col]
                try:
                    raw[col] = int(le.transform([raw[col]])[0])
                except ValueError:
                    # Default to most common class if value not found
                    raw[col] = 0

        # Create feature vector
        X = np.array([[raw[col] for col in FEATURE_COLS]])

        # Get prediction probabilities
        prob = MODEL.predict_proba(X)[0]

        # Get class labels
        classes = list(ENCODERS['Attrition'].classes_)  # ['No', 'Yes']
        leave_idx = classes.index('Yes')
        stay_idx = classes.index('No')

        leave_prob = float(prob[leave_idx]) * 100
        stay_prob = float(prob[stay_idx]) * 100

        # Determine risk level
        if leave_prob >= 70:
            risk = 'HIGH'
        elif leave_prob >= 40:
            risk = 'MEDIUM'
        else:
            risk = 'LOW'

        # Identify risk factors and recommendations
        risk_factors = []
        recommendations = []

        for factor, (msg, condition) in RISK_FACTORS_MAP.items():
            if factor in raw:
                value = raw[factor]
                if factor in CATEGORICAL:
                    # Decode categorical value for condition
                    decoded_value = ENCODERS[factor].inverse_transform([value])[0]
                    if condition(decoded_value):
                        risk_factors.append(msg)
                        if msg in RECOMMENDATIONS:
                            recommendations.append(RECOMMENDATIONS[msg])
                else:
                    if condition(value):
                        risk_factors.append(msg)
                        if msg in RECOMMENDATIONS:
                            recommendations.append(RECOMMENDATIONS[msg])

        # Remove duplicates
        recommendations = list(set(recommendations))

        # Prepare response
        response = {
            'success': True,
            'prediction': 'Yes' if leave_prob >= 50 else 'No',
            'leave_probability': round(leave_prob, 1),
            'stay_probability': round(stay_prob, 1),
            'risk_level': risk,
            'risk_factors': risk_factors[:5],  # Limit to top 5
            'recommendations': recommendations[:3],  # Limit to top 3
            'model_confidence': round(max(leave_prob, stay_prob), 1),
            'feature_used': len(FEATURE_COLS),
        }

        return JsonResponse(response)

    except Exception as e:
        return JsonResponse({'error': str(e), 'success': False}, status=400)


def batch_predict(request):
    """Batch prediction endpoint (for future enhancement)"""
    return JsonResponse({'message': 'Batch prediction coming soon!'})