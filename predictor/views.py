import os
import pickle
import json
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

# All categorical columns (excluding Attrition which is the target)
CATEGORICAL = ['BusinessTravel', 'Department', 'EducationField',
               'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

# All numeric columns (auto-detected from feature columns)
NUMERIC = [col for col in FEATURE_COLS if col not in CATEGORICAL]

# Risk factor mapping for user-friendly messages
RISK_FACTORS_MAP = {
    'OverTime': ('Working Overtime', lambda x: x == 'Yes'),
    'JobSatisfaction': ('Low Job Satisfaction', lambda x: int(x) <= 2),
    'WorkLifeBalance': ('Poor Work-Life Balance', lambda x: int(x) <= 2),
    'MonthlyIncome': ('Below-average Income', lambda x: int(x) < 5000),
    'YearsAtCompany': ('Short Tenure (< 2 years)', lambda x: int(x) < 2),
    'DistanceFromHome': ('Long Commute (> 15 km)', lambda x: int(x) > 15),
    'JobInvolvement': ('Low Job Involvement', lambda x: int(x) <= 2),
    'EnvironmentSatisfaction': ('Poor Work Environment', lambda x: int(x) <= 2),
    'StockOptionLevel': ('No Stock Options', lambda x: int(x) == 0),
    'YearsSinceLastPromotion': ('No Recent Promotion (> 3 years)', lambda x: int(x) > 3),
    'NumCompaniesWorked': ('Frequent Job Changes', lambda x: int(x) > 5),
    'TrainingTimesLastYear': ('Low Training Opportunities', lambda x: int(x) < 2),
    'PerformanceRating': ('Low Performance Rating', lambda x: int(x) <= 2),
    'RelationshipSatisfaction': ('Poor Work Relationships', lambda x: int(x) <= 2),
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
    'Low Performance Rating': 'Provide additional training and performance improvement support.',
    'Poor Work Relationships': 'Facilitate team-building activities and conflict resolution.',
}


def dashboard(request):
    """Dashboard view with analytics"""
    # Note: These are sample metrics. In production, calculate from actual data
    dept_attrition = {'Sales': 21, 'Research & Development': 14, 'Human Resources': 19}
    age_attrition = {'18-25': 35, '26-30': 28, '31-35': 18, '36-40': 12, '41-50': 10, '51+': 8}
    satisfaction_attrition = {'Low (1)': 42, 'Medium (2)': 22, 'High (3)': 13, 'Very High (4)': 8}
    overtime_attrition = {'With Overtime': 31, 'Without Overtime': 10}

    # Get feature importance (if available, otherwise calculate)
    feature_importance = {}
    if hasattr(MODEL, 'feature_importances_'):
        for col, imp in zip(FEATURE_COLS, MODEL.feature_importances_):
            feature_importance[col] = imp
    else:
        # Fallback if no feature importance
        feature_importance = {col: 0 for col in FEATURE_COLS[:10]}

    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

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
        'numeric_fields': NUMERIC,
        'categorical_fields': CATEGORICAL,
    }
    return render(request, 'predictor/predict.html', context)


def predict_api(request):
    """API endpoint for predictions"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        # Collect input data
        raw = {}
        
        # Get numeric values
        for col in NUMERIC:
            try:
                raw[col] = int(request.POST.get(col, 0))
            except (ValueError, TypeError):
                raw[col] = 0
        
        # Get categorical values and encode them
        for col in CATEGORICAL:
            value = request.POST.get(col, '')
            if value and col in ENCODERS:
                le = ENCODERS[col]
                try:
                    raw[col] = int(le.transform([value])[0])
                except ValueError:
                    # Use most common value (index 0) if not found
                    raw[col] = 0
            else:
                raw[col] = 0

        # Create feature vector in the correct order
        X = np.array([[raw[col] for col in FEATURE_COLS]])

        # Get prediction probabilities
        prob = MODEL.predict_proba(X)[0]

        # Determine which index is "Yes" (Leave)
        # Attrition encoder should have classes ['No', 'Yes']
        if 'Attrition' in ENCODERS:
            classes = list(ENCODERS['Attrition'].classes_)
            if 'Yes' in classes:
                leave_idx = classes.index('Yes')
            else:
                leave_idx = 1  # Assume index 1 is Yes
        else:
            # Fallback: assume binary classification with Yes=1
            leave_idx = 1
        
        leave_prob = float(prob[leave_idx]) * 100
        stay_prob = float(100 - leave_prob)

        # Determine risk level
        if leave_prob >= 70:
            risk = 'HIGH'
        elif leave_prob >= 40:
            risk = 'MEDIUM'
        else:
            risk = 'LOW'

        # Identify risk factors
        risk_factors = []
        recommendations = []

        # Get decoded values for categorical risk factors
        decoded_values = {}
        for col in CATEGORICAL:
            if col in ENCODERS and raw[col] is not None:
                try:
                    decoded_values[col] = ENCODERS[col].inverse_transform([raw[col]])[0]
                except:
                    decoded_values[col] = str(raw[col])
        
        # Check each risk factor
        for factor, (msg, condition) in RISK_FACTORS_MAP.items():
            if factor in raw:
                value = raw[factor]
                
                # Handle categorical vs numeric
                if factor in CATEGORICAL:
                    # Use decoded value for condition
                    decoded_val = decoded_values.get(factor, value)
                    try:
                        if condition(decoded_val):
                            risk_factors.append(msg)
                            if msg in RECOMMENDATIONS:
                                recommendations.append(RECOMMENDATIONS[msg])
                    except:
                        pass
                else:
                    # Numeric factor
                    try:
                        if condition(value):
                            risk_factors.append(msg)
                            if msg in RECOMMENDATIONS:
                                recommendations.append(RECOMMENDATIONS[msg])
                    except:
                        pass

        # Remove duplicates and limit
        risk_factors = list(dict.fromkeys(risk_factors))[:5]
        recommendations = list(dict.fromkeys(recommendations))[:3]

        # Prepare response
        response = {
            'success': True,
            'prediction': 'Yes' if leave_prob >= 50 else 'No',
            'leave_probability': round(leave_prob, 1),
            'stay_probability': round(stay_prob, 1),
            'risk_level': risk,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'model_confidence': round(max(leave_prob, stay_prob), 1),
            'features_used': len(FEATURE_COLS),
        }

        return JsonResponse(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e), 'success': False}, status=400)


def batch_predict(request):
    """Batch prediction endpoint (for future enhancement)"""
    return JsonResponse({'message': 'Batch prediction coming soon!'})