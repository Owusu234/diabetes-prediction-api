from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import json
import os
import logging
from dotenv import load_dotenv
from functools import wraps
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

#  API KEY AUTHENTICATION DECORATOR

def require_api_key(f):
    """
    Decorator to require API key for protected routes
    Usage: @app.route('/predict', methods=['POST'])
            @require_api_key
            def predict():
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get API key from headers
        api_key = request.headers.get('X-API-Key')
        
        # Get expected API key from environment
        valid_api_key = os.getenv('API_KEY')
        
        # Check if API key is provided
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'Missing API key',
                'message': 'Please provide X-API-Key header'
            }), 401
        
        # Check if API key is valid
        if not valid_api_key or api_key != valid_api_key:
            logger.warning(f"Invalid API key attempt: {api_key[:10]}...")
            return jsonify({
                'success': False,
                'error': 'Invalid API key',
                'message': 'Access denied'
            }), 403
        
        # API key is valid, proceed with the request
        logger.info(f" API key authenticated successfully")
        return f(*args, **kwargs)
    
    return decorated_function


# Model Loading (same as before)

def download_model_if_missing():
    """Download model from cloud storage if not present"""
    model_path = os.getenv('MODEL_PATH', 'models/diabetes_model.pkl')
    download_url = os.getenv('MODEL_DOWNLOAD_URL')
    
    os.makedirs('models', exist_ok=True)
    
    if not os.path.exists(model_path) and download_url:
        logger.info(f" Model not found. Downloading from {download_url}...")
        try:
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f" Model downloaded and saved to {model_path}")
        except Exception as e:
            logger.error(f" Failed to download model: {e}")
            return False
    
    return os.path.exists(model_path)

# Download model on startup
model_available = download_model_if_missing()

# Load model and feature names
model_path = os.getenv('MODEL_PATH', 'models/diabetes_model.pkl')
features_path = 'models/feature_names.json'

model = None
feature_names = []

if model_available:
    try:
        model = joblib.load(model_path)
        logger.info(f" Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f" Failed to load model: {e}")

try:
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
        logger.info(f" Features loaded: {feature_names}")
except Exception as e:
    logger.error(f" Failed to load features: {e}")

# API ENDPOINTS


@app.route('/')
def home():
    """Public endpoint - no API key required"""
    return jsonify({
        'message': 'Diabetes Prediction API',
        'version': '1.0',
        'status': 'running',
        'environment': os.getenv('FLASK_ENV', 'production'),
        'authentication': 'API key required for /predict endpoint',
        'documentation': '/docs'
    })

@app.route('/health')
def health():
    """Public health check - no API key required"""
    return jsonify({
        'status': 'healthy' if model else 'unhealthy',
        'model_loaded': model is not None,
        'features_loaded': len(feature_names) > 0,
        'environment': os.getenv('FLASK_ENV', 'production'),
        'timestamp': str(datetime.now()) if 'datetime' in dir() else 'N/A'
    })

@app.route('/docs')
def docs():
    """Public API documentation"""
    return jsonify({
        'api_name': 'Diabetes Prediction API',
        'version': '1.0',
        'base_url': request.host_url,
        'authentication': {
            'type': 'API Key',
            'header': 'X-API-Key',
            'location': 'Request header'
        },
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'GET /docs': 'API documentation',
            'POST /predict': 'Diabetes prediction (requires API key)'
        },
        'example_request': {
            'endpoint': 'POST /predict',
            'headers': {
                'Content-Type': 'application/json',
                'X-API-Key': 'your-api-key-here'
            },
            'body': {
                'Pregnancies': 6,
                'Glucose': 148,
                'BloodPressure': 72,
                'SkinThickness': 35,
                'Insulin': 0,
                'BMI': 33.6,
                'DiabetesPedigreeFunction': 0.627,
                'Age': 50
            }
        }
    })

@app.route('/predict', methods=['POST'])
@require_api_key  # This endpoint requires API key
def predict():
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded',
            'message': 'Prediction service unavailable'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 
                          'SkinThickness', 'Insulin', 'BMI', 
                          'DiabetesPedigreeFunction', 'Age']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Create base features
        base_features = [
            float(data['Pregnancies']),
            float(data['Glucose']),
            float(data['BloodPressure']),
            float(data['SkinThickness']),
            float(data['Insulin']),
            float(data['BMI']),
            float(data['DiabetesPedigreeFunction']),
            float(data['Age'])
        ]
        
        # Add engineered features
        glucose_age = float(data['Glucose']) * float(data['Age']) / 100
        bmi_insulin = float(data['BMI']) * float(data['Insulin']) / 100
        
        # Combine all features
        input_data = np.array(base_features + [glucose_age, bmi_insulin]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Prepare response
        result = {
            'success': True,
            'prediction': int(prediction),
            'prediction_label': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'diabetes_probability': round(float(probability[1]), 4),
            'no_diabetes_probability': round(float(probability[0]), 4),
            'risk_level': get_risk_level(probability[1]),
            'message': get_prediction_message(prediction, probability[1])
        }
        
        return jsonify(result)
    
    except ValueError as ve:
        return jsonify({
            'success': False,
            'error': f'Invalid input: {str(ve)}',
            'message': 'Please check your input values'
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Prediction failed'
        }), 500

def get_risk_level(probability):
    if probability < 0.3:
        return 'Low Risk'
    elif probability < 0.6:
        return 'Moderate Risk'
    else:
        return 'High Risk'

def get_prediction_message(prediction, probability):
    confidence = probability if prediction == 1 else (1 - probability)
    if prediction == 0:
        return f'You are not diabetic (Confidence: {confidence*100:.1f}%)'
    else:
        return f'You are diabetic (Confidence: {confidence*100:.1f}%)'

if __name__ == '__main__':
    # Only used for development/testing
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
