# 🎯 Module 5: Deploying Machine Learning Models

> **Transform your ML models into production-ready applications**

This module covers the complete journey from trained models to production deployment. You'll learn modern deployment strategies, containerization, cloud platforms, and monitoring techniques.

## 📚 Learning Objectives

By the end of this module, you will:
- **Deploy** ML models as web services using Flask
- **Containerize** applications with Docker for consistent deployment
- **Use** cloud platforms (AWS, GCP, Azure) for scalable deployment
- **Implement** model versioning and monitoring
- **Build** CI/CD pipelines for ML applications
- **Handle** production challenges like scaling and monitoring

## 🚀 Deployment Overview

### Why Model Deployment Matters
- **Business Value**: Models only create value when used in production
- **Accessibility**: Make predictions available to users and systems
- **Scalability**: Handle multiple users and high request volumes
- **Reliability**: Ensure consistent performance and uptime
- **Monitoring**: Track model performance and detect drift

## 🗂️ Module Contents

### **5.1 Introduction to Model Deployment**
**Key Concepts:**
- Deployment strategies and architectures
- Batch vs real-time inference
- Model serving patterns

**Deployment Patterns:**
```python
# Batch Prediction Pattern
def batch_prediction_pipeline():
    """Process large datasets in batches"""
    
    # Load model
    model = joblib.load('model.pkl')
    
    # Process data in chunks
    chunk_size = 1000
    predictions = []
    
    for chunk in pd.read_csv('data.csv', chunksize=chunk_size):
        # Preprocess chunk
        chunk_processed = preprocess_data(chunk)
        
        # Make predictions
        chunk_predictions = model.predict(chunk_processed)
        predictions.extend(chunk_predictions)
    
    return predictions

# Real-time Prediction Pattern
def real_time_prediction_api():
    """Serve predictions via API"""
    
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    model = joblib.load('model.pkl')
    
    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        features = preprocess_features(data)
        prediction = model.predict([features])[0]
        
        return jsonify({
            'prediction': prediction,
            'probability': model.predict_proba([features])[0].tolist()
        })
    
    return app
```

### **5.2 Model Serialization with Pickle**
**Advanced Model Persistence:**
```python
import pickle
import joblib
import json
from datetime import datetime

class ModelManager:
    def __init__(self, model_dir='models/'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model(self, model, model_name, metadata=None):
        """Save model with metadata and versioning"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version = f"{model_name}_v{timestamp}"
        
        # Save model
        model_path = os.path.join(self.model_dir, f"{version}.pkl")
        joblib.dump(model, model_path)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'model_name': model_name,
            'version': version,
            'timestamp': timestamp,
            'model_path': model_path,
            'model_type': type(model).__name__
        })
        
        metadata_path = os.path.join(self.model_dir, f"{version}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved: {model_path}")
        print(f"Metadata saved: {metadata_path}")
        
        return version
    
    def load_model(self, version):
        """Load model by version"""
        model_path = os.path.join(self.model_dir, f"{version}.pkl")
        metadata_path = os.path.join(self.model_dir, f"{version}_metadata.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model, metadata
    
    def list_models(self):
        """List all available model versions"""
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
        versions = [f.replace('.pkl', '') for f in model_files]
        return sorted(versions, reverse=True)
```

### **5.3 Flask Web Service Creation**
**Production-Ready Flask Application:**
```python
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import traceback

class MLFlaskApp:
    def __init__(self, model, preprocessor=None):
        self.app = Flask(__name__)
        self.model = model
        self.preprocessor = preprocessor
        self.setup_logging()
        self.setup_routes()
        self.prediction_count = 0
    
    def setup_logging(self):
        """Setup logging for the application"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_routes(self):
        """Setup all application routes"""
        
        @self.app.route('/')
        def home():
            return render_template('index.html')
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'predictions_served': self.prediction_count
            })
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Main prediction endpoint"""
            try:
                # Get request data
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
                # Preprocess data
                if self.preprocessor:
                    features = self.preprocessor.transform([data])
                else:
                    features = np.array(list(data.values())).reshape(1, -1)
                
                # Make prediction
                prediction = self.model.predict(features)[0]
                
                # Get probability if available
                probability = None
                if hasattr(self.model, 'predict_proba'):
                    probability = self.model.predict_proba(features)[0].tolist()
                
                # Log prediction
                self.logger.info(f"Prediction made: {prediction}")
                self.prediction_count += 1
                
                # Return response
                response = {
                    'prediction': float(prediction),
                    'timestamp': datetime.now().isoformat(),
                    'request_id': self.prediction_count
                }
                
                if probability:
                    response['probability'] = probability
                
                return jsonify(response)
            
            except Exception as e:
                self.logger.error(f"Prediction error: {str(e)}")
                self.logger.error(traceback.format_exc())
                
                return jsonify({
                    'error': 'Prediction failed',
                    'message': str(e)
                }), 500
        
        @self.app.route('/batch_predict', methods=['POST'])
        def batch_predict():
            """Batch prediction endpoint"""
            try:
                data = request.get_json()
                
                if 'instances' not in data:
                    return jsonify({'error': 'No instances provided'}), 400
                
                instances = data['instances']
                predictions = []
                
                for instance in instances:
                    if self.preprocessor:
                        features = self.preprocessor.transform([instance])
                    else:
                        features = np.array(list(instance.values())).reshape(1, -1)
                    
                    prediction = self.model.predict(features)[0]
                    predictions.append(float(prediction))
                
                self.prediction_count += len(predictions)
                
                return jsonify({
                    'predictions': predictions,
                    'count': len(predictions),
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                self.logger.error(f"Batch prediction error: {str(e)}")
                return jsonify({
                    'error': 'Batch prediction failed',
                    'message': str(e)
                }), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        self.app.run(host=host, port=port, debug=debug)
```

### **5.4 Containerization with Docker**
**Complete Docker Setup:**

**Dockerfile:**
```dockerfile
# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

**requirements.txt:**
```
Flask==2.3.3
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
joblib==1.3.2
gunicorn==21.2.0
```

**Docker Compose for Development:**
```yaml
version: '3.8'

services:
  ml-app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=development
      - MODEL_PATH=/app/models/model.pkl
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - ml-app
```

### **5.5 Cloud Deployment Strategies**

**AWS Deployment with Elastic Beanstalk:**
```python
# eb_deploy.py
import boto3
import zipfile
import os

class AWSDeployer:
    def __init__(self, region='us-east-1'):
        self.eb_client = boto3.client('elasticbeanstalk', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
    
    def create_application_version(self, app_name, version_label, source_bundle):
        """Create new application version"""
        
        # Upload source bundle to S3
        bucket_name = f"{app_name}-deployments"
        key = f"versions/{version_label}.zip"
        
        self.s3_client.upload_file(source_bundle, bucket_name, key)
        
        # Create application version
        response = self.eb_client.create_application_version(
            ApplicationName=app_name,
            VersionLabel=version_label,
            SourceBundle={
                'S3Bucket': bucket_name,
                'S3Key': key
            }
        )
        
        return response
    
    def deploy_to_environment(self, app_name, env_name, version_label):
        """Deploy version to environment"""
        
        response = self.eb_client.update_environment(
            ApplicationName=app_name,
            EnvironmentName=env_name,
            VersionLabel=version_label
        )
        
        return response
```

**Google Cloud Run Deployment:**
```yaml
# cloudbuild.yaml
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/ml-app:$COMMIT_SHA', '.']
  
  # Push the container image to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/ml-app:$COMMIT_SHA']
  
  # Deploy container image to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
    - 'run'
    - 'deploy'
    - 'ml-app'
    - '--image'
    - 'gcr.io/$PROJECT_ID/ml-app:$COMMIT_SHA'
    - '--region'
    - 'us-central1'
    - '--platform'
    - 'managed'
    - '--allow-unauthenticated'
```

### **5.6 Model Monitoring and Observability**
**Production Monitoring System:**
```python
import psutil
import time
from datetime import datetime, timedelta
import json

class ModelMonitor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metrics = {
            'predictions': [],
            'response_times': [],
            'errors': [],
            'system_metrics': []
        }
    
    def log_prediction(self, input_data, prediction, response_time):
        """Log prediction for monitoring"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input_hash': hash(str(input_data)),
            'prediction': prediction,
            'response_time': response_time,
            'model_name': self.model_name
        }
        
        self.metrics['predictions'].append(log_entry)
        self.metrics['response_times'].append(response_time)
    
    def log_error(self, error_type, error_message):
        """Log errors for monitoring"""
        
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'model_name': self.model_name
        }
        
        self.metrics['errors'].append(error_entry)
    
    def collect_system_metrics(self):
        """Collect system performance metrics"""
        
        system_metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'model_name': self.model_name
        }
        
        self.metrics['system_metrics'].append(system_metrics)
        
        return system_metrics
    
    def get_performance_summary(self, hours=24):
        """Get performance summary for last N hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_predictions = [
            p for p in self.metrics['predictions']
            if datetime.fromisoformat(p['timestamp']) > cutoff_time
        ]
        
        recent_errors = [
            e for e in self.metrics['errors']
            if datetime.fromisoformat(e['timestamp']) > cutoff_time
        ]
        
        if recent_predictions:
            avg_response_time = sum(p['response_time'] for p in recent_predictions) / len(recent_predictions)
            max_response_time = max(p['response_time'] for p in recent_predictions)
        else:
            avg_response_time = max_response_time = 0
        
        summary = {
            'period_hours': hours,
            'total_predictions': len(recent_predictions),
            'total_errors': len(recent_errors),
            'error_rate': len(recent_errors) / max(len(recent_predictions), 1),
            'avg_response_time': avg_response_time,
            'max_response_time': max_response_time,
            'predictions_per_hour': len(recent_predictions) / hours
        }
        
        return summary
```

### **5.7 CI/CD Pipeline for ML Models**
**GitHub Actions Workflow:**
```yaml
# .github/workflows/ml-deploy.yml
name: ML Model Deployment

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest
    
    - name: Run tests
      run: |
        pytest tests/ -v
    
    - name: Test model loading
      run: |
        python -c "import joblib; model = joblib.load('models/model.pkl'); print('Model loaded successfully')"
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Deploy to Elastic Beanstalk
      run: |
        zip -r deploy.zip . -x "*.git*" "tests/*" "*.pytest_cache*"
        aws s3 cp deploy.zip s3://my-ml-app-deployments/
        aws elasticbeanstalk create-application-version \
          --application-name my-ml-app \
          --version-label ${{ github.sha }} \
          --source-bundle S3Bucket=my-ml-app-deployments,S3Key=deploy.zip
        aws elasticbeanstalk update-environment \
          --application-name my-ml-app \
          --environment-name production \
          --version-label ${{ github.sha }}
```

## 🛠️ Complete Deployment Example

```python
# complete_deployment.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

class ProductionMLApp:
    def __init__(self, model_path, config=None):
        self.app = Flask(__name__)
        self.model = joblib.load(model_path)
        self.config = config or {}
        self.setup_logging()
        self.setup_monitoring()
        self.setup_routes()
    
    def setup_logging(self):
        """Setup production logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_monitoring(self):
        """Setup monitoring and metrics"""
        self.monitor = ModelMonitor('production-model')
    
    def setup_routes(self):
        """Setup all routes"""
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            start_time = time.time()
            
            try:
                data = request.get_json()
                
                # Validate input
                if not self.validate_input(data):
                    return jsonify({'error': 'Invalid input'}), 400
                
                # Make prediction
                features = self.preprocess_input(data)
                prediction = self.model.predict([features])[0]
                
                # Log prediction
                response_time = time.time() - start_time
                self.monitor.log_prediction(data, prediction, response_time)
                
                return jsonify({
                    'prediction': float(prediction),
                    'timestamp': datetime.now().isoformat(),
                    'model_version': self.config.get('model_version', 'unknown')
                })
            
            except Exception as e:
                self.monitor.log_error('prediction_error', str(e))
                self.logger.error(f"Prediction error: {e}")
                return jsonify({'error': 'Prediction failed'}), 500
        
        @self.app.route('/health')
        def health():
            """Health check endpoint"""
            system_metrics = self.monitor.collect_system_metrics()
            performance = self.monitor.get_performance_summary(hours=1)
            
            return jsonify({
                'status': 'healthy',
                'system_metrics': system_metrics,
                'performance': performance,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/metrics')
        def metrics():
            """Metrics endpoint for monitoring systems"""
            return jsonify(self.monitor.get_performance_summary(hours=24))
    
    def validate_input(self, data):
        """Validate input data"""
        required_fields = self.config.get('required_fields', [])
        return all(field in data for field in required_fields)
    
    def preprocess_input(self, data):
        """Preprocess input for model"""
        # Implement your preprocessing logic here
        return list(data.values())
    
    def run(self, host='0.0.0.0', port=5000):
        """Run the application"""
        self.app.run(host=host, port=port)

if __name__ == '__main__':
    app = ProductionMLApp('models/model.pkl', {
        'model_version': '1.0.0',
        'required_fields': ['feature1', 'feature2', 'feature3']
    })
    app.run()
```

## 🎯 Module Completion Checklist

- [ ] Can deploy ML models as web services using Flask
- [ ] Understand containerization with Docker
- [ ] Know how to deploy to cloud platforms (AWS, GCP, Azure)
- [ ] Can implement model monitoring and logging
- [ ] Understand CI/CD pipelines for ML applications
- [ ] Can handle production challenges like scaling and error handling

## 🔗 Additional Resources

### **Video Lectures**
- [Deployment Playlist](https://www.youtube.com/playlist?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Documentation](https://docs.docker.com/)

### **Cloud Platform Guides**
- [AWS Elastic Beanstalk](https://aws.amazon.com/elasticbeanstalk/)
- [Google Cloud Run](https://cloud.google.com/run)
- [Azure Container Instances](https://azure.microsoft.com/en-us/services/container-instances/)

## 🎯 Next Steps

After completing this module, you're ready for **Module 6: Decision Trees and Ensemble Learning**, where you'll learn advanced tree-based algorithms.

---

**Navigation:**
- **Previous**: [Module 4: Evaluation](../04-evaluation/README.md)
- **Next**: [Module 6: Trees](../06-trees/README.md)
- **Course Home**: [Main Guide](../README.md)

*Last Updated: 2025-01-27*
