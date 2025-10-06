# 🎯 Module 10: Kubernetes and TensorFlow Serving

> **Deploy and scale machine learning models using Kubernetes and TensorFlow Serving for production environments**

This module covers enterprise-grade ML model deployment using Kubernetes orchestration and TensorFlow Serving. You'll learn to build scalable, resilient ML inference systems.

## 📚 Learning Objectives

By the end of this module, you will:
- **Deploy** ML models using TensorFlow Serving
- **Orchestrate** ML services with Kubernetes
- **Implement** auto-scaling and load balancing
- **Monitor** production ML systems
- **Handle** model versioning and A/B testing
- **Ensure** high availability and fault tolerance

## ⚙️ Kubernetes for ML Fundamentals

### Why Kubernetes for ML?
✅ **Advantages:**
- **Scalability**: Auto-scale based on demand
- **Reliability**: Self-healing and fault tolerance
- **Resource management**: Efficient GPU/CPU allocation
- **Service discovery**: Automatic load balancing
- **Rolling updates**: Zero-downtime deployments

### TensorFlow Serving Benefits
✅ **Features:**
- **High performance**: Optimized for inference
- **Model versioning**: Multiple model versions
- **Batching**: Automatic request batching
- **Monitoring**: Built-in metrics and health checks
- **gRPC/REST APIs**: Flexible client interfaces

## 🗂️ Module Contents

### **10.1 TensorFlow Serving Setup**
**Model Preparation and Serving:**
```python
import tensorflow as tf
import os
import shutil
from datetime import datetime

class TensorFlowServingPreparation:
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
        self.serving_dir = f"./serving_models/{model_name}"
    
    def prepare_model_for_serving(self, version=None):
        """Prepare Keras model for TensorFlow Serving"""
        
        if version is None:
            version = int(datetime.now().timestamp())
        
        # Create serving directory structure
        model_version_dir = f"{self.serving_dir}/{version}"
        os.makedirs(model_version_dir, exist_ok=True)
        
        # Load and save model in SavedModel format
        model = tf.keras.models.load_model(self.model_path)
        
        # Save in TensorFlow Serving format
        tf.saved_model.save(model, model_version_dir)
        
        print(f"Model prepared for serving:")
        print(f"  Model name: {self.model_name}")
        print(f"  Version: {version}")
        print(f"  Path: {model_version_dir}")
        
        # Verify the saved model
        self.verify_saved_model(model_version_dir)
        
        return model_version_dir, version
    
    def verify_saved_model(self, model_dir):
        """Verify the saved model structure"""
        
        # Check if required files exist
        required_files = ['saved_model.pb', 'variables']
        
        for file in required_files:
            file_path = os.path.join(model_dir, file)
            if os.path.exists(file_path):
                print(f"✅ {file} found")
            else:
                print(f"❌ {file} missing")
        
        # Load and inspect the model
        try:
            loaded_model = tf.saved_model.load(model_dir)
            
            # Get signature information
            signatures = loaded_model.signatures
            print(f"Available signatures: {list(signatures.keys())}")
            
            if 'serving_default' in signatures:
                serving_signature = signatures['serving_default']
                print(f"Input spec: {serving_signature.structured_input_signature}")
                print(f"Output spec: {serving_signature.structured_outputs}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def create_model_config(self, versions):
        """Create model configuration for TensorFlow Serving"""
        
        config = {
            "model_config_list": [
                {
                    "name": self.model_name,
                    "base_path": f"/models/{self.model_name}",
                    "model_platform": "tensorflow",
                    "model_version_policy": {
                        "specific": {
                            "versions": versions
                        }
                    }
                }
            ]
        }
        
        # Save configuration
        import json
        config_path = f"{self.serving_dir}/model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model configuration saved to: {config_path}")
        
        return config_path
```

### **10.2 Docker Containerization**
**TensorFlow Serving Docker Setup:**
```dockerfile
# Dockerfile for TensorFlow Serving
FROM tensorflow/serving:latest

# Copy model to container
COPY serving_models/fashion_classifier /models/fashion_classifier

# Set environment variables
ENV MODEL_NAME=fashion_classifier
ENV MODEL_BASE_PATH=/models

# Expose ports
EXPOSE 8500 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/v1/models/${MODEL_NAME} || exit 1

# Start TensorFlow Serving
CMD ["tensorflow_model_server", \
     "--rest_api_port=8501", \
     "--model_name=${MODEL_NAME}", \
     "--model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}"]
```

**Client Application Container:**
```dockerfile
# Dockerfile for ML client application
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

### **10.3 Kubernetes Deployment Manifests**
**Complete Kubernetes Configuration:**
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-serving
  labels:
    name: ml-serving

---
# tensorflow-serving-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tensorflow-serving
  namespace: ml-serving
  labels:
    app: tensorflow-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tensorflow-serving
  template:
    metadata:
      labels:
        app: tensorflow-serving
    spec:
      containers:
      - name: tensorflow-serving
        image: tensorflow/serving:latest
        ports:
        - containerPort: 8500
          name: grpc
        - containerPort: 8501
          name: rest
        env:
        - name: MODEL_NAME
          value: "fashion_classifier"
        - name: MODEL_BASE_PATH
          value: "/models"
        volumeMounts:
        - name: model-volume
          mountPath: /models
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /v1/models/fashion_classifier
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /v1/models/fashion_classifier
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc

---
# tensorflow-serving-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: tensorflow-serving-service
  namespace: ml-serving
spec:
  selector:
    app: tensorflow-serving
  ports:
  - name: grpc
    port: 8500
    targetPort: 8500
  - name: rest
    port: 8501
    targetPort: 8501
  type: ClusterIP

---
# model-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
  namespace: ml-serving
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard

---
# horizontal-pod-autoscaler.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tensorflow-serving-hpa
  namespace: ml-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tensorflow-serving
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-serving-ingress
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - ml-api.yourdomain.com
    secretName: ml-serving-tls
  rules:
  - host: ml-api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: tensorflow-serving-service
            port:
              number: 8501
```

### **10.4 Advanced Client Implementation**
**Production-Ready Client with Connection Pooling:**
```python
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor
import logging

class TensorFlowServingClient:
    def __init__(self, server_url, model_name, model_version=None, 
                 use_grpc=True, max_workers=10):
        self.server_url = server_url
        self.model_name = model_name
        self.model_version = model_version or 'latest'
        self.use_grpc = use_grpc
        self.max_workers = max_workers
        
        if use_grpc:
            self.setup_grpc_client()
        else:
            self.setup_rest_client()
    
    def setup_grpc_client(self):
        """Setup gRPC client with connection pooling"""
        
        # Create gRPC channel with options for production
        options = [
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.min_ping_interval_without_data_ms', 300000)
        ]
        
        self.channel = grpc.insecure_channel(self.server_url, options=options)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        
        # Test connection
        self.test_connection()
    
    def setup_rest_client(self):
        """Setup REST client with session pooling"""
        
        self.session = requests.Session()
        
        # Configure session for production
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers,
            max_retries=3
        )
        
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Test connection
        self.test_connection()
    
    def test_connection(self):
        """Test connection to TensorFlow Serving"""
        
        try:
            if self.use_grpc:
                # Test gRPC connection
                request = predict_pb2.PredictRequest()
                request.model_spec.name = self.model_name
                request.model_spec.signature_name = 'serving_default'
                
                # This will fail but confirms connection
                try:
                    self.stub.Predict(request, timeout=5.0)
                except grpc.RpcError as e:
                    if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                        logging.info("gRPC connection successful")
                    else:
                        raise
            else:
                # Test REST connection
                url = f"{self.server_url}/v1/models/{self.model_name}"
                response = self.session.get(url, timeout=5)
                if response.status_code == 200:
                    logging.info("REST connection successful")
                else:
                    raise Exception(f"Connection failed: {response.status_code}")
        
        except Exception as e:
            logging.error(f"Connection test failed: {e}")
            raise
    
    def predict_grpc(self, input_data):
        """Make prediction using gRPC"""
        
        # Create prediction request
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = 'serving_default'
        
        if self.model_version != 'latest':
            request.model_spec.version.value = int(self.model_version)
        
        # Convert input data to tensor proto
        request.inputs['input_1'].CopyFrom(
            tf.make_tensor_proto(input_data, dtype=tf.float32)
        )
        
        # Make prediction
        start_time = time.time()
        result = self.stub.Predict(request, timeout=30.0)
        inference_time = time.time() - start_time
        
        # Extract predictions
        predictions = tf.make_ndarray(result.outputs['predictions'])
        
        return {
            'predictions': predictions,
            'inference_time': inference_time
        }
    
    def predict_rest(self, input_data):
        """Make prediction using REST API"""
        
        url = f"{self.server_url}/v1/models/{self.model_name}"
        if self.model_version != 'latest':
            url += f"/versions/{self.model_version}"
        url += ":predict"
        
        # Prepare request data
        data = {
            "instances": input_data.tolist()
        }
        
        # Make prediction
        start_time = time.time()
        response = self.session.post(url, json=data, timeout=30)
        inference_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            return {
                'predictions': np.array(result['predictions']),
                'inference_time': inference_time
            }
        else:
            raise Exception(f"Prediction failed: {response.status_code} - {response.text}")
    
    def predict(self, input_data):
        """Make prediction (automatically choose gRPC or REST)"""
        
        if self.use_grpc:
            return self.predict_grpc(input_data)
        else:
            return self.predict_rest(input_data)
    
    def batch_predict(self, input_batch, batch_size=32):
        """Make batch predictions with parallel processing"""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Split into smaller batches
            batches = [
                input_batch[i:i + batch_size] 
                for i in range(0, len(input_batch), batch_size)
            ]
            
            # Submit all batches
            futures = [
                executor.submit(self.predict, batch) 
                for batch in batches
            ]
            
            # Collect results
            for future in futures:
                result = future.result()
                results.append(result)
        
        # Combine results
        all_predictions = np.concatenate([r['predictions'] for r in results])
        total_time = sum(r['inference_time'] for r in results)
        
        return {
            'predictions': all_predictions,
            'total_inference_time': total_time,
            'average_inference_time': total_time / len(batches)
        }
    
    def benchmark_performance(self, input_data, num_requests=100, concurrent_requests=10):
        """Benchmark serving performance"""
        
        print(f"Benchmarking with {num_requests} requests, {concurrent_requests} concurrent...")
        
        def single_request():
            return self.predict(input_data)
        
        # Warm up
        for _ in range(10):
            single_request()
        
        # Benchmark
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [
                executor.submit(single_request) 
                for _ in range(num_requests)
            ]
            
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        inference_times = [r['inference_time'] for r in results]
        
        metrics = {
            'total_requests': num_requests,
            'total_time': total_time,
            'requests_per_second': num_requests / total_time,
            'average_inference_time': np.mean(inference_times),
            'p50_inference_time': np.percentile(inference_times, 50),
            'p95_inference_time': np.percentile(inference_times, 95),
            'p99_inference_time': np.percentile(inference_times, 99)
        }
        
        print("=== BENCHMARK RESULTS ===")
        for key, value in metrics.items():
            if 'time' in key:
                print(f"{key}: {value:.3f}s")
            else:
                print(f"{key}: {value:.2f}")
        
        return metrics
    
    def close(self):
        """Close connections"""
        
        if self.use_grpc and hasattr(self, 'channel'):
            self.channel.close()
        elif hasattr(self, 'session'):
            self.session.close()
```

### **10.5 Monitoring and Observability**
**Comprehensive Monitoring Setup:**
```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: ml-serving
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'tensorflow-serving'
      static_configs:
      - targets: ['tensorflow-serving-service:8501']
      metrics_path: '/monitoring/prometheus/metrics'
    - job_name: 'kubernetes-pods'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - ml-serving

---
# grafana-dashboard.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard
  namespace: ml-serving
data:
  tensorflow-serving.json: |
    {
      "dashboard": {
        "title": "TensorFlow Serving Dashboard",
        "panels": [
          {
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(tensorflow_serving_request_count[5m])",
                "legendFormat": "Requests/sec"
              }
            ]
          },
          {
            "title": "Request Latency",
            "type": "graph",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(tensorflow_serving_request_latency_bucket[5m]))",
                "legendFormat": "95th percentile"
              }
            ]
          },
          {
            "title": "Model Loading Time",
            "type": "graph",
            "targets": [
              {
                "expr": "tensorflow_serving_model_warmup_latency",
                "legendFormat": "Model warmup time"
              }
            ]
          }
        ]
      }
    }
```

### **10.6 A/B Testing and Model Versioning**
**Advanced Model Management:**
```python
class ModelVersionManager:
    def __init__(self, kubernetes_client):
        self.k8s_client = kubernetes_client
        self.namespace = "ml-serving"
    
    def deploy_new_version(self, model_version, traffic_percentage=10):
        """Deploy new model version with traffic splitting"""
        
        # Create new deployment for the version
        deployment_name = f"tensorflow-serving-v{model_version}"
        
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": deployment_name,
                "namespace": self.namespace,
                "labels": {
                    "app": "tensorflow-serving",
                    "version": f"v{model_version}"
                }
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": "tensorflow-serving",
                        "version": f"v{model_version}"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "tensorflow-serving",
                            "version": f"v{model_version}"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "tensorflow-serving",
                            "image": "tensorflow/serving:latest",
                            "env": [
                                {
                                    "name": "MODEL_NAME",
                                    "value": "fashion_classifier"
                                },
                                {
                                    "name": "MODEL_BASE_PATH",
                                    "value": "/models"
                                }
                            ],
                            "ports": [
                                {"containerPort": 8500, "name": "grpc"},
                                {"containerPort": 8501, "name": "rest"}
                            ]
                        }]
                    }
                }
            }
        }
        
        # Deploy new version
        self.k8s_client.create_namespaced_deployment(
            namespace=self.namespace,
            body=deployment_manifest
        )
        
        # Update service to include new version
        self.update_traffic_split(model_version, traffic_percentage)
        
        print(f"Deployed model version {model_version} with {traffic_percentage}% traffic")
    
    def update_traffic_split(self, new_version, new_traffic_percentage):
        """Update traffic split between model versions"""
        
        # This would typically use Istio or similar service mesh
        # For simplicity, we'll update service selector weights
        
        service_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "tensorflow-serving-service",
                "namespace": self.namespace,
                "annotations": {
                    f"traffic.v{new_version}": str(new_traffic_percentage),
                    "traffic.stable": str(100 - new_traffic_percentage)
                }
            },
            "spec": {
                "selector": {
                    "app": "tensorflow-serving"
                },
                "ports": [
                    {"name": "grpc", "port": 8500, "targetPort": 8500},
                    {"name": "rest", "port": 8501, "targetPort": 8501}
                ]
            }
        }
        
        self.k8s_client.patch_namespaced_service(
            name="tensorflow-serving-service",
            namespace=self.namespace,
            body=service_manifest
        )
    
    def monitor_ab_test(self, duration_minutes=60):
        """Monitor A/B test metrics"""
        
        import time
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        metrics = {
            'stable_version': {'requests': 0, 'errors': 0, 'latency': []},
            'new_version': {'requests': 0, 'errors': 0, 'latency': []}
        }
        
        while time.time() < end_time:
            # Collect metrics from Prometheus
            # This is a simplified version
            
            time.sleep(60)  # Check every minute
            
            print(f"A/B Test Progress: {(time.time() - start_time) / 60:.1f}/{duration_minutes} minutes")
        
        # Analyze results
        self.analyze_ab_test_results(metrics)
        
        return metrics
    
    def analyze_ab_test_results(self, metrics):
        """Analyze A/B test results and make recommendations"""
        
        stable_error_rate = metrics['stable_version']['errors'] / max(metrics['stable_version']['requests'], 1)
        new_error_rate = metrics['new_version']['errors'] / max(metrics['new_version']['requests'], 1)
        
        stable_avg_latency = np.mean(metrics['stable_version']['latency']) if metrics['stable_version']['latency'] else 0
        new_avg_latency = np.mean(metrics['new_version']['latency']) if metrics['new_version']['latency'] else 0
        
        print("=== A/B TEST RESULTS ===")
        print(f"Stable version error rate: {stable_error_rate:.3f}")
        print(f"New version error rate: {new_error_rate:.3f}")
        print(f"Stable version avg latency: {stable_avg_latency:.3f}ms")
        print(f"New version avg latency: {new_avg_latency:.3f}ms")
        
        # Decision logic
        if new_error_rate < stable_error_rate * 1.1 and new_avg_latency < stable_avg_latency * 1.2:
            print("✅ Recommendation: Promote new version")
            return "promote"
        else:
            print("❌ Recommendation: Rollback to stable version")
            return "rollback"
    
    def promote_version(self, version):
        """Promote new version to 100% traffic"""
        
        self.update_traffic_split(version, 100)
        print(f"Version {version} promoted to 100% traffic")
    
    def rollback_version(self, version):
        """Rollback to previous stable version"""
        
        # Scale down the new version
        self.k8s_client.patch_namespaced_deployment_scale(
            name=f"tensorflow-serving-v{version}",
            namespace=self.namespace,
            body={"spec": {"replicas": 0}}
        )
        
        # Reset traffic to stable version
        self.update_traffic_split("stable", 100)
        
        print(f"Rolled back version {version}")
```

## 🛠️ Complete Kubernetes ML Pipeline

```python
class KubernetesMLPipeline:
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
        self.serving_prep = TensorFlowServingPreparation(model_path, model_name)
    
    def deploy_complete_pipeline(self):
        """Deploy complete ML pipeline to Kubernetes"""
        
        print("=== KUBERNETES ML DEPLOYMENT PIPELINE ===")
        
        # 1. Prepare model for serving
        print("1. Preparing model for TensorFlow Serving...")
        model_dir, version = self.serving_prep.prepare_model_for_serving()
        
        # 2. Build and push Docker images
        print("2. Building and pushing Docker images...")
        self.build_and_push_images()
        
        # 3. Deploy to Kubernetes
        print("3. Deploying to Kubernetes...")
        self.deploy_to_kubernetes()
        
        # 4. Setup monitoring
        print("4. Setting up monitoring...")
        self.setup_monitoring()
        
        # 5. Test deployment
        print("5. Testing deployment...")
        self.test_deployment()
        
        print("🎉 Kubernetes ML pipeline deployed successfully!")
    
    def build_and_push_images(self):
        """Build and push Docker images"""
        
        # Build TensorFlow Serving image
        os.system(f"""
        docker build -t {self.model_name}-serving:latest -f Dockerfile.serving .
        docker tag {self.model_name}-serving:latest your-registry/{self.model_name}-serving:latest
        docker push your-registry/{self.model_name}-serving:latest
        """)
        
        print("Docker images built and pushed successfully!")
    
    def deploy_to_kubernetes(self):
        """Deploy all Kubernetes resources"""
        
        # Apply all manifests
        manifests = [
            "namespace.yaml",
            "model-pvc.yaml",
            "tensorflow-serving-deployment.yaml",
            "tensorflow-serving-service.yaml",
            "horizontal-pod-autoscaler.yaml",
            "ingress.yaml"
        ]
        
        for manifest in manifests:
            os.system(f"kubectl apply -f {manifest}")
        
        print("Kubernetes resources deployed successfully!")
    
    def setup_monitoring(self):
        """Setup monitoring stack"""
        
        # Deploy Prometheus and Grafana
        os.system("kubectl apply -f prometheus-config.yaml")
        os.system("kubectl apply -f grafana-dashboard.yaml")
        
        print("Monitoring stack deployed successfully!")
    
    def test_deployment(self):
        """Test the deployed ML service"""
        
        # Wait for deployment to be ready
        os.system("kubectl wait --for=condition=available --timeout=300s deployment/tensorflow-serving -n ml-serving")
        
        # Port forward for testing
        import subprocess
        import time
        
        # Start port forward in background
        port_forward = subprocess.Popen([
            "kubectl", "port-forward", 
            "service/tensorflow-serving-service", 
            "8501:8501", "-n", "ml-serving"
        ])
        
        time.sleep(5)  # Wait for port forward to establish
        
        try:
            # Test the service
            client = TensorFlowServingClient(
                server_url="http://localhost:8501",
                model_name=self.model_name,
                use_grpc=False
            )
            
            # Create test data
            test_data = np.random.random((1, 224, 224, 3)).astype(np.float32)
            
            # Make prediction
            result = client.predict(test_data)
            
            print("✅ Deployment test successful!")
            print(f"Prediction shape: {result['predictions'].shape}")
            print(f"Inference time: {result['inference_time']:.3f}s")
            
        finally:
            # Clean up port forward
            port_forward.terminate()

# Usage example
if __name__ == "__main__":
    pipeline = KubernetesMLPipeline("fashion_classifier.h5", "fashion_classifier")
    pipeline.deploy_complete_pipeline()
```

## 🎯 Module Completion Checklist

- [ ] Can prepare models for TensorFlow Serving
- [ ] Understand Kubernetes deployment manifests for ML services
- [ ] Can implement auto-scaling and load balancing
- [ ] Know how to monitor ML services in production
- [ ] Can implement A/B testing and model versioning
- [ ] Understand high availability and fault tolerance patterns

## 🔗 Additional Resources

### **Video Lectures**
- [Kubernetes and TensorFlow Serving Playlist](https://www.youtube.com/playlist?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)
- [TensorFlow Serving Documentation](https://www.tensorflow.org/tfx/guide/serving)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

### **Tools and Platforms**
- [Kubeflow](https://www.kubeflow.org/) - ML workflows on Kubernetes
- [Seldon Core](https://www.seldon.io/) - ML deployment on Kubernetes
- [KServe](https://kserve.github.io/website/) - Serverless ML inference

## 🎯 Course Completion

Congratulations! You have completed the comprehensive ML Zoomcamp guide. You now have the skills to:

- Build end-to-end ML projects from data to deployment
- Deploy models using various strategies (serverless, containers, Kubernetes)
- Monitor and maintain ML systems in production
- Handle real-world ML challenges and best practices

---

**Navigation:**
- **Previous**: [Module 9: Serverless](../09-serverless/README.md)
- **Course Home**: [Main Guide](../README.md)

*Last Updated: 2025-01-27*
