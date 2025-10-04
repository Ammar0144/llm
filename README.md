# DistilGPT### Core Capabilities
- **🧠 Lightweight Model**: DistilGPT-2 (82M parameters) optimized for text generation
- **⚡ FastAPI Framework**: High-performance REST API with automatic documentation
- **🎛️ Configurable Generation**: Adjustable parameters (temperature, top_p, max_tokens)
- **📝 Streamlined Interfaces**: Text generation, completion, and chat optimized for DistilGPT-2
- **🎯 Focused Performance**: Endpoints designed around model's core strengthsM Backend Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.7+-3776AB.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![API Version](https://img.shields.io/badge/API-v1.0.0-blue.svg)](https://github.com/Ammar0144/llm)

A secure, lightweight LLM server using DistilGPT-2 for text generation with FastAPI. Designed as an internal backend service for the AI Service API with comprehensive access control and production-ready features.

## 🚀 Features

### Core Capabilities
- **🧠 Lightweight Model**: DistilGPT-2 (82M parameters) for fast inference
- **⚡ FastAPI Framework**: High-performance REST API with automatic documentation
- **🎛️ Configurable Generation**: Adjustable parameters (temperature, top_p, max_length)
- **� Multiple Interfaces**: Text generation, Q&A, and model information endpoints

### Security & Access Control
- **🔒 IP-Based Access Control**: Restricts access to internal networks only
- **🛡️ Network Security**: Configurable IP whitelist for production environments
- **🚨 Health Monitoring**: Public health endpoint for load balancer monitoring
- **🔐 Environment-Based Config**: Flexible security configuration

### Production Features
- **🏭 Production Ready**: Designed for AI service backend integration
- **📊 Comprehensive Logging**: Detailed request and error logging
- **🔧 Easy Deployment**: Docker support with environment configuration
- **📖 Auto Documentation**: Interactive FastAPI docs and OpenAPI spec
- **🧹 Automated Maintenance**: Daily cleanup workflows for optimal storage management

## 🔒 Security Configuration

### Access Control Modes

#### 🛡️ Production Mode (Default - Secure)
```bash
# Restricts LLM access to internal networks only
python server.py
```
**Allowed Networks:**
- `127.0.0.0/8` - Localhost
- `10.0.0.0/8` - Private network
- `172.16.0.0/12` - Private network  
- `192.168.0.0/16` - Private network

#### 🔧 Development Mode (Open Access)
```bash
# Allows access from any IP (development only)
export DISABLE_IP_RESTRICTION=true
python server.py
```

#### ⚙️ Custom Configuration
```bash
# Custom allowed networks
export ALLOWED_NETWORKS="192.168.1.0/24,10.0.0.0/8"
python server.py
```

### Health Check Exception
The `/health` endpoint is **always accessible** from any IP for load balancer and monitoring purposes.

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- PyTorch
- 4GB+ RAM recommended

### Quick Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Ammar0144/llm.git
cd llm
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "from transformers import GPT2LMHeadModel; print('✅ Dependencies OK')"
```

## 🚀 Usage

### Starting the Server

#### Production Mode (Secure)
```bash
python server.py
```

#### Development Mode (Open Access)
```bash
export DISABLE_IP_RESTRICTION=true
python server.py
```

The server will start on `http://localhost:8082`

### 📖 Documentation & Monitoring

| Endpoint | Access | Description |
|----------|--------|-------------|
| `http://localhost:8082/docs` | Internal Only* | Interactive FastAPI documentation |
| `http://localhost:8082/health` | **Public** | Health check (always accessible) |
| `http://localhost:8082/redoc` | Internal Only* | Alternative API documentation |

*Internal Only: Restricted to allowed networks in production mode

### 🎯 API Endpoints

#### 🏥 Health Check (Public Access)
```http
GET /health
```
**Always accessible** - Used by load balancers and monitoring systems.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-09-27T19:30:00Z"
}
```

#### ℹ️ Model Information
```http
GET /model-info
```

**Response:**
```json
{
  "model_name": "distilgpt2",
  "model_type": "GPT2LMHeadModel",
  "parameters": "82M",
  "capabilities": ["text-generation", "conversation"],
  "max_length": 1024,
  "status": "loaded"
}
```

#### 🤖 Text Generation
```http
POST /generate
```

**Request Body:**
```json
{
  "prompt": "The future of artificial intelligence is",
  "max_length": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "do_sample": true
}
```

**Response:**
```json
{
  "generated_text": "The future of artificial intelligence is bright and full of possibilities, with advances in machine learning and neural networks leading to breakthrough applications in healthcare, education, and scientific research.",
  "prompt": "The future of artificial intelligence is",
  "metadata": {
    "generation_time": 1.23,
    "tokens_generated": 45,
    "model": "distilgpt2"
  }
}
```

#### 🎯 Text Completion (Optimized for DistilGPT-2)
```http
POST /complete
```

**Request Body:**
```json
{
  "prompt": "The benefits of machine learning include",
  "max_tokens": 100,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "prompt": "The benefits of machine learning include",
  "completion": "improved efficiency in data analysis, automated decision-making processes, and the ability to identify patterns in large datasets that would be impossible for humans to detect manually.",
  "raw_response": "The benefits of machine learning include improved efficiency in data analysis, automated decision-making processes, and the ability to identify patterns in large datasets that would be impossible for humans to detect manually."
}
```

#### 💬 Chat Completions (OpenAI Compatible)
```http
POST /chat/completions
```

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "Tell me about artificial intelligence"}
  ],
  "max_tokens": 150,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "content": "Artificial intelligence is a field of computer science focused on creating systems that can perform tasks typically requiring human intelligence.",
  "role": "assistant"
}
```

### 🧪 Testing the Server

#### Using the Provided Client
```bash
python client_example.py
```

#### Manual Testing with curl

##### Health Check (Always Works)
```bash
curl http://localhost:8082/health
```

##### Text Generation (Internal Networks Only)
```bash
# From allowed network
curl -X POST "http://localhost:8082/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Once upon a time",
       "max_length": 50,
       "temperature": 0.7
     }'
```

##### Text Completion Endpoint
```bash
curl -X POST "http://localhost:8082/complete" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Neural networks are computational models that",
       "max_tokens": 100,
       "temperature": 0.7
     }'
```

##### Chat Completions Endpoint
```bash
curl -X POST "http://localhost:8082/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [{"role": "user", "content": "Explain neural networks briefly"}],
       "max_tokens": 100
     }'
```

#### Access Control Testing

##### ✅ From Allowed Network (Success)
```bash
# From localhost or internal network
curl http://localhost:8082/model-info
# Response: 200 OK with model information
```

##### ❌ From External Network (Blocked)
```bash
# From external IP (if restrictions enabled)
curl http://external-server:8082/model-info
# Response: 403 Forbidden
```

##### ✅ Health Check Always Works
```bash
# From ANY network (health check exception)
curl http://external-server:8082/health
# Response: 200 OK with health status
```

#### Performance Testing
```bash
# Basic performance test
for i in {1..10}; do
  time curl -s -X POST "http://localhost:8082/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Test prompt", "max_length": 50}'
done
```

## ⚙️ Configuration

### 🎛️ Generation Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `prompt` | string | - | **required** | Input text to continue |
| `max_length` | integer | 1-1024 | 100 | Maximum length of generated text |
| `temperature` | float | 0.1-2.0 | 0.7 | Controls randomness (lower = more focused) |
| `top_p` | float | 0.1-1.0 | 0.9 | Nucleus sampling parameter |
| `do_sample` | boolean | - | true | Whether to use sampling vs greedy decoding |
| `repetition_penalty` | float | 0.1-2.0 | 1.0 | Penalty for repeated tokens |
| `pad_token_id` | integer | - | 50256 | Padding token ID for batch processing |

### 🔒 Security Configuration

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DISABLE_IP_RESTRICTION` | `false` | Disable IP access control for development |
| `ALLOWED_NETWORKS` | Internal networks | Comma-separated list of allowed CIDR networks |
| `HOST` | `0.0.0.0` | Server host binding |
| `PORT` | `8082` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

#### Network Configuration Examples

##### Allow Specific Subnet
```bash
export ALLOWED_NETWORKS="192.168.1.0/24"
python server.py
```

##### Allow Multiple Networks
```bash
export ALLOWED_NETWORKS="10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
python server.py
```

##### Development Mode (All Access)
```bash
export DISABLE_IP_RESTRICTION=true
python server.py
```

### 🔧 Advanced Configuration

#### Custom Model Loading
```python
# config.json - Custom model configuration
{
  "model_name": "distilgpt2",
  "max_length": 1024,
  "device": "auto",
  "torch_dtype": "float16"
}
```

#### Performance Tuning
```bash
# For better performance on CPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# For GPU usage (if available)
export CUDA_VISIBLE_DEVICES=0

python server.py
```

## Model Details

- **Model**: DistilGPT-2
- **Parameters**: 82M
- **Type**: Causal Language Model
- **Description**: A distilled version of GPT-2 that maintains good performance while being smaller and faster

## ⚡ CI/CD & Automation

### Optimized Build Process

The LLM service includes streamlined CI/CD workflows:

#### **Automated Testing Pipeline**
- **Lightweight Builds**: Fast syntax and dependency checking
- **Python Environment**: Automated Python 3.11 setup with pip caching
- **No Heavy Testing**: Optimized for quick validation without resource-intensive model testing
- **Build Verification**: Syntax validation for core Python files

#### **Automated Maintenance**
- **Daily Cleanup**: Automatic removal of old workflow runs and artifacts
- **Storage Management**: Keeps last 30 workflow runs to manage GitHub storage
- **Efficient Workflow**: No artifact uploads for faster builds and lower storage usage
- **Background Cleanup**: Scheduled maintenance at 2 AM UTC daily

#### **Manual Cleanup Tools**
- **PowerShell Script**: `cleanup-artifacts.ps1` for Windows environments
- **Bash Script**: `cleanup-artifacts.sh` for Linux/macOS environments
- **GitHub CLI Integration**: Direct API access for immediate cleanup
- **Storage Monitoring**: Tools to check and manage GitHub storage quotas

### Benefits of Optimized CI/CD
- **Fast Builds**: Quick validation without heavy model loading
- **Storage Efficient**: Minimal artifact generation and automatic cleanup
- **Production Ready**: Focuses on deployment preparation rather than extensive testing
- **Cost Effective**: Reduces GitHub Actions minutes and storage costs

## 🐳 Deployment

### Docker Deployment (Recommended)

#### 🚀 Quick Start with Docker Compose
```bash
# Clone and start both services
git clone https://github.com/Ammar0144/llm.git
cd llm

# Start LLM backend with AI service
docker-compose up -d

# Verify deployment
curl http://localhost:8082/health
curl http://localhost:8081/health  # AI service
```

#### 📦 Standalone Docker
```bash
# Build the image
docker build -t llm-backend .

# Run with default security (internal networks only)
docker run -d \
  --name llm-backend \
  -p 8082:8082 \
  llm-backend

# Run with custom configuration
docker run -d \
  --name llm-backend \
  -p 8082:8082 \
  -e ALLOWED_NETWORKS="10.0.0.0/8,172.16.0.0/12" \
  -e LOG_LEVEL=DEBUG \
  llm-backend
```

#### 🔗 Integration with AI Service
```yaml
# docker-compose.yml
version: '3.8'
services:
  ai-service:
    build: ../ai
    ports:
      - "8081:8081"
    environment:
      - LLM_SERVICE_URL=http://llm-backend:8082
    depends_on:
      - llm-backend

  llm-backend:
    build: .
    ports:
      - "8082:8082"
    environment:
      - ALLOWED_NETWORKS=172.16.0.0/12  # Docker internal network
```

### 🏭 Production Deployment

#### 🎯 Deployment Checklist

1. **✅ Security Configuration**
```bash
# Ensure IP restrictions are enabled (default)
unset DISABLE_IP_RESTRICTION

# Configure allowed networks for your infrastructure
export ALLOWED_NETWORKS="10.0.0.0/8,172.16.0.0/12"
```

2. **✅ Resource Requirements**
- **Minimum**: 2GB RAM, 1 CPU core
- **Recommended**: 4GB RAM, 2+ CPU cores
- **Storage**: 2GB for model and dependencies

3. **✅ Network Configuration**
```bash
# Internal service (recommended)
# Only AI service should be publicly accessible
# LLM backend should be internal-only

AI Service (Public)  →  LLM Backend (Internal)
Port 8081                Port 8082 (restricted)
```

#### ☁️ Cloud Deployment Options

##### **Docker on Cloud VM**
```bash
# Deploy on any cloud provider
ssh user@cloud-server
git clone https://github.com/Ammar0144/llm.git
cd llm

# Start with production settings
docker-compose -f docker-compose.prod.yml up -d
```

##### **Kubernetes**
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm-backend
  template:
    spec:
      containers:
      - name: llm-backend
        image: llm-backend:latest
        ports:
        - containerPort: 8082
        env:
        - name: ALLOWED_NETWORKS
          value: "10.244.0.0/16"  # K8s pod network
```

##### **AWS ECS/Fargate**
```bash
# Build and deploy
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin
docker build -t llm-backend .
docker tag llm-backend:latest ECR_URI/llm-backend:latest
docker push ECR_URI/llm-backend:latest

# Deploy via ECS console or Terraform
```

### � Monitoring & Health Checks

#### Health Check Configuration
```bash
# Health endpoint is always accessible
curl http://your-server:8082/health

# For load balancers
# Use /health endpoint which bypasses IP restrictions
```

#### Docker Health Check
```dockerfile
# Dockerfile health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8082/health || exit 1
```

#### Production Monitoring
```bash
# Log monitoring
docker logs -f llm-backend

# Resource monitoring
docker stats llm-backend

# Network monitoring
netstat -tulpn | grep 8082
```

### 🔧 Troubleshooting

#### Common Issues

##### ❌ Access Denied (403 Forbidden)
```bash
# Check if IP restriction is blocking your requests
export DISABLE_IP_RESTRICTION=true  # For testing only
python server.py

# Or configure allowed networks
export ALLOWED_NETWORKS="your-network/24"
```

##### ❌ Model Loading Failed
```bash
# Check available memory
free -h

# Check disk space
df -h

# Restart with debug logging
export LOG_LEVEL=DEBUG
python server.py
```

##### ❌ Connection Refused
```bash
# Check if service is running
curl http://localhost:8082/health

# Check port binding
netstat -tulpn | grep 8082

# Check firewall rules
ufw status
```

## 🏗️ Architecture Integration

### System Architecture
```
┌─────────────────────┐    ┌─────────────────────┐
│   AI Service        │────│   LLM Backend       │
│   (Public Access)   │    │   (Internal Only)   │
│   Port 8081         │    │   Port 8082         │
│                     │    │                     │
│   • Rate Limiting   │    │   • IP Access Control│
│   • CORS Policy     │    │   • Text Generation │
│   • API Gateway     │    │   • Model Management│
│   • Documentation   │    │   • Health Monitoring│
└─────────────────────┘    └─────────────────────┘
```

### 🔗 AI Service Integration

The LLM backend is designed specifically to work with the AI Service:

#### Communication Flow
1. **Client** → AI Service (Public API)
2. **AI Service** → LLM Backend (Internal)
3. **LLM Backend** → AI Service (Response)
4. **AI Service** → Client (Formatted Response)

#### Integration Benefits
- **Security**: LLM backend protected from direct external access
- **Performance**: Optimized internal communication
- **Scalability**: Independent scaling of AI service and LLM backend
- **Monitoring**: Centralized logging and health checks

### 🔒 Security Model

#### Multi-Layer Security
```
Internet Traffic
       ↓
   Firewall/Load Balancer
       ↓
   AI Service (Public)
   • Rate limiting
   • Input validation
   • Response formatting
       ↓
   Internal Network
       ↓
   LLM Backend (Private)
   • IP access control
   • Model security
   • Resource management
```

#### Access Control Details

##### **Always Accessible**
- `/health` - Health monitoring (bypasses IP restrictions)

##### **Internal Network Only** 
- `/generate` - Text generation (primary strength)
- `/complete` - Text completion (primary strength)
- `/chat/completions` - Chat conversations (OpenAI compatible)
- `/model-info` - Model information
- `/docs` - API documentation

##### **Network Restrictions**
- **Localhost**: `127.0.0.0/8`
- **Private Class A**: `10.0.0.0/8`
- **Private Class B**: `172.16.0.0/12`
- **Private Class C**: `192.168.0.0/16`
- **Docker Networks**: Automatically detected

## 🛡️ Security Best Practices

### Production Security Checklist

#### ✅ Network Security
```bash
# Ensure IP restrictions are enabled (default)
# Only disable for development environments
if [[ "$ENV" == "development" ]]; then
    export DISABLE_IP_RESTRICTION=true
else
    unset DISABLE_IP_RESTRICTION  # Secure by default
fi
```

#### ✅ Firewall Configuration
```bash
# Allow only necessary ports
ufw allow 8082/tcp from 10.0.0.0/8    # Internal network only
ufw allow 8082/tcp from 172.16.0.0/12  # Docker networks
ufw allow 8082/tcp from 192.168.0.0/16 # Local networks
ufw deny 8082/tcp from any             # Block everything else
```

#### ✅ Container Security
```bash
# Run container as non-root user
docker run --user 1000:1000 llm-backend

# Limit container resources
docker run --memory=4g --cpus=2 llm-backend

# Use read-only filesystem where possible
docker run --read-only --tmpfs /tmp llm-backend
```

#### ✅ Monitoring & Alerting
```bash
# Monitor access attempts
tail -f /var/log/llm-backend/access.log | grep "403 Forbidden"

# Alert on external access attempts
# Set up log monitoring for blocked IPs
```

### 🔧 Development vs Production

#### Development Mode
```bash
# Open access for development
export DISABLE_IP_RESTRICTION=true
export LOG_LEVEL=DEBUG
python server.py
```

#### Production Mode
```bash
# Secure configuration (default)
export LOG_LEVEL=INFO
export ALLOWED_NETWORKS="10.0.0.0/8,172.16.0.0/12"
python server.py
```

## 📋 Requirements

### System Requirements
- **Python**: 3.7+ (3.9+ recommended)
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB for model and dependencies
- **CPU**: Multi-core recommended for better performance

### Dependencies
```
torch>=1.9.0          # PyTorch for model inference
transformers>=4.20.0   # Hugging Face transformers
fastapi>=0.68.0       # FastAPI web framework
uvicorn>=0.15.0       # ASGI server
pydantic>=1.8.0       # Data validation
python-multipart      # File upload support
```

### Optional Dependencies
```
# For better performance
torch-audio           # Audio processing capabilities
accelerate            # Model acceleration

# For monitoring
prometheus-client     # Metrics collection
structlog            # Structured logging
```

## 📚 Additional Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [DistilGPT-2 Model Card](https://huggingface.co/distilgpt2)

### Related Projects
- [AI Service Frontend](https://github.com/Ammar0144/ai) - Public-facing API service
- [Model Deployment Examples](https://github.com/huggingface/transformers/tree/main/examples)

### 🧹 Storage & Artifact Management

#### Automated Cleanup Features

The LLM repository includes efficient storage management:

##### **Daily Cleanup Workflow**
- **Automatic Execution**: Runs daily at 2 AM UTC
- **Workflow Cleanup**: Maintains only the last 30 workflow runs
- **Artifact Management**: Removes artifacts older than 7 days (when present)
- **Storage Optimization**: Prevents GitHub storage quota issues

##### **Manual Cleanup Tools**

When immediate cleanup is needed:

```bash
# Using provided cleanup scripts
./cleanup-artifacts.sh          # Linux/macOS
.\cleanup-artifacts.ps1         # Windows PowerShell

# Or using GitHub CLI directly
gh auth login
gh api repos/Ammar0144/llm/actions/runs --paginate | \
  jq '.workflow_runs[30:] | .[] | .id' | \
  xargs -I {} gh api repos/Ammar0144/llm/actions/runs/{} -X DELETE
```

##### **Storage Monitoring**
- **Check Usage**: [GitHub Billing Settings](https://github.com/settings/billing)
- **Minimal Artifacts**: LLM builds generate no artifacts by design
- **Efficient Workflows**: Optimized for speed and low storage impact
- **Preventive Maintenance**: Regular cleanup prevents quota issues

### Support & Contributing

#### Getting Help
- Check the [Issues](https://github.com/Ammar0144/llm/issues) for common problems
- Review logs with `export LOG_LEVEL=DEBUG`
- Test with health endpoint: `curl http://localhost:8082/health`

#### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Add tests for new functionality
4. Ensure security features remain intact
5. Submit a pull request with detailed description

#### Security Reports
For security vulnerabilities, please email privately rather than opening public issues.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🏷️ Version Information

- **Current Version**: 1.0.0
- **API Version**: v1
- **Model**: DistilGPT-2 (82M parameters)
- **Python Version**: 3.7+ (3.11+ recommended)
- **Last Updated**: September 2025

### Initial Release (v1.0.0)
- ✅ Complete LLM backend with DistilGPT-2 integration
- ✅ IP-based access control for production security
- ✅ FastAPI framework with automatic documentation
- ✅ Docker containerization and CI/CD pipeline
- ✅ Production-ready deployment configurations
- ✅ Seamless integration with AI Service

---

### 🚀 Quick Links
- [🔧 AI Service Repository](https://github.com/Ammar0144/ai)
- [📖 API Documentation](http://localhost:8082/docs) (when running)
- [🏥 Health Check](http://localhost:8082/health)
- [🐳 Docker Hub](https://hub.docker.com/) (if published)
