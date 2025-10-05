# DistilGPT-2 LLM Backend Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.7+-3776AB.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![API Version](https://img.shields.io/badge/API-v1.0.0-blue.svg)](https://github.com/Ammar0144/llm)

> **🎓 Learning Project**: This is an educational LLM server designed for learning and experimentation. Built with affordability in mind to run on modest hardware, making AI accessible to all learners!

A secure, lightweight LLM server using DistilGPT-2 for text generation with FastAPI. Designed as an internal backend service for the AI Service API with comprehensive access control and production-ready features.

## 📚 About This Project

This is a **learning-focused** LLM server designed to help developers understand how to:
- 🧠 **Deploy LLM Models**: Learn to set up and serve language models
- ⚡ **Build AI APIs**: Create production-ready API endpoints with FastAPI
- 🔒 **Implement Security**: Understand access control and security best practices
- 🐳 **Containerize AI Services**: Master Docker deployment for AI applications

### 🎯 Why DistilGPT-2?

We chose **DistilGPT-2 (82M parameters)** specifically because it's:
- 💰 **Affordable**: Runs on low-spec servers (even 2GB RAM)
- 🚀 **Fast**: Quick inference times for learning and testing
- 🎓 **Educational**: Perfect for understanding LLM fundamentals
- 🌍 **Accessible**: Makes AI learning available to everyone, regardless of hardware

### ⚠️ Important Limitations

This is a **learning tool**, not a production AI system:

**Model Limitations (DistilGPT-2):**
- ✋ Small model (82M parameters) - limited knowledge and reasoning
- 📅 Training data cutoff (older knowledge)
- 🎲 Responses may be inaccurate, nonsensical, or repetitive
- 🚫 Not suitable for critical applications or factual queries
- 🎨 Best for: learning, experimentation, and creative text generation

**Why These Limitations?**
- 💻 Designed to run on **affordable hardware** (2-4GB RAM)
- 🎓 Focused on **learning AI integration**, not AI quality
- ⚡ Prioritizes **accessibility** over performance
- 🌱 Perfect starting point for understanding LLM deployment

### 🔮 Future Improvements

This project is an **ongoing learning journey**:
- 📈 Better model configurations and parameters
- 🎯 Improved response quality techniques
- 📚 Support for larger models (optional)

## 🔗 Companion Project: AI Service Gateway

**This LLM Backend powers the AI Service Gateway!**

### 🤝 How They Work Together

This **LLM Backend** is the **internal AI engine** that:
- 🧠 Runs the DistilGPT-2 language model
- 🔒 Restricted to internal networks only (security)
- ⚡ Processes all text generation requests
- 🎯 Optimized for low-spec hardware

The **[AI Service Gateway](https://github.com/Ammar0144/ai)** is the **public-facing API** that:
- ✅ Provides a clean REST API for clients
- ✅ Handles rate limiting and CORS
- ✅ Routes requests to this LLM service
- ✅ Manages authentication and security

### 🏗️ Architecture Overview

```
┌─────────────┐         ┌──────────────┐         ┌──────────────┐
│   Clients   │────────▶│ AI Gateway   │────────▶│ LLM Backend  │
│  (Public)   │  HTTP   │   (Go)       │  HTTP   │ (This Repo)  │
│             │         │ Port 8081    │         │  Port 8082   │
└─────────────┘         │              │         │              │
                        │ - Rate limit │         │ - DistilGPT-2│
                        │ - CORS       │         │ - FastAPI    │
                        │ - Swagger    │         │ - IP access  │
                        │ - Public     │         │ - Internal   │
                        └──────────────┘         └──────────────┘
```

### 💡 Why This Architecture?

**Learn Professional Patterns:**
- 🏢 **Microservices**: Separation of concerns (API gateway + ML service)
- 🔒 **Security Layers**: Public-facing Go service protects internal Python ML service
- 🌐 **Language Optimization**: Go for high-performance API, Python for ML
- 📡 **Service Communication**: HTTP-based inter-service communication
- 🎯 **Scalability**: Each service can scale independently

### 🔐 Why Keep LLM Backend Internal?

**Security Best Practices:**
- 🛡️ **Attack Surface**: Only the Go gateway is exposed to internet
- 🚫 **Direct Access**: ML service can't be reached directly
- 🔒 **IP Whitelisting**: Only trusted services can call LLM backend
- ⚖️ **Rate Limiting**: Gateway controls request flow
- 📊 **Monitoring**: Centralized logging and metrics at gateway

### 🚀 Try the Complete System!

**Explore the AI Gateway** to learn:
- 🔵 Go microservices development
- ⚡ Advanced rate limiting (token bucket)
- 📖 Swagger/OpenAPI documentation
- 🔧 Middleware patterns
- 🚀 Production deployment strategies

**👉 Check it out**: [github.com/Ammar0144/ai](https://github.com/Ammar0144/ai)

### 📦 Quick Start with Both Services

**Option 1: Docker Compose (Recommended)**
```bash
# Clone the AI gateway repository
git clone https://github.com/Ammar0144/ai.git
cd ai/

# This will start both services automatically
docker-compose up -d

# AI Gateway: http://localhost:8081
# LLM Backend: http://localhost:8082 (internal only)
```

**Option 2: Run Separately**
```bash
# Terminal 1 - Start LLM Backend (this repo)
cd llm/
python server.py
# Runs on http://localhost:8082

# Terminal 2 - Start AI Gateway
cd ai/
go run main.go
# Runs on http://localhost:8081
```

**Option 3: Docker Individual Services**
```bash
# Terminal 1 - LLM Backend
cd llm/
docker build -t llm-server .
docker run -p 8082:8082 llm-server

# Terminal 2 - AI Gateway
cd ai/
docker build -t ai-service .
docker run -p 8081:8081 -e LLM_SERVICE_URL=http://host.docker.internal:8082 ai-service
```

### 🎓 Learn the Full Stack!

**Together, these projects teach you:**

| Concept | AI Gateway (Go) | LLM Backend (Python) |
|---------|----------------|----------------------|
| **API Design** | RESTful with Swagger | FastAPI with auto-docs |
| **Security** | Rate limiting, CORS | IP access control |
| **Language** | Go 1.21+ | Python 3.11+ |
| **Framework** | net/http, Gin-like | FastAPI, Pydantic |
| **Role** | Public gateway | Internal processor |
| **Deployment** | Multi-platform binary | Docker container |
| **Testing** | Go testing, httptest | Pytest, FastAPI testclient |
| **Docs** | Swagger/OpenAPI | Auto-generated OpenAPI |

**See how both sides work together to create a complete AI service!** 🎯
- 🔧 Performance optimizations
- 📖 More comprehensive tutorials
- 🤝 Community-contributed enhancements
- 🌟 Best practices as we learn them

**We're learning together!** As we discover better approaches, this project evolves.

### 🤝 Community & Contributions

**We Need Your Help!**

Whether you're just starting or experienced with AI:
- ⭐ **Star** this repo if you find it useful for learning
- 🐛 **Report issues** - every bug report helps others learn
- 💡 **Suggest improvements** - share what you've learned
- 🤲 **Contribute code** - add features, fix bugs, improve docs
- 💬 **Share experiences** - what worked? what didn't?
- 📣 **Spread the word** - help others discover this learning resource
- 📖 **Improve docs** - help make AI more accessible

**Your feedback makes this better for everyone!** Even if you're learning, your perspective as a beginner is incredibly valuable.

### 🎓 Perfect For
- Students learning AI and ML
- Developers exploring LLM integration
- Anyone wanting to understand AI APIs
- Teams needing affordable AI for development/testing
- Educators teaching AI concepts
- Hobbyists experimenting with AI

## 💻 Software Engineering Concepts

This project demonstrates **real-world software engineering practices** you can learn from:

### 🎯 Key Concepts Implemented
- **🔒 Security**: IP-based access control, network segmentation, middleware patterns
- **⚡ API Design**: FastAPI best practices, Pydantic models, async endpoints
- **🐳 Containerization**: Docker optimization for low-resource environments
- **🚀 CI/CD**: Automated testing, deployment pipelines, artifact management
- **📊 Resource Management**: Running ML models on 2-4GB RAM, memory optimization
- **🔍 Logging**: Structured logging, request tracking, error monitoring
- **🏥 Health Checks**: Service monitoring, container health, graceful startup
- **📖 API Documentation**: Auto-generated docs, interactive testing (FastAPI)
- **🔗 Microservices**: Backend service isolation, internal-only endpoints
- **⚙️ Middleware**: Request filtering, access control, logging layers

### 📚 Comprehensive Learning Guide

Want to understand everything in detail? Check out our guide:
Want to dive deeper into the software engineering concepts?

**[📖 Complete Software Engineering Guide](SOFTWARE_ENGINEERING_CONCEPTS.md)**

This guide covers:
- ✅ In-depth explanations with code examples
- ✅ Why these patterns matter
- ✅ How to implement them yourself
- ✅ Learning exercises for each concept
- ✅ Beginner to advanced learning paths
- ✅ Real-world applications

### 🎓 What You'll Learn Here

**API Development**: FastAPI patterns, Pydantic validation, async/await  
**Security**: Middleware implementation, IP filtering, network isolation  
**ML Deployment**: Model loading, inference optimization, resource constraints  
**DevOps**: Docker for ML, CI/CD for Python, automated deployment  
**Production Thinking**: Health checks, logging, error handling, monitoring

**This entire codebase is a learning resource!** Every file demonstrates production patterns.

## 🚀 Features

### Core Capabilities
- **🧠 Lightweight Model**: DistilGPT-2 (82M parameters) for fast inference
- **⚡ FastAPI Framework**: High-performance REST API with automatic documentation
- **🎛️ Configurable Generation**: Adjustable parameters (temperature, top_p, max_length)
- **📝 Multiple Interfaces**: Text generation, completion, and chat endpoints

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

## 🤝 Contributing

**Everyone is Welcome!** This is a learning project - contributions from developers at all skill levels are valued and appreciated.

### How You Can Help

#### 🐛 Report Issues
- Found unexpected behavior? Report it!
- Documentation confusing? Let us know!
- Questions are contributions too - they help improve docs

#### 💡 Suggest Improvements
- Share ideas for better performance
- Propose new features for learners
- Suggest better ways to explain concepts
- Share what confused you (helps others!)

#### 📖 Improve Documentation
- Fix typos and unclear sections
- Add examples that helped you understand
- Create tutorials or guides
- Translate to other languages

#### 💻 Code Contributions
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Add tests** for new functionality
4. **Ensure security features** remain intact
5. **Test your changes** thoroughly
6. **Commit with clear messages**: `git commit -am 'Add: description of change'`
7. **Submit a pull request** with:
   - Clear description of changes
   - Why the change is helpful for learners
   - Any testing you've done

#### 🎓 Share Knowledge
- Write about your experience using this
- Create video tutorials
- Share tips and tricks you discovered
- Help answer questions from other learners
- Share your projects built with this

#### 🔒 Security
For security vulnerabilities, please email privately rather than opening public issues.

### 💬 Community
- ⭐ **Star** if this helped your learning
- 👀 **Watch** to stay updated
- 🍴 **Fork** and experiment
- 💬 **Discuss** ideas and experiences
- 📣 **Share** with others learning AI

**Your Experience Matters!** As a learner, you have unique insights into what works and what doesn't. Share them!

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
