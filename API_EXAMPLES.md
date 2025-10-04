# DistilGPT-2 Streamlined API Examples

> **Note**: This API has been optimized to focus on DistilGPT-2's core strengths: text generation, completion, and chat conversations.

## 🚀 Starting the Server

```bash
# Method 1: Direct Python
python server.py

# Method 2: Using setup script
./setup.sh

# Method 3: Using Docker (Recommended)
docker-compose up

# Method 4: Using FastAPI directly
uvicorn app:app --host 0.0.0.0 --port 8082
```

## 📋 Available Endpoints

| Endpoint | Purpose | Strength |
|----------|---------|----------|
| `GET /health` | Service health check | ✅ Always |
| `GET /model-info` | Model information | ✅ Always |
| `POST /generate` | Text generation | ⭐⭐⭐⭐⭐ Primary |
| `POST /complete` | Text completion | ⭐⭐⭐⭐⭐ Primary |
| `POST /chat/completions` | OpenAI-style chat | ⭐⭐⭐⭐ Good |

## 🔧 API Usage Examples

### Health Check
```bash
curl http://localhost:8082/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "distilgpt2",
  "security": {
    "access_control_enabled": true,
    "client_ip": "127.0.0.1"
  }
}
```

### Model Information
```bash
curl http://localhost:8082/model-info
```

Response:
```json
{
  "model_name": "distilgpt2",
  "model_type": "GPT-2", 
  "model_size": "82M parameters",
  "description": "DistilGPT-2 optimized for text generation and completion tasks",
  "supported_endpoints": [
    "/generate - Text generation (primary strength)",
    "/complete - Text completion (primary strength)", 
    "/chat/completions - Chat-style conversations"
  ],
  "optimized_for": ["text_generation", "text_completion", "chat_conversations"]
}
```

## ⭐ Primary Endpoints (Best Performance)

### 1. Text Generation
```bash
curl -X POST "http://localhost:8082/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "The future of artificial intelligence is",
       "max_length": 100,
       "temperature": 0.7,
       "top_p": 0.9,
       "do_sample": true
     }'
```

### 2. Text Completion (NEW - Optimized)
```bash
curl -X POST "http://localhost:8082/complete" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "The benefits of renewable energy include",
       "max_tokens": 100,
       "temperature": 0.7
     }'
```

Response:
```json
{
  "prompt": "The benefits of renewable energy include",
  "completion": "reduced carbon emissions, lower long-term energy costs, improved energy security, and job creation in green technology sectors.",
  "raw_response": "The benefits of renewable energy include reduced carbon emissions, lower long-term energy costs, improved energy security, and job creation in green technology sectors."
}
```

### 3. Chat Completions (OpenAI Compatible)
```bash
curl -X POST "http://localhost:8082/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [
         {"role": "user", "content": "Tell me about machine learning"}
       ],
       "max_tokens": 100,
       "temperature": 0.7
     }'
```

Response:
```json
{
  "content": "Machine learning is a branch of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed for every task.",
  "role": "assistant"
}
```

## 🎨 Use Case Examples

### Creative Writing
```bash
curl -X POST "http://localhost:8082/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "In a world where dragons still exist",
       "max_length": 150,
       "temperature": 0.8,
       "top_p": 0.95
     }'
```

### Technical Content
```bash
curl -X POST "http://localhost:8082/complete" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Neural networks work by",
       "max_tokens": 80,
       "temperature": 0.5
     }'
```

## 🐍 Python Client Examples

### Complete Client with All Endpoints
```python
import requests
import json

class DistilGPT2Client:
    def __init__(self, base_url="http://localhost:8082"):
        self.base_url = base_url
    
    def health_check(self):
        """Check if server is healthy"""
        response = requests.get(f"{self.base_url}/health")
        return response.json() if response.status_code == 200 else None
    
    def get_model_info(self):
        """Get model information"""
        response = requests.get(f"{self.base_url}/model-info")
        return response.json() if response.status_code == 200 else None
    
    def generate_text(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """Generate text using the /generate endpoint"""
        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True
        }
        response = requests.post(f"{self.base_url}/generate", json=payload)
        return response.json() if response.status_code == 200 else {"error": response.text}
    
    def complete_text(self, prompt, max_tokens=100, temperature=0.7):
        """Complete text using the /complete endpoint (optimized for DistilGPT-2)"""
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        response = requests.post(f"{self.base_url}/complete", json=payload)
        return response.json() if response.status_code == 200 else {"error": response.text}
    
    def chat_completion(self, messages, max_tokens=100, temperature=0.7):
        """Create chat completion using OpenAI-compatible endpoint"""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        response = requests.post(f"{self.base_url}/chat/completions", json=payload)
        return response.json() if response.status_code == 200 else {"error": response.text}

# Usage Examples
client = DistilGPT2Client()

# Test server health
health = client.health_check()
print(f"Server status: {health['status']}")

# Text completion (recommended for DistilGPT-2)
completion = client.complete_text("The future of AI is")
print(f"Completion: {completion['completion']}")

# Text generation
generation = client.generate_text("Once upon a time")
print(f"Generated: {generation['generated_text']}")

# Chat completion
chat_response = client.chat_completion([
    {"role": "user", "content": "What is machine learning?"}
])
print(f"Chat: {chat_response['content']}")
```

### Simple Usage Functions
```python
import requests

def complete_text(prompt, max_tokens=100, temperature=0.7):
    """Simple text completion function"""
    url = "http://localhost:8082/complete"
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = requests.post(url, json=payload)
    return response.json() if response.status_code == 200 else None

# Quick usage
result = complete_text("The benefits of renewable energy include")
if result:
    print(result["completion"])
```

## JavaScript/Node.js Client Example

```javascript
const axios = require('axios');

async function generateText(prompt, maxLength = 100, temperature = 0.7) {
    try {
        const response = await axios.post('http://localhost:8082/generate', {
            prompt: prompt,
            max_length: maxLength,
            temperature: temperature,
            top_p: 0.9,
            do_sample: true
        });
        
        return response.data;
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
        return null;
    }
}

// Usage
generateText("The evolution of technology has").then(result => {
    if (result) {
        console.log(result.generated_text);
    }
});
```

## WebUI Example (HTML + JavaScript)

```html
<!DOCTYPE html>
<html>
<head>
    <title>DistilGPT-2 Text Generator</title>
</head>
<body>
    <h1>DistilGPT-2 Text Generator</h1>
    
    <textarea id="prompt" placeholder="Enter your prompt here..." rows="4" cols="50"></textarea><br><br>
    
    <label>Max Length: <input type="number" id="maxLength" value="100" min="10" max="500"></label><br><br>
    <label>Temperature: <input type="number" id="temperature" value="0.7" min="0.1" max="2.0" step="0.1"></label><br><br>
    
    <button onclick="generateText()">Generate Text</button><br><br>
    
    <div id="result"></div>

    <script>
        async function generateText() {
            const prompt = document.getElementById('prompt').value;
            const maxLength = parseInt(document.getElementById('maxLength').value);
            const temperature = parseFloat(document.getElementById('temperature').value);
            
            if (!prompt.trim()) {
                alert('Please enter a prompt');
                return;
            }
            
            try {
                const response = await fetch('http://localhost:8082/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        max_length: maxLength,
                        temperature: temperature,
                        top_p: 0.9,
                        do_sample: true
                    })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    document.getElementById('result').innerHTML = `
                        <h3>Generated Text:</h3>
                        <p style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
                            ${result.generated_text}
                        </p>
                    `;
                } else {
                    document.getElementById('result').innerHTML = `<p style="color: red;">Error: ${result.detail}</p>`;
                }
            } catch (error) {
                document.getElementById('result').innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
            }
        }
    </script>
</body>
</html>
```

## Production Deployment

### Using Docker
```bash
# Build the image
docker build -t distilgpt2-server .

# Run the container
docker run -p 8082:8082 distilgpt2-server

# Using docker-compose
docker-compose up -d
```

### Using systemd (Linux)
Create `/etc/systemd/system/llm-server.service`:
```ini
[Unit]
Description=DistilGPT-2 LLM Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/llm
ExecStart=/path/to/venv/bin/python server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable llm-server
sudo systemctl start llm-server
```

## Configuration

Edit `config.json` to customize model and server settings:

```json
{
  "model": {
    "name": "distilgpt2",
    "max_length": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": true
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8082,
    "log_level": "info"
  },
  "generation": {
    "max_length_limit": 500,
    "temperature_range": [0.1, 2.0],
    "top_p_range": [0.1, 1.0]
  }
}
```