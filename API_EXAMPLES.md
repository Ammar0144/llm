# DistilGPT-2 LLM Server API Examples

## Starting the Server

```bash
# Method 1: Direct Python
python server.py

# Method 2: Using setup script
./setup.sh

# Method 3: Using Docker
docker-compose up

# Method 4: Using FastAPI directly
uvicorn app:app --host 0.0.0.0 --port 8082
```

## API Usage Examples

### Health Check
```bash
curl http://localhost:8082/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "distilgpt2"
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
  "description": "DistilGPT-2 is a distilled version of GPT-2 that is smaller and faster while maintaining good performance"
}
```

### Text Generation
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

Response:
```json
{
  "generated_text": "The future of artificial intelligence is bright and full of possibilities. As technology continues to advance, we can expect to see more sophisticated AI systems that can help solve complex problems and improve our daily lives.",
  "prompt": "The future of artificial intelligence is"
}
```

### Simple Text Generation
```bash
curl -X POST "http://localhost:8082/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Once upon a time"}'
```

### Creative Writing Example
```bash
curl -X POST "http://localhost:8082/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "In a world where robots and humans coexist",
       "max_length": 150,
       "temperature": 0.8,
       "top_p": 0.95
     }'
```

### Technical Content Example
```bash
curl -X POST "http://localhost:8082/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Machine learning is a subset of artificial intelligence that",
       "max_length": 80,
       "temperature": 0.5,
       "top_p": 0.9
     }'
```

## Python Client Example

```python
import requests
import json

def generate_text(prompt, max_length=100, temperature=0.7):
    url = "http://localhost:8082/generate"
    payload = {
        "prompt": prompt,
        "max_length": max_length,
        "temperature": temperature,
        "top_p": 0.9,
        "do_sample": True
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

# Usage
result = generate_text("The benefits of renewable energy include")
print(result["generated_text"])
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