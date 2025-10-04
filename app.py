"""
LLM Server using DistilGPT-2 model with FastAPI
Provides endpoints for text generation using a lightweight GPT-2 model.
"""

from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import os
from typing import Optional, List, Dict, Any
import ipaddress

# to start deployment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DistilGPT-2 LLM Server",
    description="A lightweight LLM server using DistilGPT-2 for text generation",
    version="1.0.0"
)

# Access control configuration
ALLOWED_IPS = [
    "127.0.0.1",          # localhost
    "::1",                # localhost IPv6
    "172.17.0.0/16",      # Docker default bridge network
    "172.18.0.0/16",      # Docker custom networks
    "172.19.0.0/16",      # Docker custom networks
    "172.20.0.0/16",      # Docker custom networks
    "172.65.0.0/16",      # Docker Desktop network range
    "10.0.0.0/8",         # Private network range
    "192.168.0.0/16"      # Private network range
]

class IPAccessMiddleware(BaseHTTPMiddleware):
    """Middleware to restrict access to allowed IP addresses only"""
    
    def __init__(self, app, allowed_ips: List[str]):
        super().__init__(app)
        self.allowed_networks = []
        
        for ip in allowed_ips:
            try:
                if "/" in ip:  # CIDR notation
                    self.allowed_networks.append(ipaddress.ip_network(ip, strict=False))
                else:  # Single IP
                    self.allowed_networks.append(ipaddress.ip_network(f"{ip}/32" if ":" not in ip else f"{ip}/128", strict=False))
            except ValueError as e:
                logger.warning(f"Invalid IP/network configuration: {ip} - {e}")
    
    def get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers"""
        # Check X-Forwarded-For (for reverse proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            client_ip = forwarded_for.split(",")[0].strip()
            return client_ip
        
        # Check X-Real-IP (for nginx)
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fallback to direct connection
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def is_ip_allowed(self, client_ip: str) -> bool:
        """Check if client IP is in allowed networks"""
        try:
            client_addr = ipaddress.ip_address(client_ip)
            for network in self.allowed_networks:
                if client_addr in network:
                    return True
            return False
        except ValueError:
            logger.warning(f"Invalid client IP format: {client_ip}")
            return False
    
    async def dispatch(self, request: Request, call_next):
        # Skip access control for health check (allow external monitoring)
        if request.url.path == "/health":
            return await call_next(request)
        
        # Skip access control if disabled via environment variable
        if os.getenv("DISABLE_IP_RESTRICTION", "false").lower() == "true":
            return await call_next(request)
        
        client_ip = self.get_client_ip(request)
        
        if not self.is_ip_allowed(client_ip):
            logger.warning(f"Access denied for IP: {client_ip} to {request.url.path}")
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Access Forbidden",
                    "message": "This LLM service is restricted to internal AI service access only",
                    "client_ip": client_ip,
                    "allowed_networks": [str(net) for net in self.allowed_networks]
                }
            )
        
        logger.info(f"Access granted for IP: {client_ip} to {request.url.path}")
        return await call_next(request)

# Add access control middleware
app.add_middleware(IPAccessMiddleware, allowed_ips=ALLOWED_IPS)

# Global variables for model and tokenizer
model = None
tokenizer = None

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = True
    repetition_penalty: Optional[float] = 1.2
    no_repeat_ngram_size: Optional[int] = 2

class TextGenerationResponse(BaseModel):
    generated_text: str
    prompt: str

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7

class CompletionResponse(BaseModel):
    prompt: str
    completion: str
    raw_response: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7

class ChatCompletionResponse(BaseModel):
    content: str
    role: str = "assistant"



def load_model():
    """Load the DistilGPT-2 model and tokenizer"""
    global model, tokenizer
    
    try:
        logger.info("Loading DistilGPT-2 model and tokenizer...")
        model_name = "distilgpt2"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Model and tokenizer loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e





@app.on_event("startup")
async def startup_event():
    """Initialize the model when the server starts"""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "DistilGPT-2 LLM Server is running", "status": "healthy"}

@app.get("/health")
async def health_check(request: Request):
    """Detailed health check endpoint with security status"""
    model_loaded = model is not None and tokenizer is not None
    
    # Get client IP for security status
    client_ip = "unknown"
    if request.client:
        client_ip = request.client.host
    
    # Check if access control is enabled
    access_control_enabled = os.getenv("DISABLE_IP_RESTRICTION", "false").lower() != "true"
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_name": "distilgpt2",
        "security": {
            "access_control_enabled": access_control_enabled,
            "client_ip": client_ip,
            "allowed_networks": ALLOWED_IPS if access_control_enabled else "disabled"
        },
        "service": "LLM Server - Internal Use Only"
    }

@app.post("/complete", response_model=CompletionResponse)
async def complete_text(request: CompletionRequest):
    """Text completion endpoint - DistilGPT-2's core strength"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Use the user's prompt directly - this is what DistilGPT-2 does best
        prompt = request.prompt.strip()
        
        # Tokenize input with proper attention mask
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Generate completion using user-specified parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                early_stopping=True
            )
        
        # Decode the raw response
        raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the completion part (remove the original prompt)
        completion = raw_response[len(prompt):].strip()
        
        return CompletionResponse(
            prompt=request.prompt,
            completion=completion,
            raw_response=raw_response
        )
        
    except Exception as e:
        logger.error(f"Error processing completion: {e}")
        raise HTTPException(status_code=500, detail=f"Text completion failed: {str(e)}")

@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """Generate text using DistilGPT-2 model"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input with proper attention mask
        inputs = tokenizer(
            request.prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        # Move to CPU explicitly to avoid device issues
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                repetition_penalty=request.repetition_penalty,
                no_repeat_ngram_size=request.no_repeat_ngram_size,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return TextGenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt
        )
        
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "distilgpt2",
        "model_type": "GPT-2",
        "model_size": "82M parameters",
        "description": "DistilGPT-2 optimized for text generation and completion tasks",
        "supported_endpoints": [
            "/generate - Text generation (primary strength)",
            "/complete - Text completion (primary strength)", 
            "/chat/completions - Chat-style conversations",
            "/health - Service health check",
            "/ - Basic status"
        ],
        "optimized_for": ["text_generation", "text_completion", "chat_conversations"],
        "note": "This API is streamlined to focus on DistilGPT-2's core capabilities"
    }

@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-style chat completions endpoint"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert chat messages to a prompt
        prompt = ""
        for message in request.messages:
            if message.role == "system":
                prompt += f"System: {message.content}\n"
            elif message.role == "user":
                prompt += f"User: {message.content}\n"
            elif message.role == "assistant":
                prompt += f"Assistant: {message.content}\n"
        
        prompt += "Assistant:"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=request.max_tokens,
                temperature=request.temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "Assistant:" in response_text:
            assistant_response = response_text.split("Assistant:")[-1].strip()
        else:
            assistant_response = response_text[len(prompt):].strip()
        
        return ChatCompletionResponse(content=assistant_response)
        
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Log security configuration on startup
    access_control_enabled = os.getenv("DISABLE_IP_RESTRICTION", "false").lower() != "true"
    
    logger.info("=" * 60)
    logger.info("🤖 DistilGPT-2 LLM Server Starting")
    logger.info("=" * 60)
    logger.info(f"🔒 Access Control: {'ENABLED' if access_control_enabled else 'DISABLED'}")
    
    if access_control_enabled:
        logger.info("🛡️  Allowed Networks:")
        for network in ALLOWED_IPS:
            logger.info(f"   • {network}")
        logger.info("ℹ️  Only AI service and internal networks can access this server")
        logger.info("ℹ️  Health endpoint (/health) is accessible externally for monitoring")
    else:
        logger.warning("⚠️  IP restrictions are DISABLED - all IPs can access this service")
        logger.warning("ℹ️  Set DISABLE_IP_RESTRICTION=false to enable security")
    
    logger.info("🌐 Server will start on: http://0.0.0.0:8082")
    logger.info("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8082)