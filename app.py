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

class QuestionRequest(BaseModel):
    question: str
    max_length: Optional[int] = 150

class QuestionResponse(BaseModel):
    question: str
    answer: str
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

class ClassificationRequest(BaseModel):
    text: str
    labels: List[str]

class ClassificationResponse(BaseModel):
    text: str
    prediction: str
    confidence: Optional[float] = None

class SummarizationRequest(BaseModel):
    text: str
    max_length: Optional[int] = 100

class SummarizationResponse(BaseModel):
    summary: str
    original_text: str

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: Optional[float] = None

class EmbeddingsRequest(BaseModel):
    text: str

class EmbeddingsResponse(BaseModel):
    embeddings: str
    text: str
    method: str = "llm_generated"

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

def clean_response(raw_text: str, original_prompt: str) -> str:
    """Clean and extract the answer from the LLM response"""
    # Remove the original prompt from the response
    if raw_text.startswith(original_prompt):
        answer = raw_text[len(original_prompt):].strip()
    else:
        answer = raw_text.strip()
    
    # Split by sentences and clean up
    import re
    
    # Remove URLs, email addresses, and other noise
    answer = re.sub(r'http[s]?://\S+|www\.\S+|\S+@\S+\.\S+', '', answer)
    answer = re.sub(r'[^\w\s.,!?-]', ' ', answer)
    
    # Split by sentences
    sentences = re.split(r'[.!?]+', answer)
    cleaned_sentences = []
    
    for sentence in sentences[:3]:  # Take first 3 sentences max
        sentence = sentence.strip()
        if len(sentence) > 10 and not sentence.startswith('Question'):
            # Check if sentence makes sense (has both nouns and verbs indicators)
            words = sentence.lower().split()
            if len(words) > 3:  # Minimum length check
                cleaned_sentences.append(sentence)
    
    if cleaned_sentences:
        result = '. '.join(cleaned_sentences)
        if not result.endswith('.'):
            result += '.'
        return result
    
    # Fallback: try to extract meaningful content
    lines = answer.split('\n')
    for line in lines:
        line = line.strip()
        if len(line) > 15 and 'http' not in line.lower():
            return line + ('.' if not line.endswith('.') else '')
    
    return "I apologize, but I couldn't generate a clear answer to that question."

def format_question_prompt(question: str) -> str:
    """Format the user question into an effective prompt for the LLM"""
    # Add more context and examples to guide the model
    if "what is" in question.lower():
        prompt = f"{question}\n\n{question.replace('What is', '').replace('what is', '').strip()} is"
    elif "how does" in question.lower() or "how do" in question.lower():
        prompt = f"{question}\n\nTo understand this, "
    elif "why" in question.lower():
        prompt = f"{question}\n\nThe reason is that "
    else:
        # Default format with better context
        prompt = f"Please explain: {question}\n\nExplanation: "
    
    return prompt

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

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Intelligent question-answering endpoint with response cleaning"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format the question into an effective prompt
        formatted_prompt = format_question_prompt(request.question)
        
        # Tokenize input with proper attention mask
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Generate response with optimized parameters for Q&A
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=request.max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode the raw response
        raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean and extract the answer
        clean_answer = clean_response(raw_response, formatted_prompt)
        
        return QuestionResponse(
            question=request.question,
            answer=clean_answer,
            raw_response=raw_response
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")

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
        "description": "DistilGPT-2 is a distilled version of GPT-2 that is smaller and faster while maintaining good performance"
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

@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    """Classify text into provided labels"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create classification prompt
        labels_str = ", ".join(request.labels)
        prompt = f"Classify the following text into one of these categories: {labels_str}\nText: {request.text}\nCategory:"
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=len(inputs['input_ids'][0]) + 20,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = response_text[len(prompt):].strip().split()[0].lower()
        
        # Validate prediction is in provided labels
        valid_prediction = None
        for label in request.labels:
            if label.lower() in prediction:
                valid_prediction = label.lower()
                break
        
        if not valid_prediction:
            # Fallback: use first label
            valid_prediction = request.labels[0].lower()
        
        return ClassificationResponse(
            text=request.text,
            prediction=valid_prediction
        )
        
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/summarize", response_model=SummarizationResponse)
async def summarize_text(request: SummarizationRequest):
    """Summarize the provided text"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prompt = f"Summarize the following text:\n{request.text}\nSummary:"
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=request.max_length,
                temperature=0.5,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = response_text[len(prompt):].strip()
        
        # Clean up summary
        if summary:
            # Take first sentence or two
            sentences = summary.split('.')
            clean_summary = '. '.join(sentences[:2]).strip()
            if clean_summary and not clean_summary.endswith('.'):
                clean_summary += '.'
        else:
            clean_summary = "Summary not available."
        
        return SummarizationResponse(
            summary=clean_summary,
            original_text=request.text
        )
        
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of the provided text"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prompt = f"Analyze the sentiment of the following text. Respond with only 'positive', 'negative', or 'neutral':\n{request.text}\nSentiment:"
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=len(inputs['input_ids'][0]) + 15,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sentiment_prediction = response_text[len(prompt):].strip().split()[0].lower()
        
        # Validate sentiment
        valid_sentiments = ['positive', 'negative', 'neutral']
        final_sentiment = 'neutral'  # default
        
        for sentiment in valid_sentiments:
            if sentiment in sentiment_prediction:
                final_sentiment = sentiment
                break
        
        return SentimentResponse(
            text=request.text,
            sentiment=final_sentiment
        )
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@app.post("/embeddings", response_model=EmbeddingsResponse)
async def generate_embeddings(request: EmbeddingsRequest):
    """Generate text embeddings using LLM-based approach"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Since we don't have a dedicated embedding model, we'll use the LLM to generate
        # a semantic representation of the text
        prompt = f"Generate a semantic representation and key concepts for the following text:\nText: {request.text}\nSemantic representation:"
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=150,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        embeddings_representation = response_text[len(prompt):].strip()
        
        # Clean up the embeddings representation
        if embeddings_representation:
            # Take the first meaningful sentence
            sentences = embeddings_representation.split('.')
            clean_repr = sentences[0].strip()
            if len(clean_repr) < 10:
                clean_repr = embeddings_representation[:100].strip()
        else:
            clean_repr = f"Semantic representation of: {request.text[:50]}..."
        
        return EmbeddingsResponse(
            embeddings=clean_repr,
            text=request.text,
            method="llm_generated"
        )
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Embeddings generation failed: {str(e)}")

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