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
    """
    Clean and extract the completion from DistilGPT-2 response.
    DistilGPT-2 generates text completions, so we need to extract the
    completed text that makes sense as an answer.
    """
    # Remove the original prompt from the response
    if raw_text.startswith(original_prompt):
        completion = raw_text[len(original_prompt):].strip()
    else:
        completion = raw_text.strip()
    
    if not completion:
        return "Unable to generate a response."
    
    import re
    
    # Clean up basic formatting issues
    completion = re.sub(r'\s+', ' ', completion)  # Normalize whitespace
    completion = completion.strip()
    
    # For DistilGPT-2, take the first logical stopping point
    # Look for natural sentence endings within reasonable length
    sentences = re.split(r'([.!?]+)', completion)
    
    result_parts = []
    current_length = 0
    
    i = 0
    while i < len(sentences) and current_length < 100:  # Keep responses concise
        part = sentences[i].strip()
        
        if part and not part in '.!?':
            # This is actual content, not punctuation
            words = part.split()
            
            # Skip very short fragments or obvious continuations
            if len(words) >= 3:
                result_parts.append(part)
                current_length += len(part)
                
                # If we have punctuation after this, add it
                if i + 1 < len(sentences) and sentences[i + 1] in '.!?':
                    result_parts.append(sentences[i + 1])
                    break  # Stop at first complete sentence for clarity
        i += 1
    
    if result_parts:
        result = ''.join(result_parts).strip()
        
        # Ensure proper ending
        if result and not result[-1] in '.!?':
            result += '.'
            
        # Remove any trailing incomplete thoughts
        result = re.sub(r'\s+and\s*$', '.', result)
        result = re.sub(r'\s+but\s*$', '.', result)
        result = re.sub(r'\s+or\s*$', '.', result)
        
        return result
    
    # If no good sentences found, take the first reasonable chunk
    words = completion.split()
    if len(words) >= 5:
        # Take first 15 words and add period
        return ' '.join(words[:15]) + '.'
    
    return "Unable to generate a clear response."

def format_question_prompt(question: str) -> str:
    """
    Format questions as text completion prompts for DistilGPT-2.
    DistilGPT-2 is a text generation model, not a Q&A model, so we need
    to format questions as text that it can naturally complete.
    """
    question_lower = question.lower().strip()
    
    # Remove question marks to make it more completion-friendly
    clean_question = question.rstrip('?').strip()
    
    # Format based on question type for better completion
    if question_lower.startswith(("what is", "what are")):
        # Convert "What is X?" to "X is a"
        topic = clean_question[7:].strip() if question_lower.startswith("what is") else clean_question[8:].strip()
        if topic:
            prompt = f"{topic.title()} is a"
        else:
            prompt = f"The answer to '{clean_question}' is that"
    
    elif question_lower.startswith(("how does", "how do", "how to")):
        # Convert "How does X work?" to "X works by"
        if "how does" in question_lower:
            topic = clean_question[8:].strip()
            prompt = f"To understand {topic}, it works by"
        else:
            prompt = f"The process involves"
    
    elif question_lower.startswith(("why", "why is", "why do", "why does")):
        # Convert "Why is X?" to "The reason X is because"
        topic = clean_question[3:].strip() if question_lower.startswith("why ") else clean_question
        prompt = f"The reason this happens is because"
    
    elif question_lower.startswith(("where", "when", "who")):
        # Convert to completion format
        prompt = f"The answer is"
    
    elif "capital" in question_lower and "france" in question_lower:
        # Specific case for capital questions
        prompt = "The capital of France is"
    
    elif "benefits" in question_lower:
        prompt = f"The main benefits include"
    
    else:
        # Generic format - create a context that encourages factual completion
        prompt = f"Here's what you need to know about {clean_question.lower()}: It"
    
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
        
        # Generate response with parameters optimized for factual completion
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,  # Focus on completing the thought, not generating long text
                temperature=0.2,    # Lower temperature for more factual, focused responses
                top_p=0.8,         # Slightly more focused sampling
                do_sample=True,
                repetition_penalty=1.1,  # Lighter penalty to allow natural repetition
                no_repeat_ngram_size=2,  # Allow some repetition for natural flow
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                early_stopping=True
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
    """Classify text into provided labels with improved LLM handling"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format as completion for DistilGPT-2
        labels_str = " or ".join(request.labels)
        prompt = f"The text '{request.text}' is classified as"
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=8,   # Just need a few tokens for classification
                temperature=0.1,    # Very low for consistent classification
                top_p=0.7,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        after_prompt = response_text[len(prompt):].strip()
        words = after_prompt.split()
        
        # Find matching label from model response
        best_match = None
        if words:
            prediction = words[0].lower()
            for label in request.labels:
                if label.lower() in prediction:
                    best_match = label.lower()
                    break
        
        # Use first label as fallback if model doesn't match any
        if not best_match:
            best_match = request.labels[0].lower()
        
        return ClassificationResponse(
            text=request.text,
            prediction=best_match
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
        # Format as completion - DistilGPT-2 works better with natural continuations
        prompt = f"In summary, the key points of '{request.text[:100]}...' are:"
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=50,  # Concise summaries
                temperature=0.3,    # Lower for more focused summaries
                top_p=0.8,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
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
    """Analyze sentiment using pure LLM model response"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format as completion task - DistilGPT-2 works better with completion prompts
        prompt = f"The sentiment of the text '{request.text}' is"
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=5,  # Just need a few tokens for sentiment
                temperature=0.1,   # Very low for consistent sentiment classification
                top_p=0.7,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract model's response
        after_prompt = response_text[len(prompt):].strip()
        words = after_prompt.split()
        
        # Use model's first word as sentiment, validate against expected values
        if not words:
            sentiment_prediction = "neutral"  # Only if model produces no output
        else:
            sentiment_prediction = words[0].lower()
        
        # Ensure response is one of valid sentiments
        valid_sentiments = ['positive', 'negative', 'neutral']
        final_sentiment = 'neutral'  # default only if invalid
        
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