"""
Simple client example to test the DistilGPT-2 LLM Server
"""

import requests
import json

def test_server():
    """Test the LLM server endpoints"""
    base_url = "http://localhost:8082"
    
    # Test health check
    print("Testing health check...")
    response = requests.get(f"{base_url}/health")
    print(f"Health check: {response.json()}")
    
    # Test model info
    print("\nGetting model info...")
    response = requests.get(f"{base_url}/model-info")
    print(f"Model info: {response.json()}")
    
    # Test text generation
    print("\nTesting text generation...")
    prompt = "The future of artificial intelligence is"
    
    payload = {
        "prompt": prompt,
        "max_length": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }
    
    response = requests.post(f"{base_url}/generate", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prompt: {result['prompt']}")
        print(f"Generated text: {result['generated_text']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_server()