# -*- coding: utf-8 -*-
"""
Comprehensive client example to test the streamlined DistilGPT-2 LLM Server
Optimized for DistilGPT-2's core strengths: generation, completion, and chat
"""

import requests
import json

def test_server():
    """Test all available LLM server endpoints"""
    base_url = "http://localhost:8082"
    
    print("DistilGPT-2 LLM Server Test Client")
    print("=" * 50)
    
    # Test health check
    print("1. Testing health check...")
    try:
        response = requests.get(base_url + "/health")
        if response.status_code == 200:
            health = response.json()
            print("   Status: " + health['status'])
            print("   Model: " + health['model_name'])
            security_enabled = health.get('security', {}).get('access_control_enabled', False)
            print("   Security: " + ("Enabled" if security_enabled else "Disabled"))
        else:
            print("   Health check failed: " + str(response.status_code))
            return
    except Exception as e:
        print("   Connection error: " + str(e))
        return
    
    # Test model info
    print("\n2. Getting model info...")
    try:
        response = requests.get(base_url + "/model-info")
        if response.status_code == 200:
            info = response.json()
            print("   Model: " + info['model_name'] + " (" + info['model_size'] + ")")
            optimized_for = info.get('optimized_for', [])
            print("   Optimized for: " + ', '.join(optimized_for))
        else:
            print("   Model info unavailable: " + str(response.status_code))
    except Exception as e:
        print("   Error getting model info: " + str(e))
    
    # Test text completion (PRIMARY STRENGTH)
    print("\n3️⃣ Testing text completion (⭐ PRIMARY STRENGTH)...")
    completion_prompt = "The benefits of artificial intelligence include"
    
    payload = {
        "prompt": completion_prompt,
        "max_tokens": 80,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(f"{base_url}/complete", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"   📝 Prompt: {result['prompt']}")
            print(f"   ✨ Completion: {result['completion']}")
        else:
            print(f"   ❌ Completion failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ❌ Error in completion: {e}")
    
    # Test text generation (PRIMARY STRENGTH)
    print("\n4️⃣ Testing text generation (⭐ PRIMARY STRENGTH)...")
    generation_prompt = "Once upon a time in a digital world"
    
    payload = {
        "prompt": generation_prompt,
        "max_length": 100,
        "temperature": 0.8,
        "top_p": 0.9,
        "do_sample": True
    }
    
    try:
        response = requests.post(f"{base_url}/generate", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"   📝 Prompt: {result['prompt']}")
            print(f"   📖 Generated: {result['generated_text']}")
        else:
            print(f"   ❌ Generation failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ❌ Error in generation: {e}")
    
    # Test chat completions (GOOD PERFORMANCE)
    print("\n5️⃣ Testing chat completions (OpenAI compatible)...")
    chat_payload = {
        "messages": [
            {"role": "user", "content": "What is machine learning in simple terms?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(f"{base_url}/chat/completions", json=chat_payload)
        if response.status_code == 200:
            result = response.json()
            print(f"   👤 User: What is machine learning in simple terms?")
            print(f"   🤖 Assistant: {result['content']}")
        else:
            print(f"   ❌ Chat completion failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ❌ Error in chat completion: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Testing complete! All endpoints optimized for DistilGPT-2's strengths.")

def test_creative_examples():
    """Test creative use cases"""
    base_url = "http://localhost:8082"
    
    print("\n🎨 Creative Use Case Examples")
    print("=" * 30)
    
    examples = [
        {
            "name": "Story Completion",
            "endpoint": "/complete",
            "payload": {
                "prompt": "The scientist made an incredible discovery when she looked through the microscope and saw",
                "max_tokens": 60,
                "temperature": 0.8
            }
        },
        {
            "name": "Technical Explanation", 
            "endpoint": "/complete",
            "payload": {
                "prompt": "Quantum computing differs from classical computing because",
                "max_tokens": 80,
                "temperature": 0.5
            }
        },
        {
            "name": "Creative Writing",
            "endpoint": "/generate", 
            "payload": {
                "prompt": "In the year 2075, humanity discovered",
                "max_length": 120,
                "temperature": 0.9
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}:")
        try:
            response = requests.post(f"{base_url}{example['endpoint']}", json=example['payload'])
            if response.status_code == 200:
                result = response.json()
                if 'completion' in result:
                    print(f"   Result: {result['completion']}")
                else:
                    print(f"   Result: {result['generated_text']}")
            else:
                print(f"   ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_server()
    
    # Ask if user wants to see creative examples
    print("\n🤔 Would you like to see creative use case examples? (y/n): ", end="")
    try:
        if input().lower() == 'y':
            test_creative_examples()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except:
        pass