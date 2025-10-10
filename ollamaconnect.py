
# Enhanced FastAPI with Ollama Streaming Support
import uvicorn
import requests
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

app = FastAPI(title="AI Text Generator with Streaming", description="Generate AI responses using Ollama with streaming support")

# Request/Response models
class PromptRequest(BaseModel):
    prompt: str
    model: str = "mistral:instruct"
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = True

class AIResponse(BaseModel):
    response: str
    model: str
    prompt: str

# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"

@app.get("/")
async def root():
    return {"message": "AI Text Generator API is running with streaming support"}

def parse_streaming_response(response):
    """
    Generator function to parse streaming JSON responses from Ollama
    """
    for line in response.iter_lines():
        if line:
            try:
                # Decode the line and parse JSON
                decoded_line = line.decode('utf-8')
                json_data = json.loads(decoded_line)
                
                # Extract the response content
                content = json_data.get("response", "")
                done = json_data.get("done", False)
                
                # Yield the content chunk
                if content:
                    yield content
                
                # Break if done
                if done:
                    break
                    
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # Skip invalid lines
                continue

@app.post("/generate")
async def generate_text(request: PromptRequest):
    """
    Generate AI text using Ollama's local API with optional streaming
    """
    try:
        # Prepare the request payload for Ollama
        ollama_payload = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": request.stream,  # Enable/disable streaming
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens
            }
        }
        
        # Make request to Ollama API
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=ollama_payload,
            stream=request.stream,  # Important: enable streaming in requests
            timeout=60
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        if request.stream:
            # Return streaming response
            def generate_stream():
                full_response = ""
                for chunk in parse_streaming_response(response):
                    full_response += chunk
                    # Format each chunk as JSON for the client
                    chunk_data = {
                        "content": chunk,
                        "done": False,
                        "model": request.model
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # Send final message
                final_data = {
                    "content": "",
                    "done": True,
                    "model": request.model,
                    "full_response": full_response
                }
                yield f"data: {json.dumps(final_data)}\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            # Return complete response
            ollama_response = response.json()
            return AIResponse(
                response=ollama_response.get("response", ""),
                model=request.model,
                prompt=request.prompt
            )
        
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503, 
            detail="Cannot connect to Ollama. Make sure Ollama is running on localhost:11434"
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=408, 
            detail="Request to Ollama timed out"
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error communicating with Ollama: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )
'''
@app.get("/health")
async def health_check():
    """
    Check if Ollama is running and accessible
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        return {"status": "healthy", "ollama": "connected"}
    except:
        return {"status": "unhealthy", "ollama": "disconnected"}
'''
# Email-specific endpoint with streaming support
@app.post("/generate-email-reply")
async def generate_email_reply(
    subject: str, 
    body: str, 
    model: str = "mistral:instruct",
    stream: bool = True
):
    """
    Generate a professional email reply with streaming support
    """
    prompt = (
        "You are an AI email assistant. Compose a professional reply.\n\n"
        f"Subject: {subject}\n\n"
        f"Body:\n{body}\n\n"
        "Reply:"
    )
    
    request = PromptRequest(prompt=prompt, model=model, stream=stream)
    return await generate_text(request)

# Simple streaming endpoint for testing
@app.get("/stream-test")
async def stream_test():
    """
    Simple streaming test endpoint
    """
    def generate_test_stream():
        for i in range(10):
            data = {
                "message": f"Stream chunk {i+1}",
                "done": i == 9
            }
            yield f"data: {json.dumps(data)}\n\n"
            # Simulate processing time
            import time
            time.sleep(1)
    
    return StreamingResponse(
        generate_test_stream(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)