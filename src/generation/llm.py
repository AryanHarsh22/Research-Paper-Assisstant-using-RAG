import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

class LLMClient:
    """
    Base client interface for interacting with LLM backends.
    """
    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement generate().")


class OllamaClient(LLMClient):
    """
    Client for local Ollama instances.
    """
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434", temperature: float = 0.0):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                resp_data = json.loads(response.read().decode('utf-8'))
                return resp_data.get("response", "").strip()
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running and accessible. Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")


class OpenAIClient(LLMClient):
    """
    Client for OpenAI completions.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None, temperature: float = 0.0):
        self.model_name = model_name
        
        # Check explicit parameter -> st.secrets -> os.environ
        if not api_key:
            try:
                import streamlit as st
                if "OPENAI_API_KEY" in st.secrets:
                    api_key = st.secrets["OPENAI_API_KEY"]
            except Exception:
                pass

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable "
                "or pass it directly to the client."
            )

    def generate(self, prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature
        }
        
        data = json.dumps(payload).encode('utf-8')
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        req = urllib.request.Request(url, data=data, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                resp_data = json.loads(response.read().decode('utf-8'))
                choices = resp_data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
                raise RuntimeError("Invalid response structure from OpenAI API.")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            raise RuntimeError(f"OpenAI API error ({e.code}): {error_body}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to OpenAI API: {e}")
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")


def get_llm_client(provider: str, model_name: str, **kwargs) -> LLMClient:
    """
    Factory function to retrieve the configured LLMClient.
    """
    provider = provider.lower().strip()
    if provider == "ollama":
        return OllamaClient(model_name=model_name, **kwargs)
    elif provider == "openai":
        return OpenAIClient(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Choose 'ollama' or 'openai'.")
