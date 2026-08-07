import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List

class LLMClient:
    """
    Base client interface for interacting with LLM backends.
    """
    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement generate().")


def list_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """
    Fetches the list of locally available models from an active Ollama instance.
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as response:
            resp_data = json.loads(response.read().decode('utf-8'))
            models = resp_data.get("models", [])
            return [m.get("name", "") for m in models if m.get("name")]
    except Exception:
        return []


class OllamaClient(LLMClient):
    """
    Client for local Ollama instances.
    """
    def __init__(self, model_name: str = "mistral:latest", base_url: str = "http://localhost:11434", temperature: float = 0.0):
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
            with urllib.request.urlopen(req, timeout=120) as response:
                resp_data = json.loads(response.read().decode('utf-8'))
                return resp_data.get("response", "").strip()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            available = list_ollama_models(self.base_url)
            avail_str = f" Available local models: {available}" if available else ""
            if e.code == 404:
                raise RuntimeError(
                    f"Model '{self.model_name}' was not found on Ollama server at {self.base_url}.{avail_str} "
                    f"Please pull the model using `ollama pull {self.model_name}` or select an available model."
                )
            raise RuntimeError(f"Ollama server error ({e.code}): {error_body}")
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                f"Ensure Ollama is running (`ollama serve`). Error: {e.reason}"
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


class GroqClient(LLMClient):
    """
    Client for Groq Cloud API (ultra-fast cloud inference).
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None, temperature: float = 0.0):
        self.model_name = model_name
        if not api_key:
            try:
                import streamlit as st
                if "GROQ_API_KEY" in st.secrets:
                    api_key = st.secrets["GROQ_API_KEY"]
            except Exception:
                pass
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.temperature = temperature

        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set the GROQ_API_KEY environment variable or pass it directly."
            )

    def generate(self, prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
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
                raise RuntimeError("Invalid response structure from Groq API.")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            raise RuntimeError(f"Groq API error ({e.code}): {error_body}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Groq API: {e}")
        except Exception as e:
            raise RuntimeError(f"Groq generation failed: {e}")


class GeminiClient(LLMClient):
    """
    Client for Google Gemini REST API.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None, temperature: float = 0.0):
        self.model_name = model_name
        if not api_key:
            try:
                import streamlit as st
                if "GEMINI_API_KEY" in st.secrets:
                    api_key = st.secrets["GEMINI_API_KEY"]
                elif "GOOGLE_API_KEY" in st.secrets:
                    api_key = st.secrets["GOOGLE_API_KEY"]
            except Exception:
                pass
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.temperature = temperature

        if not self.api_key:
            raise ValueError(
                "Gemini/Google API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable or pass it directly."
            )

    def generate(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": self.temperature
            }
        }
        data = json.dumps(payload).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                resp_data = json.loads(response.read().decode('utf-8'))
                candidates = resp_data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
                raise RuntimeError("Invalid response structure from Gemini API.")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            raise RuntimeError(f"Gemini API error ({e.code}): {error_body}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Gemini API: {e}")
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}")


def get_llm_client(provider: str, model_name: str, **kwargs) -> LLMClient:
    """
    Factory function to retrieve the configured LLMClient.
    """
    provider = provider.lower().strip()
    if provider == "ollama":
        return OllamaClient(model_name=model_name, **kwargs)
    elif provider == "openai":
        return OpenAIClient(model_name=model_name, **kwargs)
    elif provider == "groq":
        return GroqClient(model_name=model_name, **kwargs)
    elif provider in ("gemini", "google"):
        return GeminiClient(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Choose 'ollama', 'openai', 'groq', or 'gemini'.")
