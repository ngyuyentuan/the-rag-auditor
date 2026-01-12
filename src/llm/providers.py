"""
LLM Provider Module - Multiple Backend Support

Supports:
1. Groq (FREE tier - Llama3, Mixtral)
2. OpenAI (GPT-3.5, GPT-4)
3. Ollama (Local - Llama3, Mistral)
4. Smart Template (Fallback - no API needed)
"""
import os
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger("llm_provider")


class LLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


class GroqProvider(LLMProvider):
    """Groq API - FREE tier with Llama3."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.1-8b-instant"  # Fast and free
    
    @property
    def name(self) -> str:
        return "Groq (Free)"
    
    async def generate(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        if not self.api_key:
            return None
        
        import httpx
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    logger.warning(f"Groq error: {response.status_code}")
        except Exception as e:
            logger.debug(f"Groq failed: {e}")
        return None


class OpenAIProvider(LLMProvider):
    """OpenAI API - GPT-3.5/4."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-3.5-turbo"
    
    @property
    def name(self) -> str:
        return "OpenAI"
    
    async def generate(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        if not self.api_key:
            return None
        
        import httpx
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.debug(f"OpenAI failed: {e}")
        return None


class OllamaProvider(LLMProvider):
    """Ollama Local LLM."""
    
    def __init__(self, model: str = "llama3.2:1b"):
        self.base_url = "http://localhost:11434/api/generate"
        self.model = model
    
    @property
    def name(self) -> str:
        return "Ollama (Local)"
    
    async def generate(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.7, "num_predict": max_tokens}
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "").strip()
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
        return None


class SmartTemplateProvider(LLMProvider):
    """Fallback: Smart template-based answers."""
    
    @property
    def name(self) -> str:
        return "Template (Offline)"
    
    async def generate(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        # Extract context and question from prompt
        if "Context:" in prompt and "Question:" in prompt:
            parts = prompt.split("Question:")
            context = parts[0].replace("Context:", "").strip()
            question = parts[1].split("Answer:")[0].strip() if len(parts) > 1 else ""
            return self._smart_answer(question, context)
        return None
    
    def _smart_answer(self, question: str, context: str) -> str:
        sentences = [s.strip() for s in context.replace('\n', '. ').split('.') if len(s.strip()) > 20]
        if not sentences:
            return f"From the documents: {context[:300]}..."
        
        key_info = ". ".join(sentences[:3])
        q_lower = question.lower()
        
        if any(w in q_lower for w in ['what', 'gì', 'là gì']):
            return f"According to the documents: {key_info}"
        elif any(w in q_lower for w in ['how', 'làm sao', 'như thế nào', 'cách']):
            return f"Based on the knowledge base: {key_info}"
        elif any(w in q_lower for w in ['why', 'tại sao', 'vì sao']):
            return f"The documents explain: {key_info}"
        else:
            return f"From the uploaded documents: {key_info}"


class LLMManager:
    """Manager for multiple LLM providers with fallback."""
    
    def __init__(self):
        self.providers = []
        self._active_provider = None
        
        # Add providers in priority order (NO template fallback)
        self.providers.append(GroqProvider())
        self.providers.append(OpenAIProvider())
        self.providers.append(OllamaProvider())
        # Template removed - require real LLM
    
    def set_api_keys(self, groq_key: str = None, openai_key: str = None):
        """Update API keys at runtime."""
        for p in self.providers:
            if isinstance(p, GroqProvider) and groq_key:
                p.api_key = groq_key
            elif isinstance(p, OpenAIProvider) and openai_key:
                p.api_key = openai_key
    
    async def generate(self, prompt: str, max_tokens: int = 300) -> tuple[str, str]:
        """Try providers in order, return (response, provider_name)."""
        for provider in self.providers:
            result = await provider.generate(prompt, max_tokens)
            if result:
                self._active_provider = provider.name
                return result, provider.name
        
        # No LLM available - return helpful message
        return "⚠️ Vui lòng cấu hình LLM để có câu trả lời tự nhiên!\n\n👉 Click ⚙️ LLM Settings → Nhập Groq API Key (miễn phí tại console.groq.com)", "No LLM"
    
    def get_available_providers(self) -> list[Dict[str, Any]]:
        """Return list of available providers and their status."""
        result = []
        for p in self.providers:
            has_key = True
            if isinstance(p, GroqProvider):
                has_key = bool(p.api_key)
            elif isinstance(p, OpenAIProvider):
                has_key = bool(p.api_key)
            
            result.append({
                "name": p.name,
                "available": has_key,
                "type": type(p).__name__
            })
        return result


# Global instance
llm_manager = LLMManager()


async def generate_answer(question: str, context: str, groq_key: str = None, openai_key: str = None) -> tuple[str, str]:
    """Generate answer using best available LLM provider."""
    
    # Update keys if provided
    llm_manager.set_api_keys(groq_key=groq_key, openai_key=openai_key)
    
    prompt = f"""Based on the following context, answer the question naturally and concisely in the same language as the question.

Context:
{context[:2000]}

Question: {question}

Answer:"""
    
    return await llm_manager.generate(prompt)
