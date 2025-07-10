"""
Unified AI client wrapper for OpenAI and Gemini APIs
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import retry
from openai import OpenAI
from google import genai


@dataclass
class TokenUsage:
    """Unified token usage representation"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class CompletionResponse:
    """Unified completion response"""
    content: str
    usage: TokenUsage


class AIClientWrapper(ABC):
    """Abstract base class for AI client wrappers"""
    
    @abstractmethod
    def generate_completion(self, prompt: str, model: str, temperature: float = 0.0, seed: Optional[int] = 0) -> CompletionResponse:
        """Generate a completion from the AI model"""
        pass


class OpenAIWrapper(AIClientWrapper):
    """Wrapper for OpenAI client"""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    @retry.retry(tries=3, delay=2)
    def generate_completion(self, prompt: str, model: str, temperature: float = 0.0, seed: Optional[int] = 0) -> CompletionResponse:
        completion = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            seed=seed,
        )
        
        usage = TokenUsage(
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            total_tokens=completion.usage.prompt_tokens + completion.usage.completion_tokens
        )
        
        return CompletionResponse(
            content=completion.choices[0].message.content,
            usage=usage
        )


class GeminiWrapper(AIClientWrapper):
    """Wrapper for Gemini client"""
    
    def __init__(self, client: genai.Client):
        self.client = client
    
    @retry.retry(tries=3, delay=2)
    def generate_completion(self, prompt: str, model: str, temperature: float = 0.0, seed: Optional[int] = None) -> CompletionResponse:
        # Note: Gemini doesn't support seed parameter
        completion = self.client.models.generate_content(
            model=model,
            contents=prompt
        )
        
        # Gemini doesn't provide token usage, so we return zeros
        usage = TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )
        
        return CompletionResponse(
            content=completion.text,
            usage=usage
        )


def create_ai_client(client_type: str, **kwargs) -> AIClientWrapper:
    """Factory function to create the appropriate AI client wrapper"""
    if client_type.lower() == "gemini":
        if "api_key" not in kwargs:
            raise ValueError("api_key is required for Gemini client")
        gemini_client = genai.Client(api_key=kwargs["api_key"])
        return GeminiWrapper(gemini_client)
    else:  # Default to OpenAI
        if "api_key" not in kwargs:
            raise ValueError("api_key is required for OpenAI client")
        openai_client = OpenAI(
            api_key=kwargs["api_key"],
            base_url=kwargs.get("base_url")
        )
        return OpenAIWrapper(openai_client)