"""LLM client for AutoSOTA."""

import os
import json
from typing import Dict, Any, Optional, List
from openai import OpenAI
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core import get_logger, AutoSOTAError


logger = get_logger(__name__)


class LLMClient:
    """Unified LLM client supporting OpenAI and Anthropic."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 120,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        if provider == "openai":
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            if not api_key:
                raise AutoSOTAError("OPENAI_API_KEY not set")
            self.client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
        elif provider == "anthropic":
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise AutoSOTAError("ANTHROPIC_API_KEY not set")
            self.client = Anthropic(api_key=api_key, timeout=timeout)
        else:
            raise AutoSOTAError(f"Unsupported LLM provider: {provider}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send chat completion request."""
        try:
            if self.provider == "openai":
                return self._chat_openai(messages, system_prompt, response_format)
            elif self.provider == "anthropic":
                return self._chat_anthropic(messages, system_prompt)
            else:
                raise AutoSOTAError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise

    def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """OpenAI chat completion."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if system_prompt:
            kwargs["messages"] = [{"role": "system", "content": system_prompt}] + messages

        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _chat_anthropic(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Anthropic chat completion."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send chat completion request and parse JSON response."""
        response = self.chat(messages, system_prompt)

        # Try to extract JSON from response
        try:
            # Look for JSON in code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response.strip()

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response: {response}")
            raise AutoSOTAError(f"Failed to parse JSON response: {e}")

    def load_prompt_template(self, template_path: str) -> Dict[str, Any]:
        """Load prompt template from YAML file."""
        import yaml

        with open(template_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def format_prompt(
        self,
        template: Dict[str, Any],
        variables: Dict[str, Any],
    ) -> tuple:
        """Format prompt template with variables."""
        system_prompt = template.get("system_prompt", "")
        user_prompt = template.get("user_prompt", "")

        # Replace variables in prompts
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            system_prompt = system_prompt.replace(placeholder, str(value))
            user_prompt = user_prompt.replace(placeholder, str(value))

        return system_prompt, user_prompt
