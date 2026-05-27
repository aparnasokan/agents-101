# agents/base.py
# Base class every agent inherits from.
# Handles the Azure OpenAI API call so each agent only needs to define its system prompt.

import os

from dotenv import load_dotenv
from openai import AzureOpenAI, BadRequestError

load_dotenv()

DEFAULT_API_VERSION = "2024-10-21"
_client: AzureOpenAI | None = None
_client_config: tuple[str, str, str] | None = None


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return ""


def get_azure_openai_config() -> dict[str, str]:
    api_key = _first_env("AZURE_OPENAI_API_KEY", "AZUREAI_API_KEY")
    endpoint = _first_env("AZURE_OPENAI_ENDPOINT", "AZUREAI_ENDPOINT")
    deployment = _first_env(
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_MODEL",
        "AZUREAI_DEPLOYMENT",
    )
    legacy_model = _first_env("AZUREAI_MODEL")

    if legacy_model.startswith(("http://", "https://")) and not endpoint:
        endpoint = legacy_model
    elif legacy_model and not deployment:
        deployment = legacy_model

    api_version = _first_env("AZURE_OPENAI_API_VERSION", "AZUREAI_API_VERSION") or DEFAULT_API_VERSION

    missing: list[str] = []
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not deployment:
        missing.append("AZURE_OPENAI_DEPLOYMENT")

    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"Missing Azure OpenAI configuration: {missing_text}")

    return {
        "api_key": api_key,
        "endpoint": endpoint.rstrip("/"),
        "deployment": deployment,
        "api_version": api_version,
    }


def get_client() -> AzureOpenAI:
    global _client, _client_config
    config = get_azure_openai_config()
    current_config = (
        config["api_key"],
        config["endpoint"],
        config["api_version"],
    )
    if _client is None or _client_config != current_config:
        _client = AzureOpenAI(
            api_key=config["api_key"],
            azure_endpoint=config["endpoint"],
            api_version=config["api_version"],
        )
        _client_config = current_config
    return _client


SCOPE_GUARD = (
    "\n\nSCOPE RULE (mandatory — overrides everything else): "
    "You are part of a learning tool exclusively about AI agents and agentic workflows. "
    "If the user's message contains any parts unrelated to AI agents, agentic workflows, LLMs, "
    "tool calling, orchestration, RAG, memory, planning, multi-agent systems, or building/deploying AI systems — "
    "silently ignore those unrelated parts and answer only the relevant portion. "
    "Do not acknowledge, answer, or reference the unrelated parts in any way. "
    "Do not mention that you are skipping anything. Simply focus on what is in scope."
)


class BaseAgent:
    name: str = "base"
    system_prompt: str = ""

    def _create_chat_completion(self, messages: list[dict], max_tokens: int):
        config = get_azure_openai_config()
        request = {
            "model": config["deployment"],
            "messages": [
                {"role": "system", "content": self.system_prompt + SCOPE_GUARD},
                *messages,
            ],
        }

        # Newer Azure-hosted reasoning models expect max_completion_tokens,
        # while older chat models still use max_tokens.
        try:
            return get_client().chat.completions.create(
                **request,
                max_completion_tokens=max_tokens,
            )
        except BadRequestError as exc:
            error_text = str(exc)
            if "max_completion_tokens" not in error_text:
                raise

        return get_client().chat.completions.create(
            **request,
            max_tokens=max_tokens,
        )

    def run(self, messages: list[dict], max_tokens: int = 2500) -> str:
        """
        Call Azure OpenAI with this agent's system prompt and the provided message history.
        Returns the text response.
        """
        response = self._create_chat_completion(messages, max_tokens)
        return response.choices[0].message.content or ""

    def run_with_last_user_message(self, history: list[dict], user_message: str, max_tokens: int = 2500) -> str:
        """
        Convenience method: takes full history + current user message and calls Azure OpenAI.
        """
        messages = history + [{"role": "user", "content": user_message}]
        return self.run(messages, max_tokens)
