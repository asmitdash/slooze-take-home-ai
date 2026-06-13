"""Thin wrapper over AWS Bedrock for Anthropic Claude models.

Single point of configuration so both the web-search agent and the PDF RAG
agent share the same client and prompting conventions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import boto3


DEFAULT_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "global.anthropic.claude-sonnet-4-6",
)
DEFAULT_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


class BedrockClaude:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, region: str = DEFAULT_REGION):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1500,
        temperature: float = 0.2,
    ) -> LLMResponse:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        resp = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(resp["body"].read())
        text = "".join(
            block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
        )
        usage = payload.get("usage", {})
        return LLMResponse(
            text=text.strip(),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
