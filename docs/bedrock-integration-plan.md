# Plan: Add Amazon Bedrock Provider

## Context

The eval harness supports Anthropic, OpenAI, Google, and Ollama providers via direct HTTP calls. Adding Bedrock support enables evaluating any model available through AWS Bedrock (Claude, Llama, Mistral, Cohere, etc.) using AWS credentials instead of per-vendor API keys - useful if you already have Bedrock access or want to test models only available there.

## Key Design Decision: boto3 vs raw httpx

The project's convention is "no SDKs - use httpx directly". However, Bedrock requires **AWS Signature Version 4** authentication, which involves HMAC-SHA256 signing of canonical requests with timestamps, credential scopes, and signed headers. Implementing this manually would be ~100 lines of crypto boilerplate.

**Approach: Use `boto3`** for the Bedrock provider only. This is the pragmatic choice - SigV4 is the one case where the SDK genuinely earns its keep. The alternative (manual SigV4 or a lightweight signing library) would add fragile, hard-to-debug auth code for no real benefit.

## AWS Credentials Required

Bedrock needs standard AWS credentials. These are resolved by boto3's credential chain in this order:

1. **Environment variables** (recommended for this project):
   - `AWS_ACCESS_KEY_ID` - your IAM access key
   - `AWS_SECRET_ACCESS_KEY` - your IAM secret key
   - `AWS_SESSION_TOKEN` (optional) - only needed for temporary/assumed-role credentials
   - `AWS_DEFAULT_REGION` - the region where Bedrock models are enabled (e.g. `us-east-1`)

2. **AWS credentials file** (`~/.aws/credentials`) - boto3 picks these up automatically
3. **IAM instance roles** - if running on EC2/Lambda

The IAM user/role needs the `bedrock:InvokeModel` permission.

Store the env vars in your `.env` file (already loaded by `python-dotenv`):
```
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=wJal...
AWS_DEFAULT_REGION=us-east-1
```

## What Changes

### 1. Add `BedrockProvider` class in `scripts/providers.py`

New provider using boto3's `bedrock-runtime` client with the Converse API:

```python
class BedrockProvider(Provider):
    def __init__(self, model: str, region: str = None):
        import boto3
        self.model = model
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def complete(self, prompt, params):
        response = self.client.converse(
            modelId=self.model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={
                "maxTokens": params.get("max_tokens", 4096),
                "temperature": params.get("temperature", 0),
            },
        )
        content = response["output"]["message"]["content"][0]["text"]
        usage = {
            "input_tokens": response["usage"]["inputTokens"],
            "output_tokens": response["usage"]["outputTokens"],
        }
        return content, usage
```

Register in `get_provider()`:
```python
elif provider_type == "bedrock":
    region = config.get("region")
    return BedrockProvider(config["model"], region)
```

Note: boto3 import is inside the class `__init__` to keep it lazy (same pattern as our deepeval lazy imports) - won't break anything if boto3 isn't installed and you're not using Bedrock.

### 2. Add Bedrock model entries to `config.yaml`

All entries follow this pattern - no `api_key_env` needed (boto3 handles auth via the AWS credential chain):

```yaml
  bedrock-model-name:
    provider: bedrock
    model: vendor.model-id-string
    region: us-east-1     # optional, falls back to AWS_DEFAULT_REGION
    params:
      max_tokens: 4096
      temperature: 0
```

**Available Bedrock text/chat models (grouped by provider):**

| Config name | Bedrock model ID |
|---|---|
| **Anthropic** | |
| bedrock-claude-opus-4.6 | `anthropic.claude-opus-4-6-v1` |
| bedrock-claude-opus-4.5 | `anthropic.claude-opus-4-5-20251101-v1:0` |
| bedrock-claude-opus-4.1 | `anthropic.claude-opus-4-1-20250805-v1:0` |
| bedrock-claude-sonnet-4.5 | `anthropic.claude-sonnet-4-5-20250929-v1:0` |
| bedrock-claude-sonnet-4 | `anthropic.claude-sonnet-4-20250514-v1:0` |
| bedrock-claude-haiku-4.5 | `anthropic.claude-haiku-4-5-20251001-v1:0` |
| bedrock-claude-3.5-haiku | `anthropic.claude-3-5-haiku-20241022-v1:0` |
| bedrock-claude-3-haiku | `anthropic.claude-3-haiku-20240307-v1:0` |
| **Meta** | |
| bedrock-llama3.3-70b | `meta.llama3-3-70b-instruct-v1:0` |
| bedrock-llama3.1-405b | `meta.llama3-1-405b-instruct-v1:0` |
| bedrock-llama3.1-70b | `meta.llama3-1-70b-instruct-v1:0` |
| bedrock-llama3.1-8b | `meta.llama3-1-8b-instruct-v1:0` |
| bedrock-llama3-70b | `meta.llama3-70b-instruct-v1:0` |
| bedrock-llama3-8b | `meta.llama3-8b-instruct-v1:0` |
| bedrock-llama3.2-3b | `meta.llama3-2-3b-instruct-v1:0` |
| bedrock-llama3.2-1b | `meta.llama3-2-1b-instruct-v1:0` |
| **Mistral** | |
| bedrock-magistral-small | `mistral.magistral-small-2509` |
| bedrock-mistral-large-2407 | `mistral.mistral-large-2407-v1:0` |
| bedrock-mistral-large-2402 | `mistral.mistral-large-2402-v1:0` |
| bedrock-mistral-small | `mistral.mistral-small-2402-v1:0` |
| bedrock-mixtral-8x7b | `mistral.mixtral-8x7b-instruct-v0:1` |
| bedrock-mistral-7b | `mistral.mistral-7b-instruct-v0:2` |
| **DeepSeek** | |
| bedrock-deepseek-r1 | `deepseek.r1-v1:0` |
| bedrock-deepseek-v3 | `deepseek.v3-v1:0` |
| **OpenAI** | |
| bedrock-gpt-oss-120b | `openai.gpt-oss-120b-1:0` |
| bedrock-gpt-oss-20b | `openai.gpt-oss-20b-1:0` |
| **Cohere** | |
| bedrock-command-r-plus | `cohere.command-r-plus-v1:0` |
| bedrock-command-r | `cohere.command-r-v1:0` |
| **AI21** | |
| bedrock-jamba-1.5-large | `ai21.jamba-1-5-large-v1:0` |
| bedrock-jamba-1.5-mini | `ai21.jamba-1-5-mini-v1:0` |
| **Amazon** | |
| bedrock-titan-text | `amazon.titan-tg1-large` |
| **Qwen** | |
| bedrock-qwen3-235b | `qwen.qwen3-235b-a22b-2507-v1:0` |
| bedrock-qwen3-32b | `qwen.qwen3-32b-v1:0` |
| bedrock-qwen3-coder-480b | `qwen.qwen3-coder-480b-a35b-v1:0` |
| bedrock-qwen3-coder-30b | `qwen.qwen3-coder-30b-a3b-v1:0` |
| **Google** | |
| bedrock-gemma3-27b | `google.gemma-3-27b-it` |
| bedrock-gemma3-12b | `google.gemma-3-12b-it` |
| bedrock-gemma3-4b | `google.gemma-3-4b-it` |
| **Writer** | |
| bedrock-palmyra-x5 | `writer.palmyra-x5-v1:0` |
| bedrock-palmyra-x4 | `writer.palmyra-x4-v1:0` |
| **Moonshot** | |
| bedrock-kimi-k2 | `moonshot.kimi-k2-thinking` |
| **MiniMax** | |
| bedrock-minimax-m2 | `minimax.minimax-m2` |

### 3. Add `boto3` to `requirements.txt`

```
boto3>=1.35
```

## Files Modified

| File | Change |
|---|---|
| `scripts/providers.py` | Add `BedrockProvider` class, register in `get_provider()` |
| `config.yaml` | Add Bedrock model entries (commented out by default) |
| `requirements.txt` | Add `boto3>=1.35` |

## Implementation Order

1. Add `boto3>=1.35` to `requirements.txt`, install
2. Add `BedrockProvider` class to `scripts/providers.py`
3. Register `"bedrock"` in `get_provider()` factory
4. Add Bedrock model configs to `config.yaml`
5. Test: `python run.py eval bedrock-claude-sonnet-4 --ids C01`

## Verification

- `python run.py eval bedrock-claude-sonnet-4 --ids C01 C02` completes successfully, stores results with token counts
- `python run.py models` shows the new Bedrock model
- Existing providers continue working unchanged
- Running without AWS credentials configured gives a clear error message (boto3's default "Unable to locate credentials")
