# Per-model fatal error log

Each entry records a model whose run was aborted because the API returned permanent failures.

## `qwen3-0.6b-base` — 2026-07-10T17:37:29

```
Model `qwen3-0.6b-base` aborted after 8 consecutive permanent failures. Last error: HTTP 400: {"error":{"message":"max_tokens=8192 cannot be greater than max_model_len=max_total_tokens=4096. Please request fewer output tokens. (parameter=max_tokens, value=8192)","type":"BadRequestError","param":"max_tokens","code":400}}
```

