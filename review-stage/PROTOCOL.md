# Administration Protocol

## Item Presentation
- All 221 items administered in a single batch per persona condition
- Items ordered by scale (EPQR-A → IPIP-NEO-120 → SD3 → ZKPQ-50-CC) then by item_id within each scale
- Forward and reverse items interleaved within each domain (not separated)
- No randomization of item order across conditions or models

## Response Formats
- IPIP-NEO-120: 5-point Likert (1=Very Inaccurate to 5=Very Accurate)
- SD3: 5-point Likert (1=Strongly Disagree to 5=Strongly Agree)
- ZKPQ-50-CC: True/False
- EPQR-A: Yes/No

## Prompting
- Default condition: no system prompt, items framed as self-referential statements
- MBTI conditions: system prompt assigns MBTI type as persona
- Example prompt: 'Please respond to the following statement as honestly as possible.'
- Response parsing: deterministic extraction of Likert value or Yes/No from model output

## Decoding Parameters
- Temperature: 0.7
- Max tokens: 8192
- Batch size: 16 concurrent requests
- Max retries: 10 per item

## Session Management
- Each item is an independent API call (no conversation context)
- No session memory between items (stateless administration)
- Parse failures: retried up to max_retries times; if still failing, recorded as parse_failed
- Refusals treated as missing data (not scored)

## Scoring
- Forward items (+): scored as raw response value
- Reverse items (-): Likert scored as 6-raw, binary scored as 1-raw
- Domain scores: mean of scored values within domain

## Limitations
- Single administration (no test-retest reliability)
- Temperature = 0.7 (not deterministic; introduces stochastic variance)
- No item-order randomization (order effects not controlled)
- No session memory (items not conditioned on prior responses)