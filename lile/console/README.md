# lile trial — Qwen3.5-9B on 127.0.0.1:8765

Server PID: tail of `/tmp/lile-trial/server.log`. Stop with `fuser -k 8765/tcp`.

## Quick chat

```bash
curl -sS -X POST http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"YOUR PROMPT"}],"max_tokens":256}' | jq
```

Response includes `lile.response_id` and `lile.commit_cursor`. Keep the `response_id` — feedback references it.

## Feedback — four kinds

```bash
# (1) binary → KTO (label: "desirable" | "undesirable")
curl -sS -X POST http://127.0.0.1:8765/v1/feedback -H 'Content-Type: application/json' \
  -d '{"response_id":"<id>","kind":"binary","label":"desirable"}'

# (2) rewrite → weighted SFT on your correction
curl -sS -X POST http://127.0.0.1:8765/v1/feedback -H 'Content-Type: application/json' \
  -d '{"response_id":"<id>","kind":"rewrite","rewrite":"what the reply SHOULD have been"}'

# (3) preferred → hinge contrastive (chosen=your rewrite, rejected=original)
curl -sS -X POST http://127.0.0.1:8765/v1/feedback -H 'Content-Type: application/json' \
  -d '{"response_id":"<id>","kind":"preferred","preferred":"your preferred reply"}'

# (4) nl_critique → CoH (critique-conditioned SFT) or CCPD v2 if opted in
curl -sS -X POST http://127.0.0.1:8765/v1/feedback -H 'Content-Type: application/json' \
  -d '{"response_id":"<id>","kind":"nl_critique","critique":"be more concise"}'
```

Every feedback returns `commit_token: N`. To make the NEXT chat block until that training step is reflected, pass `after_commit_token: N`:

```bash
curl -sS -X POST http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"same question again"}],"after_commit_token":N}'
```

The response body's `lile.commit_cursor` will be ≥ N.

## Inspect

```bash
curl -sS http://127.0.0.1:8765/health | jq
curl -sS 'http://127.0.0.1:8765/v1/state/trajectory/tail?n=10' | jq
```

## Direct training batches (/v1/train)

```bash
curl -sS -X POST http://127.0.0.1:8765/v1/train -H 'Content-Type: application/json' -d '{
  "objective":"sft",
  "samples":[{"prompt":"2+2=","response":"4"}]
}'
```

## Known good probe

- First chat (cold): ~8.7 s
- KTO feedback → commit cursor bumps to 0
- Chat with `after_commit_token=0`: cursor=0 reflected, ~3.5 s
- VRAM at rest: ~9.4 GB / 24 GB
