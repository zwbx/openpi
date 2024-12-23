# Simple Client

A minimal client that sends observations to the server and prints the inference rate.

## With Docker

```bash
export SERVER_ARGS="--example aloha"
docker compose -f examples/simple_client/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
uv run examples/simple_client/main.py
```

Terminal window 2:

```bash
uv run scripts/serve_policy.py
```