# MAIRA-2 Local Deployment

Local FastAPI deployment for `microsoft/maira-2` with a browser frontend, SSE token streaming, and sample chest X-ray images for testing.

MAIRA-2 is gated on Hugging Face. Before the first model load, sign in with an account that has accepted the model terms:

```bash
source .venv/bin/activate
huggingface-cli login
```

## Run

Start the app on the default port `8100`:

```bash
./run_server.sh
```

Hot reload is enabled by default for local development. Disable it when needed:

```bash
RELOAD=0 ./run_server.sh
```

Optional runtime settings:

```bash
export HOST=0.0.0.0
export PORT=8100
export MAIRA_MODEL_ID=microsoft/maira-2
export MAIRA_DEVICE=cuda
export MAIRA_TORCH_DTYPE=bfloat16
```

Open the frontend:

```text
http://127.0.0.1:8100/
```

Health check:

```bash
curl http://127.0.0.1:8100/health
```

Warm up and load the model:

```bash
curl -X POST http://127.0.0.1:8100/warmup
```

## Frontend Workflows

The browser UI supports the MAIRA-2 model-card workflows:

- **Report**: generate a plain report from frontal/lateral/prior images and study text.
- **Grounded Report**: generate a report plus finding boxes over the frontal image.
- **Phrase Grounding**: localize one specific finding phrase on the frontal image.

The study-text and phrase fields are placeholders only; they start empty. `Max new tokens` is the generation ceiling, not a required output length. Current defaults are:

- Report: `300`
- Grounded Report: `450`
- Phrase Grounding: `150`
- Frontend max: `800`

## API Examples

Generate a report:

```bash
curl -X POST http://127.0.0.1:8100/report \
  -F frontal=@/path/to/frontal.png \
  -F lateral=@/path/to/lateral.png \
  -F indication='Dyspnea.' \
  -F comparison='None.' \
  -F technique='PA and lateral views of the chest.' \
  -F get_grounding=false
```

Stream report tokens with SSE:

```bash
curl -N -X POST http://127.0.0.1:8100/report-stream \
  -F frontal=@/path/to/frontal.png \
  -F get_grounding=false
```

Generate a grounded report:

```bash
curl -N -X POST http://127.0.0.1:8100/report-stream \
  -F frontal=@/path/to/frontal.png \
  -F get_grounding=true
```

Phrase grounding:

```bash
curl -N -X POST http://127.0.0.1:8100/phrase-ground-stream \
  -F frontal=@/path/to/frontal.png \
  -F phrase='Pneumothorax.'
```

The streaming endpoints emit SSE events:

- `start`: runtime metadata
- `token`: incremental generated text
- `final`: parsed output, including adjusted boxes where applicable
- `error`: generation or parsing error

## Notes

The frontend is intentionally plain HTML/CSS/JS served by FastAPI. There is no separate Node or React build step.

The model card states MAIRA-2 is for research and development only, not clinical decision-making.
