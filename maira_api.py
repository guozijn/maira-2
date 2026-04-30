import io
import json
import os
from threading import Thread
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer


MODEL_ID = os.getenv("MAIRA_MODEL_ID", "microsoft/maira-2")
DEVICE = torch.device(os.getenv("MAIRA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
DTYPE_NAME = os.getenv("MAIRA_TORCH_DTYPE", "bfloat16" if DEVICE.type == "cuda" else "float32")
DTYPE = getattr(torch, DTYPE_NAME)

app = FastAPI(
    title="MAIRA-2 local inference",
    description="Research-only deployment wrapper for microsoft/maira-2.",
)
app.mount("/static", StaticFiles(directory="static"), name="static")

_model: Any | None = None
_processor: Any | None = None


@app.middleware("http")
async def no_cache_frontend(request: Any, call_next: Any) -> Any:
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def _load_image(upload: UploadFile | None) -> Image.Image | None:
    if upload is None:
        return None
    try:
        return Image.open(io.BytesIO(upload.file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image upload {upload.filename!r}: {exc}") from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _adjust_prediction_boxes(prediction: Any, processor: Any, image: Image.Image) -> Any:
    if not isinstance(prediction, list):
        return prediction
    width, height = image.size
    adjusted = []
    for item in prediction:
        if not isinstance(item, tuple) or len(item) != 2:
            adjusted.append(item)
            continue
        phrase, boxes = item
        if boxes is None:
            adjusted.append((phrase, None))
            continue
        adjusted.append((phrase, [processor.adjust_box_for_original_image_size(box, width, height) for box in boxes]))
    return adjusted


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _prediction_payload(decoded: str, prediction: Any, processor: Any, frontal_image: Image.Image) -> dict[str, Any]:
    adjusted_prediction = _adjust_prediction_boxes(prediction, processor, frontal_image)
    return {
        "decoded": decoded,
        "prediction": _jsonable(adjusted_prediction),
        "raw_prediction": _jsonable(prediction),
        "research_only": True,
    }


def _clean_streamed_text(text: str, processor: Any) -> str:
    tokenizer = getattr(processor, "tokenizer", None)
    for token in (
        getattr(tokenizer, "eos_token", None),
        getattr(tokenizer, "bos_token", None),
        getattr(tokenizer, "pad_token", None),
    ):
        if token:
            text = text.replace(token, "")
    return text


def _stream_generation(
    model: Any,
    processor: Any,
    processed_inputs: Any,
    frontal_image: Image.Image,
    max_new_tokens: int,
    skip_special_tokens: bool,
    lstrip_decoded: bool,
) -> StreamingResponse:
    tokenizer = getattr(processor, "tokenizer", processor)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=skip_special_tokens,
    )
    generation_error: list[BaseException] = []
    generation_kwargs = {
        **processed_inputs,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "streamer": streamer,
    }

    def generate() -> None:
        try:
            with torch.inference_mode():
                model.generate(**generation_kwargs)
        except BaseException as exc:
            generation_error.append(exc)
            streamer.on_finalized_text("", stream_end=True)

    def events():
        chunks: list[str] = []
        yield _sse("start", {"model_id": MODEL_ID, "device": str(DEVICE), "dtype": DTYPE_NAME})

        thread = Thread(target=generate, daemon=True)
        thread.start()
        for chunk in streamer:
            chunks.append(chunk)
            yield _sse("token", {"text": chunk})
        thread.join()

        if generation_error:
            yield _sse("error", {"detail": str(generation_error[0])})
            return

        decoded = "".join(chunks)
        decoded = _clean_streamed_text(decoded, processor)
        if lstrip_decoded:
            decoded = decoded.lstrip()
        try:
            prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)
        except Exception as exc:
            yield _sse("error", {"detail": f"Could not parse generated output: {exc}", "decoded": decoded})
            return
        yield _sse("final", _prediction_payload(decoded, prediction, processor, frontal_image))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def get_model() -> tuple[Any, Any]:
    global _model, _processor
    if _model is None or _processor is None:
        try:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                torch_dtype=DTYPE,
                low_cpu_mem_usage=True,
            ).eval()
            _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
            _model.to(DEVICE)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Could not load {MODEL_ID}. If this is an authentication error, run "
                    "`huggingface-cli login` with an account that has accepted the model terms. "
                    f"Original error: {exc}"
                ),
            ) from exc
    return _model, _processor


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        "static/index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_id": MODEL_ID,
        "loaded": _model is not None,
        "device": str(DEVICE),
        "dtype": DTYPE_NAME,
        "cuda_available": torch.cuda.is_available(),
        "research_only": True,
    }


@app.post("/warmup")
def warmup() -> dict[str, Any]:
    get_model()
    return {"loaded": True, "model_id": MODEL_ID, "device": str(DEVICE), "dtype": DTYPE_NAME}


@app.post("/report")
def report(
    frontal: UploadFile = File(...),
    lateral: UploadFile | None = File(None),
    prior_frontal: UploadFile | None = File(None),
    indication: str | None = Form(None),
    technique: str | None = Form(None),
    comparison: str | None = Form(None),
    prior_report: str | None = Form(None),
    get_grounding: bool = Form(False),
    max_new_tokens: int | None = Form(None),
) -> dict[str, Any]:
    model, processor = get_model()
    frontal_image = _load_image(frontal)
    lateral_image = _load_image(lateral)
    prior_frontal_image = _load_image(prior_frontal)

    tokens = max_new_tokens if max_new_tokens is not None else (450 if get_grounding else 300)
    processed_inputs = processor.format_and_preprocess_reporting_input(
        current_frontal=frontal_image,
        current_lateral=lateral_image,
        prior_frontal=prior_frontal_image,
        indication=indication,
        technique=technique,
        comparison=comparison,
        prior_report=prior_report,
        return_tensors="pt",
        get_grounding=get_grounding,
    ).to(DEVICE)

    with torch.inference_mode():
        output = model.generate(**processed_inputs, max_new_tokens=tokens, use_cache=True)

    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded = processor.decode(output[0][prompt_length:], skip_special_tokens=True).lstrip()
    prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)
    return _prediction_payload(decoded, prediction, processor, frontal_image)


@app.post("/report-stream")
def report_stream(
    frontal: UploadFile = File(...),
    lateral: UploadFile | None = File(None),
    prior_frontal: UploadFile | None = File(None),
    indication: str | None = Form(None),
    technique: str | None = Form(None),
    comparison: str | None = Form(None),
    prior_report: str | None = Form(None),
    get_grounding: bool = Form(False),
    max_new_tokens: int | None = Form(None),
) -> StreamingResponse:
    model, processor = get_model()
    frontal_image = _load_image(frontal)
    lateral_image = _load_image(lateral)
    prior_frontal_image = _load_image(prior_frontal)

    tokens = max_new_tokens if max_new_tokens is not None else (450 if get_grounding else 300)
    processed_inputs = processor.format_and_preprocess_reporting_input(
        current_frontal=frontal_image,
        current_lateral=lateral_image,
        prior_frontal=prior_frontal_image,
        indication=indication,
        technique=technique,
        comparison=comparison,
        prior_report=prior_report,
        return_tensors="pt",
        get_grounding=get_grounding,
    ).to(DEVICE)
    return _stream_generation(
        model=model,
        processor=processor,
        processed_inputs=processed_inputs,
        frontal_image=frontal_image,
        max_new_tokens=tokens,
        skip_special_tokens=not get_grounding,
        lstrip_decoded=True,
    )


@app.post("/phrase-ground")
def phrase_ground(
    frontal: UploadFile = File(...),
    phrase: str = Form(...),
    max_new_tokens: int = Form(150),
) -> dict[str, Any]:
    model, processor = get_model()
    frontal_image = _load_image(frontal)
    processed_inputs = processor.format_and_preprocess_phrase_grounding_input(
        frontal_image=frontal_image,
        phrase=phrase,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        output = model.generate(**processed_inputs, max_new_tokens=max_new_tokens, use_cache=True)

    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded = processor.decode(output[0][prompt_length:], skip_special_tokens=True)
    prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)
    return _prediction_payload(decoded, prediction, processor, frontal_image)


@app.post("/phrase-ground-stream")
def phrase_ground_stream(
    frontal: UploadFile = File(...),
    phrase: str = Form(...),
    max_new_tokens: int = Form(150),
) -> StreamingResponse:
    model, processor = get_model()
    frontal_image = _load_image(frontal)
    processed_inputs = processor.format_and_preprocess_phrase_grounding_input(
        frontal_image=frontal_image,
        phrase=phrase,
        return_tensors="pt",
    ).to(DEVICE)
    return _stream_generation(
        model=model,
        processor=processor,
        processed_inputs=processed_inputs,
        frontal_image=frontal_image,
        max_new_tokens=max_new_tokens,
        skip_special_tokens=False,
        lstrip_decoded=False,
    )
