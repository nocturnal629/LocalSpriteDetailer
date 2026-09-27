"""
Sprite Detail Frontend - a minimal local UI for the ComfyUI sprite-upscale workflow.

Talks to a running ComfyUI server (http://127.0.0.1:8188) on the user's behalf so the
browser never has to call ComfyUI's API directly (ComfyUI doesn't send CORS headers).
"""
import base64
import io
import json
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory, render_template

load_dotenv()

COMFY_URL = "http://127.0.0.1:8188"
COMFY_INPUT_DIR = Path.home() / "ComfyUI" / "input"
COMFY_OUTPUT_DIR = Path.home() / "ComfyUI" / "output"

CHECKPOINT = "sd_xl_base_1.0.safetensors"
LORA = "pixel-art-xl.safetensors"
CONTROLNET = "controlnet-canny-sdxl-fp16.safetensors"

# Optional "design-fidelity QA" feature: asks a CloudIQ multimodal model to compare
# a generated sprite against its source. Not required for the core generate pipeline.
CLOUDIQ_URL = os.getenv("CLOUDIQ_URL", "https://cloudiq-2t4e.onrender.com")
CLOUDIQ_API_KEY = os.getenv("CLOUDIQ_API_KEY")

# SDXL's VAE needs dimensions that are multiples of 8. These bounds keep the working
# resolution in a range SDXL was actually trained for, even for tiny sprite sources
# (e.g. a 32x32 or 64x64 sprite scaled up client-side by a chosen detail factor).
MIN_WORKING_DIM = 64
MAX_WORKING_DIM = 1536
MAX_BATCH_SIZE = 8


def snap_to_working_dim(value: int) -> int:
    clamped = max(MIN_WORKING_DIM, min(MAX_WORKING_DIM, value))
    return int(round(clamped / 8) * 8)


app = Flask(__name__)

# in-memory session history: list of dicts, newest first
HISTORY = []


def build_workflow(image_name: str, params: dict, prefix: str) -> dict:
    """Build the API-format ComfyUI workflow graph for the img2img + ControlNet + LoRA pipeline."""
    return {
        "prompt": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": CHECKPOINT},
            },
            "2": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["1", 0],
                    "clip": ["1", 1],
                    "lora_name": LORA,
                    "strength_model": params["lora_strength"],
                    "strength_clip": params["lora_strength"],
                },
            },
            "3": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name},
            },
            "4": {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["3", 0],
                    "upscale_method": "nearest-exact",
                    "width": params["width"],
                    "height": params["height"],
                    "crop": "disabled",
                },
            },
            "5": {
                "class_type": "Canny",
                "inputs": {
                    "image": ["4", 0],
                    "low_threshold": 0.4,
                    "high_threshold": 0.8,
                },
            },
            "6": {
                "class_type": "ControlNetLoader",
                "inputs": {"control_net_name": CONTROLNET},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["2", 1], "text": params["positive_prompt"]},
            },
            "8": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["2", 1], "text": params["negative_prompt"]},
            },
            "9": {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": ["7", 0],
                    "negative": ["8", 0],
                    "control_net": ["6", 0],
                    "image": ["5", 0],
                    "strength": params["controlnet_strength"],
                    "start_percent": 0.0,
                    "end_percent": 1.0,
                },
            },
            "10": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["4", 0], "vae": ["1", 2]},
            },
            "10b": {
                "class_type": "RepeatLatentBatch",
                "inputs": {"samples": ["10", 0], "amount": params["batch_size"]},
            },
            "11": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["2", 0],
                    "seed": params["seed"],
                    "steps": params["steps"],
                    "cfg": params["cfg"],
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "positive": ["9", 0],
                    "negative": ["9", 1],
                    "latent_image": ["10b", 0],
                    "denoise": params["denoise"],
                },
            },
            "12": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["11", 0], "vae": ["1", 2]},
            },
            "13": {
                "class_type": "SaveImage",
                "inputs": {"images": ["12", 0], "filename_prefix": prefix},
            },
        }
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    upload = request.files["image"]

    job_id = uuid.uuid4().hex[:8]
    ext = Path(upload.filename).suffix or ".png"
    input_name = f"upload_{job_id}{ext}"
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    upload.save(COMFY_INPUT_DIR / input_name)

    def f(name, default):
        val = request.form.get(name, default)
        return val

    params = {
        "width": snap_to_working_dim(int(f("width", 512))),
        "height": snap_to_working_dim(int(f("height", 512))),
        "batch_size": max(1, min(MAX_BATCH_SIZE, int(f("batch_size", 1)))),
        "denoise": float(f("denoise", 0.55)),
        "controlnet_strength": float(f("controlnet_strength", 0.6)),
        "lora_strength": float(f("lora_strength", 0.8)),
        "steps": int(f("steps", 30)),
        "cfg": float(f("cfg", 7.0)),
        "seed": int(f("seed", 42)),
        "positive_prompt": f(
            "positive_prompt",
            "pixel art, detailed pixel character sprite, sharp clean pixels, "
            "retro game character, high detail shading, crisp outlines",
        ),
        "negative_prompt": f(
            "negative_prompt",
            "blurry, smooth gradient, photo, realistic, 3d render, low quality, "
            "jpeg artifacts, anti-aliased",
        ),
    }

    prefix = f"frontend_{job_id}"
    workflow = build_workflow(input_name, params, prefix)

    try:
        resp = requests.post(f"{COMFY_URL}/prompt", json=workflow, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": f"Could not reach ComfyUI at {COMFY_URL}: {e}"}), 502

    resp_json = resp.json()
    if resp_json.get("node_errors"):
        return jsonify({"error": "ComfyUI rejected the workflow", "details": resp_json["node_errors"]}), 400
    prompt_id = resp_json["prompt_id"]

    # Poll for completion (up to ~5 minutes, to cover cold model loads)
    output_filenames = []
    for _ in range(300):
        time.sleep(1)
        try:
            hist = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10).json()
        except requests.RequestException:
            continue
        if prompt_id in hist:
            outputs = hist[prompt_id].get("outputs", {})
            for node_out in outputs.values():
                for image in node_out.get("images", []):
                    output_filenames.append(image["filename"])
            break

    if not output_filenames:
        return jsonify({"error": "Generation timed out or produced no image"}), 504

    entry = {
        "id": job_id,
        "input_image": f"/input-image/{input_name}",
        "output_images": [f"/output-image/{name}" for name in output_filenames],
        "params": params,
    }
    HISTORY.insert(0, entry)
    return jsonify(entry)


@app.route("/history")
def history():
    return jsonify(HISTORY)


@app.route("/input-image/<path:filename>")
def serve_input(filename):
    return send_from_directory(COMFY_INPUT_DIR, filename)


@app.route("/output-image/<path:filename>")
def serve_output(filename):
    return send_from_directory(COMFY_OUTPUT_DIR, filename)


def _basename_from_url(value: str) -> str:
    """Turn a served image URL (possibly absolute, possibly cache-busted with
    ?t=...) back into a bare filename, safely discarding any path traversal."""
    return Path(str(value).split("?", 1)[0]).name


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


QA_MAX_TOKENS = 700
# CloudIQ intermittently crashes (returns a raw, unhandled-exception HTML page)
# on this request shape (long prompt + 2 images); retrying the identical
# request often succeeds, so retry a couple of times before giving up.
QA_MAX_ATTEMPTS = 3
QA_RAW_DISPLAY_LIMIT = 300

# The verdict line is asked for FIRST, before any explanation. Some of CloudIQ's
# models "think out loud" in the visible reply and can run out of max_tokens
# mid-thought - putting VERDICT/REASON last risked it never being emitted at all.
QA_PROMPT = (
    "You are QA-checking an AI-detailed pixel-art game sprite against its original "
    "source image. The FIRST image is the ORIGINAL sprite. The SECOND image is the "
    "DETAILED version, generated to add shading, sharpness and extra pixel detail "
    "while intending to keep the exact same character design.\n\n"
    "Judge only whether the DETAILED version preserved the original's silhouette, "
    "pose, color palette, and identifying features. Added shading, sharper edges, "
    "and extra detail are expected and should NOT count against it — only flag it "
    "if the character/object design itself changed (different colors, proportions, "
    "pose, or missing/added features).\n\n"
    "Answer with the verdict line FIRST, before any explanation. Do not write "
    "anything before it. Use exactly this format:\n"
    "VERDICT: PRESERVED or DRIFTED\n"
    "REASON: <one short sentence>"
)


def _parse_qa_reply(reply: str):
    verdict_match = re.search(r"VERDICT:\s*(PRESERVED|DRIFTED)", reply, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+)", reply, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).upper()
        reason = reason_match.group(1).strip() if reason_match else ""
        return verdict, reason

    # Some models "think out loud" instead of following the exact VERDICT:/REASON:
    # format - fall back to whichever of the two words appears last (the
    # conclusion typically comes after the reasoning, not before it).
    loose_matches = list(re.finditer(r"\b(PRESERVED|DRIFTED)\b", reply, re.IGNORECASE))
    if loose_matches:
        verdict = loose_matches[-1].group(1).upper()
        reason = reply.strip()
        if len(reason) > QA_RAW_DISPLAY_LIMIT:
            reason = reason[:QA_RAW_DISPLAY_LIMIT].rstrip() + "…"
        return verdict, reason

    reason = reply.strip()
    if len(reason) > QA_RAW_DISPLAY_LIMIT:
        reason = reason[:QA_RAW_DISPLAY_LIMIT].rstrip() + "…"
    return "UNKNOWN", reason


def _call_cloudiq_vision(messages, max_tokens, max_attempts=QA_MAX_ATTEMPTS):
    """POST to CloudIQ's chat/completions with a small automatic retry.

    Empirically, CloudIQ occasionally returns a raw, unhandled-exception HTML
    page (not its usual sanitized JSON error) for multimodal requests - retrying
    the identical request often succeeds. A clean JSON error (bad key, rate
    limit, etc.) is NOT retried since that won't change on a retry.
    """
    last_error = "No response from CloudIQ"
    for _ in range(max_attempts):
        try:
            resp = requests.post(
                f"{CLOUDIQ_URL}/v1/chat/completions",
                headers={"X-API-Key": CLOUDIQ_API_KEY, "Content-Type": "application/json"},
                json={"messages": messages, "max_tokens": max_tokens},
                timeout=90,
            )
        except requests.RequestException as e:
            last_error = f"Could not reach CloudIQ at {CLOUDIQ_URL}: {e}"
            continue

        if resp.ok:
            try:
                return resp.json(), None
            except ValueError:
                last_error = "CloudIQ returned a malformed response"
                continue

        try:
            detail = resp.json().get("error", resp.text)
        except ValueError:
            last_error = "CloudIQ hit an internal error and returned a non-JSON response"
            continue

        return None, f"CloudIQ QA check failed: {detail}"

    return None, last_error


@app.route("/qa-check", methods=["POST"])
def qa_check():
    if not CLOUDIQ_API_KEY:
        return jsonify({"error": "CLOUDIQ_API_KEY is not set. Add it to a .env file to enable QA checks."}), 500

    data = request.get_json(silent=True) or {}
    input_name = _basename_from_url(data.get("input_image", ""))
    output_name = _basename_from_url(data.get("output_image", ""))
    if not input_name or not output_name:
        return jsonify({"error": "input_image and output_image are required"}), 400

    input_path = COMFY_INPUT_DIR / input_name
    output_path = COMFY_OUTPUT_DIR / output_name
    if not input_path.is_file() or not output_path.is_file():
        return jsonify({"error": "Could not find one of the images on disk"}), 404

    try:
        before_data_url = _image_data_url(input_path)
        after_data_url = _image_data_url(output_path)
    except OSError as e:
        return jsonify({"error": f"Could not read image: {e}"}), 500

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QA_PROMPT},
                {"type": "image_url", "image_url": {"url": before_data_url}},
                {"type": "image_url", "image_url": {"url": after_data_url}},
            ],
        }
    ]

    resp_json, error = _call_cloudiq_vision(messages, QA_MAX_TOKENS)
    if error:
        return jsonify({"error": error}), 502

    try:
        reply = resp_json["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        reply = ""

    verdict, reason = _parse_qa_reply(reply)
    return jsonify({
        "verdict": verdict,
        "reason": reason,
        "raw": reply,
        "model": resp_json.get("served_model"),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
