"""
Sprite Detail Frontend - a minimal local UI for the ComfyUI sprite-upscale workflow.

Talks to a running ComfyUI server (http://127.0.0.1:8188) on the user's behalf so the
browser never has to call ComfyUI's API directly (ComfyUI doesn't send CORS headers).
"""
import io
import json
import time
import uuid
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_from_directory, render_template

COMFY_URL = "http://127.0.0.1:8188"
COMFY_INPUT_DIR = Path.home() / "ComfyUI" / "input"
COMFY_OUTPUT_DIR = Path.home() / "ComfyUI" / "output"

CHECKPOINT = "sd_xl_base_1.0.safetensors"
LORA = "pixel-art-xl.safetensors"
CONTROLNET = "controlnet-canny-sdxl-fp16.safetensors"

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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
