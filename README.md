# LocalSpriteDetailer

A local web UI for turning a low-res pixel-art sprite into a higher-detail version of
the *same* design, using ComfyUI (Stable Diffusion XL + ControlNet + a pixel-art LoRA)
under the hood, without hand-editing JSON workflows.

This tool is **project-agnostic** — it doesn't know or care which game/Unity project a
sprite came from. You upload any image file, it runs the pipeline, you get a result back.
That means it's already reusable as-is for any future project on this machine; nothing
here is tied to Outlanders specifically.

## Prerequisites (one-time setup, already done on this machine)

- ComfyUI installed at `~\ComfyUI` (your Windows user profile), with these models in place:
  - `models/checkpoints/sd_xl_base_1.0.safetensors`
  - `models/loras/pixel-art-xl.safetensors`
  - `models/controlnet/controlnet-canny-sdxl-fp16.safetensors`
- This frontend, with its own venv (Flask + requests).

If you ever set this up on a **different machine**, you'd need to redo the ComfyUI
install (see the ComfyUI section of this project's chat history / ask for the steps
again) and re-point the paths in `app.py` (see "Changing paths or models" below).

## How to launch

Two servers need to be running. Each is a separate terminal/background process.

```
# 1. Start ComfyUI (the actual image-generation engine)
%USERPROFILE%\ComfyUI\venv\Scripts\python.exe %USERPROFILE%\ComfyUI\main.py

# 2. Start this frontend (from this project's directory)
venv\Scripts\python.exe app.py
```

Then open **http://127.0.0.1:5050** in your browser.

Neither server auto-starts on login or survives a reboot — you relaunch both manually
each time. If you want that to be less annoying later, this could be wired into a
Windows scheduled task or a simple `.bat` shortcut; ask if you want that set up.

To check if either is already running before starting it again:
```
curl http://127.0.0.1:8188   # ComfyUI
curl http://127.0.0.1:5050   # this frontend
```

## Using it

1. Upload any sprite image (works on anything, not just this game's art).
2. Adjust the settings:
   - **Denoise** — how much the output is allowed to change from the source. Lower =
     closer to the original, higher = more reinterpreted. Start around 0.5–0.6.
   - **ControlNet strength** — how tightly the output has to match the original
     silhouette/edges. Higher = more faithful shape, lower = more creative freedom.
   - **LoRA strength** — how strongly the pixel-art style is applied.
   - **Prompt / negative prompt** — text guidance, same idea as any SD-based tool.
   - **Steps / cfg / seed** — standard diffusion sampling controls.
3. Hit Generate, compare before/after, tweak, repeat. Past attempts in the session
   show up in the history strip so you can click back to compare.

First generation after starting ComfyUI is slow (~2 minutes) because it has to load
the model into VRAM. Every generation after that is much faster.

## Changing paths or models later

Everything project-specific lives at the top of `app.py`:

```python
COMFY_URL = "http://127.0.0.1:8188"
COMFY_INPUT_DIR = Path.home() / "ComfyUI" / "input"
COMFY_OUTPUT_DIR = Path.home() / "ComfyUI" / "output"
CHECKPOINT = "sd_xl_base_1.0.safetensors"
LORA = "pixel-art-xl.safetensors"
CONTROLNET = "controlnet-canny-sdxl-fp16.safetensors"
```

To use a different checkpoint/LoRA/ControlNet (e.g. a different art style for a
different project), drop the new `.safetensors` file into the matching ComfyUI
`models/` subfolder and update the constant here to match its filename.

## Known limitations

- History is in-memory only — resets when `app.py` restarts. Nothing is deleted from
  disk though; every generated image stays in `~\ComfyUI\output\`.
- Flask's built-in dev server is used — fine for this single-user local tool, not
  meant to be exposed beyond localhost.
- `server.log` / `server.err.log` in this folder are just runtime logs from manual
  testing, safe to delete or ignore.
