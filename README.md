# Monster Vision — YOLOv8 Live Object Detection in Browser

A real-time object detection web app that runs a finetuned YOLOv8n model directly in your phone's browser. No backend server needed. Test your custom YOLO models instantly across different lighting, angles, and real-world conditions.

**Live Status:** ✅ Working — 0.3–0.5 FPS on mobile (4GB Android), 1.1 FPS on 4GB Linux laptop

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Architecture & How It Works](#architecture--how-it-works)
3. [Setup Guide](#setup-guide)
4. [Key Concepts You Should Know](#key-concepts-you-should-know)
5. [Common Issues & Solutions](#common-issues--solutions)
6. [Performance Optimization Tips](#performance-optimization-tips)
7. [Next Steps & Learning Path](#next-steps--learning-path)
8. [Project Structure](#project-structure)

---

## What This Project Does

You've trained a custom YOLOv8n model to detect "Spiderman figure" objects from 116 images using Roboflow. Now you want to **test it in the real world** before deploying to your drone's ESP32-CAM.

This project lets you:
- **Load your ONNX model** from GitHub directly into the browser
- **Capture live video** from your phone camera
- **Run inference** (object detection) in real-time using WebAssembly
- **Visualize detections** with bounding boxes, class names, and confidence scores
- **Monitor performance:** FPS, inference latency, detection count
- **Adjust thresholds** on-the-fly (confidence, IoU) without rebuilding
- **Cache the model** locally so it loads instantly next time
- **Test different conditions:** lighting, angles, distances, obstacles

All **without a server**. The model runs entirely in your browser's JavaScript engine.

---

## Architecture & How It Works

### The Big Picture

```
[Your Phone Camera]
        ↓
   [Browser Video Stream]
        ↓
   [YOLOv8n ONNX Model]
   (WebAssembly runtime)
        ↓
   [Detection Results]
        ↓
   [Draw Boxes on Canvas]
```

### How Each Piece Works

#### 1. **Model Loading & Caching (Browser Storage)**

Your model is large (~20MB unquantized). Downloading it every time you open the page is slow.

**Solution:** IndexedDB (browser's local key-value database)

- First load: Download from GitHub → Parse ONNX → Cache in IndexedDB → Show in browser
- Subsequent loads: Check IndexedDB → If found, use cached version → Skip download

This is why you see "Checking cache…" on startup.

**Why not `localStorage`?** It's limited to ~5-10MB per domain. IndexedDB can store 50MB+ depending on the browser.

#### 2. **Model Serving from GitHub (CORS Problem)**

GitHub's release CDN **blocks cross-origin requests** by default (CORS policy). When your browser tries to fetch the model from a different origin, it gets blocked.

**Solution:** jsDelivr CDN proxy

jsDelivr mirrors GitHub repos and adds proper `Access-Control-Allow-Origin` headers. So instead of:

```
❌ https://github.com/user/repo/releases/download/v1/model.onnx
✅ https://cdn.jsdelivr.net/gh/user/repo@branch/path/model.onnx
```

The jsDelivr URL works because jsDelivr *adds* the correct CORS headers that tell the browser "yes, it's safe to use this file cross-origin."

#### 3. **ONNX Runtime Web (Inference Engine)**

ONNX is a standard format for ML models (works with TensorFlow, PyTorch, etc.). ONNX Runtime Web is a WebAssembly library that runs ONNX models in the browser using JavaScript.

**Why ONNX and not PyTorch?**
- PyTorch.js exists but is clunky and slow for mobile
- ONNX Runtime is optimized for WebAssembly and mobile browsers
- ONNX is framework-agnostic (you can train in PyTorch, export to ONNX, run anywhere)

**Why WebAssembly?**
- JavaScript is too slow for real-time ML inference
- WebAssembly (WASM) is compiled code that runs near-native speed in the browser
- It gets automatic multi-threading, memory optimization, etc.

#### 4. **Video Stream → Model Input**

Your phone camera sends raw video frames. YOLO expects normalized RGB tensors (0–1 range), so we:

1. **Resize** the frame to 640×640 (your model's input size)
2. **Normalize** pixel values: divide by 255 to get 0–1 range
3. **Reorder channels** from RGBA to RGB
4. **Create tensor** in the correct shape `[1, 3, 640, 640]` (batch=1, channels=3, height=640, width=640)

This happens in `preprocess()`.

#### 5. **Model Output → Detections**

YOLOv8n outputs a tensor with shape `[1, 84, 8400]` (for 640×640 input):
- **1:** batch size (just 1 image)
- **84:** 4 bbox coords (x, y, w, h) + 1 object confidence + 79 class probabilities
- **8400:** anchor points across the grid

We convert this to human-readable bounding boxes:

1. **Parse anchors** and get class predictions (which class is most confident)
2. **Filter by confidence threshold** (default 0.4) — discard weak detections
3. **Apply NMS (Non-Maximum Suppression)** — if two boxes overlap too much (IoU > 0.45), keep only the highest-confidence one. This prevents duplicate detections.
4. **Scale coordinates** back to video frame size (model input is 640×640, but video might be 1280×720)

This happens in `postprocess()`.

#### 6. **Rendering (Canvas)**

Once we have bounding boxes, we draw them on an HTML5 canvas overlaid on the video:
- Colored rectangles with corner ticks (YOLOv8 style)
- Class name + confidence label
- "Flash" border when detection fires (visual feedback)

---

## Setup Guide

### Prerequisites

You need:
- A GitHub account (to store the model)
- A text editor (VS Code, Notepad++, etc.)
- A phone with a camera and a modern browser (Chrome, Firefox, Safari — all have WebAssembly support)

### Step 1: Prepare Your Model

1. **Train in Google Colab** (you already did this):
   - Use Roboflow for labeling
   - Train YOLOv8n for 4+ epochs
   - Export as ONNX (non-quantized)

2. **Export from PyTorch to ONNX** (in Colab):
   ```python
   from ultralytics import YOLO
   model = YOLO('path/to/best.pt')
   model.export(format='onnx', imgsz=640)
   ```
   This creates `best.onnx` (~20MB for YOLOv8n).

3. **Why not quantized?**
   - Dynamic quantization → `ConvInteger` operator → Not supported by ONNX Runtime Web
   - Static quantization → Requires calibration → More complex setup
   - For now: Unquantized ONNX is 20MB (still small), loads in ~5 seconds on 4G
   - **Later:** Learn about static quantization or distillation if you need faster inference

### Step 2: Upload to GitHub

1. Create a GitHub repo: `monster-vision`
2. Create a folder: `models/`
3. Commit your `.onnx` file into `models/`
4. Push to GitHub
5. Your repo is now the source of truth. Every time you retrain, upload a new ONNX here.

### Step 3: Deploy the Web App

1. Download both files from this project:
   - `config.js`
   - `yolo-live.html`

2. Put them in your repo root (or any accessible folder)

3. Open `yolo-live.html` in your phone browser:
   - Locally: File → Open File → yolo-live.html
   - Or host on GitHub Pages for easy sharing

4. **Allow camera permission** when the browser asks

### Step 4: Configure

Edit `config.js`:

```javascript
const CONFIG = {
  modelUrl: 'https://cdn.jsdelivr.net/gh/YOUR_USERNAME/monster-vision@main/models/YOUR_MODEL.onnx',
  classNames: ['Spiderman figure'],  // or multiple: ['class1', 'class2', ...]
  inputSize: 640,
  confidence: 0.40,
  iou: 0.45,
  showLabels: true,
  showGrid: false,
  showLatency: false,
  facingMode: 'environment',
};
```

- **modelUrl:** Full jsDelivr URL to your ONNX file
- **classNames:** Match the order from your training labels
- **inputSize:** What size you trained at (usually 640)
- **confidence:** Lower = more detections (higher false positives); Higher = fewer detections (might miss real objects)
- **iou:** NMS threshold — how much overlap before merging boxes

### Step 5: Test

1. Open `yolo-live.html` on your phone
2. Point at your Spiderman figure
3. Watch the FPS counter and detection count
4. Adjust sliders to see how confidence/IoU affect results

---

## Key Concepts You Should Know

### 1. **ONNX Format**

ONNX stands for "Open Neural Network Exchange." It's a standardized format that lets you:
- Train in **any framework** (PyTorch, TensorFlow, JAX, etc.)
- Export to ONNX (one common format)
- Run anywhere (mobile, browser, edge devices, cloud)

**Why it matters:** Your Roboflow-trained YOLOv8 model is framework-agnostic via ONNX. You can:
- Run it in Python (ONNX Runtime)
- Run it in browser (ONNX Runtime Web)
- Run it on ESP32 (later, with TensorFlow Lite or ONNX Runtime Lite)
- Run it in C++ (for faster inference)

### 2. **Quantization (Static vs Dynamic)**

**Dynamic Quantization** (what you tried):
- Post-training: convert fp32 → int8 without a calibration set
- Faster inference, smaller file
- **Problem:** Changes architecture (uses `ConvInteger` which ONNX Runtime Web doesn't support)

**Static Quantization**:
- Uses representative data to calibrate thresholds
- Converts to int8 *and* keeps supported operators
- Slightly better accuracy/speed tradeoff
- **You'll learn this later** once you need smaller models

### 3. **Confidence vs IoU**

- **Confidence threshold:** "How sure does the model need to be to say 'yes, that's a Spiderman'?" (0–1)
  - Too low (0.1) → Everything is a detection, lots of false positives
  - Too high (0.9) → Model is too picky, misses real objects
  - **Sweet spot:** Usually 0.3–0.5

- **IoU threshold (NMS):** "If two boxes overlap more than X%, keep only the high-confidence one"
  - IoU = Intersection over Union (0–1)
  - 0.45 is a standard balance

### 4. **WebAssembly & Why It Matters**

JavaScript is interpreted and slow. WebAssembly (WASM) is:
- Compiled bytecode that runs near-native speed
- Supported in all modern browsers
- Can use multiple threads (if enabled)

**Your inference runs in WASM**, which is why you get 0.3–0.5 FPS on mobile. If it ran in pure JavaScript, it'd be 10× slower.

### 5. **CORS (Cross-Origin Resource Sharing)**

When your page (origin A) tries to fetch a resource from origin B, the browser checks CORS headers:

```
Your page:    https://mysite.com/index.html
Model URL:    https://github.com/user/repo/releases/download/model.onnx

Browser says: "Is github.com OK with mysite.com fetching this?"
```

GitHub's CDN doesn't send the right headers, so the browser blocks it. jsDelivr does, so it works.

---

## Common Issues & Solutions

### Issue 1: "HTTP 404" When Loading Model

**Cause:** URL path doesn't match actual file path in repo

**Fix:**
- Check exact filename in GitHub (case-sensitive!)
- Use jsDelivr format: `https://cdn.jsdelivr.net/gh/user/repo@branch/path/file.onnx`
- Verify the branch name (usually `main` or `master`)

### Issue 2: "Can't create a session. ERROR_CODE: 9"

**Cause:** Model uses an operator ONNX Runtime Web doesn't support (e.g., `ConvInteger` from dynamic quantization)

**Fix:**
- Export ONNX **without quantization**
- Or use static quantization (more complex, learn later)

### Issue 3: "Network error" / "Failed to fetch"

**Cause:** Usually CORS (see Issue 1), or model file too large

**Fix:**
- Use jsDelivr proxy (not raw GitHub URLs)
- Check model file size (~20MB for unquantized YOLOv8n is OK)
- Check browser console for exact error message

### Issue 4: Very Low FPS (0.1–0.3)

**Cause:** Mobile browser WebAssembly isn't optimized for mobile yet, or model is too large

**Fix:**
- This is normal for browser-based inference on mobile
- Later: Switch to native inference (Android/iOS app, or ESP32-CAM firmware)
- For testing, use a laptop browser (you got 1.1 FPS on 4GB laptop)

### Issue 5: Model Loads But Detections Are Terrible

**Cause:** Training dataset mismatch (your training images ≠ real-world conditions)

**Fix:**
- This is why you're doing this test! Collect bad examples
- Retrain with more diverse data (different lighting, angles, distances)
- Lower confidence threshold to see if model is detecting but too conservative
- Check if your Roboflow labels are correct

---

## Performance Optimization Tips

### For Browser Inference

1. **Increase input size gradually**
   - You trained at 640×640 (standard)
   - If you retrain later at 320×320, inference speeds up 4× (but accuracy drops)
   - 512×512 is a middle ground

2. **Use static quantization (future)**
   - Dynamic quantization doesn't work in browser
   - Static quantization can reduce model to 5MB and speed up 2–3×
   - Worth learning once you're ready to optimize

3. **Batch multiple frames (advanced)**
   - Currently: Process 1 frame → Get detection
   - Later: Process 4 frames together → Still fast but higher throughput
   - Only useful if you're streaming to a server

4. **Use worker threads (advanced)**
   - ONNX Runtime Web can use Web Workers
   - Prevents model inference from blocking UI
   - You'll learn this if FPS becomes critical

### For Drone Deployment (ESP32-CAM)

- ESP32 doesn't run browser code
- You'll use **TensorFlow Lite** or **ONNX Runtime Lite** (C/C++)
- TensorFlow Lite is optimized for ARM processors
- Static quantization becomes **essential** (model must fit in ESP32's 4MB PSRAM)

---

## Next Steps & Learning Path

### Immediate (1–2 weeks)
- [ ] Test current model in various conditions (lighting, distance, angles)
- [ ] Collect images where model fails (false positives, false negatives)
- [ ] Retrain with this new data to improve accuracy
- [ ] Re-export ONNX and test in browser again

### Short-term (1–2 months)
- [ ] Learn **TensorFlow Lite** (for ESP32 deployment)
  - Tutorial: https://www.tensorflow.org/lite/guide
  - Focus on: Converting PyTorch → TFLite, quantization strategies
  
- [ ] Learn **static quantization**
  - For smaller, faster models in browser
  - Tutorial: https://onnxruntime.ai/docs/performance/quantization/
  
- [ ] Learn **Docker** (for reproducible training)
  - Why: Your Colab environment is ephemeral (deleted after disconnect)
  - Docker lets you create a local training environment that's identical to production
  - Guides: Docker official docs, "Docker for ML"

### Medium-term (2–6 months)
- [ ] **Learn embedded deployment** (C/C++)
  - Build a local inference pipeline that doesn't depend on cloud
  - Framework: ONNX Runtime Lite, TensorFlow Lite
  - Deploy to your ESP32-CAM once it arrives
  
- [ ] **Learn model optimization**
  - Pruning (remove unimportant weights)
  - Distillation (train a smaller model to mimic a larger one)
  - Tools: PyTorch, TensorFlow quantization APIs
  
- [ ] **Learn CI/CD pipelines** (GitHub Actions)
  - Automatically retrain and export when you push new data
  - Auto-upload ONNX to releases
  - Skip manual steps

### Long-term (6+ months)
- [ ] **Learn Hugging Face Hub**
  - Store models, datasets, and notebooks in one place
  - Community sharing, versioning, easy deployment
  - Useful once your model is production-ready
  
- [ ] **Learn advanced CV** (depthwise-separable convolutions, attention mechanisms)
  - Build custom architectures instead of using pre-built YOLOv8
  - Understand what makes models fast vs accurate
  
- [ ] **Learn edge AI frameworks**
  - OpenVINO (Intel), TVM (Apache), Qualcomm Snapdragon
  - Deploy to different hardware (Raspberry Pi, Jetson, etc.)

### Recommended Learning Resources

| Topic | Resource | Time |
|-------|----------|------|
| ONNX format | https://onnx.ai/onnx/intro/ | 1–2 hours |
| TensorFlow Lite | https://www.tensorflow.org/lite/guide | 2–3 days |
| Docker basics | "Docker for Beginners" (YouTube) | 1–2 days |
| PyTorch quantization | https://pytorch.org/docs/stable/quantization.html | 3–4 days |
| Hugging Face | https://huggingface.co/course | 1 week |
| ONNX Runtime Lite | https://onnxruntime.ai/docs/install/ | 2–3 days |

---

## Project Structure

```
monster-vision/
├── models/
│   └── tmv-quantized_dynamic.onnx    # Your trained model (20MB)
├── config.js                          # Configuration (edit this!)
├── yolo-live.html                     # Web app (don't edit, use config.js)
├── README.md                          # This file
└── (Later: Training notebook, dataset info, etc.)
```

### How to Update the Model

1. Retrain in Google Colab with new data
2. Export to ONNX (non-quantized)
3. Commit to `models/` folder
4. Push to GitHub
5. Page automatically loads the new model (jsDelivr checks for updates)

**No code changes needed.** Just update the files.

---

## Summary

You've built a **mobile-first ML testing platform** that:
- ✅ Trains locally (Google Colab)
- ✅ Serves models decoupled from code (GitHub + jsDelivr)
- ✅ Runs inference in the browser (ONNX Runtime Web)
- ✅ Caches for fast iteration (IndexedDB)
- ✅ Requires zero backend server

This is a **perfect foundation** for testing before drone deployment. The knowledge you gain here (ONNX, quantization, inference optimization) directly applies to ESP32 and production systems.

**Next milestone:** When your ESP32-CAM arrives, you'll take the same `.onnx` model, convert to TensorFlow Lite, and deploy it using the same testing methodology you're building now.

---

## Questions?

- **Model issues?** Check the `.onnx` export process in Colab
- **Browser issues?** Check browser console (F12 → Console tab) for error messages
- **Performance?** Mobile WebAssembly is inherently slow; this is expected for real-time inference
- **Need help with next steps?** Search "[technology name] for machine learning" + "tutorial"

Good luck testing, and enjoy watching your Spiderman detector work! 🕷️
