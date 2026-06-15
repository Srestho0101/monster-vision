// ─────────────────────────────────────────────────────────────────────────────
// DRONE VISION CONFIG
// ─────────────────────────────────────────────────────────────────────────────
// Edit this file when you change models, add classes, or adjust defaults.
// No need to touch the HTML file.
// ─────────────────────────────────────────────────────────────────────────────

const CONFIG = {
  // GitHub release URL to your .onnx model
  // Example: 'https://github.com/yourname/yolo-drone/releases/download/v1.0/model.onnx'
  modelUrl: '',

  // Class names (in the same order as your model was trained)
  classNames: ['target'],

  // Input size the model expects (typically 640 for YOLOv8n)
  inputSize: 640,

  // Default inference thresholds
  confidence: 0.40,
  iou: 0.45,

  // UI defaults
  showLabels: true,
  showGrid: false,
  showLatency: false,

  // Camera defaults
  facingMode: 'environment', // 'environment' (rear) or 'user' (front)
};
