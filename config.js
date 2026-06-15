// https://github.com/Srestho0101/monster-vision/blob/main/models/tmv-un.onnx

const CONFIG = {
  // jsDelivr CDN URL pointing to your model in the GitHub repo
  // Format: https://cdn.jsdelivr.net/gh/username/repo@branch/path/to/file.onnx
  modelUrl: 'https://cdn.jsdelivr.net/gh/Srestho0101/monster-vision@main/models/tmv-un.onnx',

  // Class names (in the same order as your model was trained)
  classNames: ['Spiderman figure'],

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
