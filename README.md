# 🧠 Interactive CNN Explainer – Deep Learning Visualized

[![Visit to the site](https://img.shields.io/badge/demo-live-brightgreen)](https://bytecrister.github.io/cnn-acrhitecture/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A **fully client‑side**, interactive educational tool that lets you **see inside a Convolutional Neural Network (CNN)** while it learns to classify handwritten digits (0‑9). Every operation – convolution, activation, pooling, gradient backpropagation – is visualised step‑by‑step, right in your browser.

> **Try it live:** [https://bytecrister.github.io/cnn-acrhitecture/](https://bytecrister.github.io/cnn-acrhitecture/)

---

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/ByteCrister/neural-knowledge-base.git
   cd neural-knowledge-base
   ```
2. **Open `index.html`** in any modern browser (Chrome, Firefox, Edge).
   _No server, no build step – it works straight from the file system._

That’s it! The app loads TensorFlow.js and Chart.js from CDNs, downloads a small digit dataset, and you’re ready to explore.

---

## ✨ Features

### 🔬 **CNN Basics – Interactive Demos**
- **Convolution** – edit a 5×5 input and a 3×3 kernel, watch the sliding window, adjust stride and padding.
- **Activation Functions** – pass a matrix through ReLU / Sigmoid and see the before/after.
- **Pooling** – apply max‑pooling on a 4×4 grid with animation.
- **Flatten & Dense** – visualise how a 2D feature map becomes a flat vector fed into a dense layer.

### 📊 **Dataset Exploration**
- Automatically loads a collection of digit images (28×28 grayscale).
- Browse a grid of samples with their labels – click to view batches.

### 🏗️ **Model Architecture**
A live block‑diagram of the CNN, showing each layer:
- Conv2D (8 filters, 3×3, ReLU) → MaxPool (2×2)
- Conv2D (16 filters, 3×3, ReLU) → MaxPool (2×2)
- Flatten → Dense (64, ReLU) → Dense (10, softmax)

The number of trainable parameters is displayed per layer.

### ⚡ **Training with Full Transparency**
- **Configurable training** – Epochs, batch size, learning rate (log scale).
- Real‑time charts (loss & accuracy) using Chart.js.
- **Progress bar** per epoch.
- **Pause/Resume** and **Reset** buttons.

### 🧐 **“Inspect One Training Step” – Under‑the‑Hood Wizardry**
This is the heart of the explainer. At any moment you can **freeze** training and see **every single detail** for a real training example:
- Input image as a 28×28 heatmap.
- All kernel weights (heatmaps) before the convolution.
- Feature maps before & after activation, after pooling.
- Flattened vector, dense layer outputs, and final softmax probabilities.
- **Gradients** of the weights (heatmaps) – you literally see backpropagation.
- Weight updates (before/after diff) for a subset of parameters.
- Loss computation broken down (cross‑entropy formula).

### ✏️ **Draw & Predict**
- A **280×280 drawing canvas** – draw your own digit and get an instant, transparent prediction.
- See the exact same layer‑by‑layer visualisation as in the training inspector.
- **Clear** button and a **Random Test Image** option to test from the dataset.

### 🎨 **Modern Dark UI**
- Smooth accordions, tooltips, high‑contrast heatmaps.
- Fully responsive design.

---

## 🕹️ Usage & Controls

| Action | How |
|--------|-----|
| Start / pause training | Click “Start / Continue” or press `Space` |
| Toggle training inspector | Enable checkbox “Inspect One Training Step” – training will pause after the first batch of each epoch |
| Draw a digit | Use mouse/touch on the canvas, then click “Predict” |
| Clear canvas | Click “Clear” or press `C` |
| Load random test image | Click “Random Test Image” in the Prediction section |
| Explore CNN basics | Expand the “CNN Basics” section and interact with the matricies / sliders |

---

## 📂 Repository Structure

```
neural-knowledge-base/
├── index.html        # Main app (HTML + inline CSS + JS)
├── style.css         # Additional styles (optional, linked from HTML)
├── app.js            # Core logic (TensorFlow.js, charts, visualisation)
└── README.md         # You are here
```

_All machine learning and rendering happens on the client – your data never leaves the browser._

---

## 🛠️ Built With

- **[TensorFlow.js](https://www.tensorflow.org/js)** – Training & inference in the browser
- **[Chart.js](https://www.chartjs.org/)** – Real‑time loss/accuracy charts
- **Canvas API** – Heatmaps, matrices, and drawing
- **Vanilla HTML/CSS/JS** – Zero frameworks, highly performant

---

## 🎓 Educational Value

This tool was designed with **teaching in mind**. Every matrix, kernel, feature map, and gradient is rendered as a colour‑coded heatmap – no black boxes. The “Inspect One Training Step” mode corresponds to a single forward+backward pass, showing exactly:

1. Forward propagation through all layers
2. Calculation of the loss
3. Backpropagation of errors (gradients)
4. Weight updates

Ideal for students, educators, and anyone curious about how CNNs really work under the hood.

---

## 🤝 Contributing

Found a bug, want to add a feature, or improve the documentation? Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- MNIST dataset (adapted for in‑browser use)
- TensorFlow.js team
- The vibrant open‑source ML community

---

**Made with ❤️ to make deep learning accessible to everyone.**