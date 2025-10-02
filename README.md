# 🖼️ Image Captioning with Attention (PyTorch)

Implementation of the **“Show, Attend and Tell”** paper (Xu et al., 2015) using PyTorch.  
The model generates natural language captions for images using an **encoder–decoder with attention**.

---

## ⚡ Features

- **Encoder:** ResNet50 (default) or VGG19 extracts spatial image features.
- **Attention:** Additive attention mechanism to focus on informative image regions.
- **Decoder:** LSTM with gating and attention to generate captions.
- **Data Preprocessing:**
  - Converts Flickr-style `captions.txt` → `captions.json`.
  - Splits into train/test datasets.
- **Training Pipeline:**  
  Automated cleanup, preprocessing, training, and checkpointing (`run_pipeline.py`).
- **Evaluation:**  
  - `eval.py` computes BLEU-1 to BLEU-4 scores.  
  - `test.py` generates captions on the test split.  
- **Visualization:** `visualize_captions.py` shows random images with predicted captions.  
- **Model Summary:** `print_modelsummary.py` prints encoder/decoder architecture with parameters.  
- **Results:** Captions and metrics saved in `experiments/results/`.

---

## 🛠 Installation

```bash
# Clone repository
git clone <repo_url>
cd image_caption

# Create virtual environment
python3 -m venv venv
source venv312/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt


🚀 Usage
1️⃣ Preprocess + Train
    python3.12 run_pipeline.py
2️⃣ Evaluate (BLEU scores)
    python3.12 eval.py --checkpoint experiments/checkpoints/latest.pth --split test
3️⃣ Generate Captions (Test)
    python3 test.py
4️⃣ Visualize Captions
    python3 visualize_captions.py
5️⃣ Model Summary
    python3 print_modelsummary.py

