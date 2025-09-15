
---

## ⚡ Features

- **Encoder:** ResNet50 extracts spatial image features.
- **Attention:** Additive attention to focus on image regions.
- **Decoder:** LSTM generates captions using attention-weighted features.
- **Data preprocessing:** Converts `captions.txt` → `captions.json` and splits dataset into train/val/test.
- **Pipeline:** Automated cleanup, preprocessing, training, and checkpointing.
- **Results:** Generated captions saved in `results/generated_captions.json`.
- **Visualization:** `visualize_captions.py` shows random images with predicted captions.
- **Testing:** `test.py` evaluates model performance on the test set.

---

## 🛠 Installation

1. Clone the repository:
```bash
git clone <repo_url>
cd image_caption


python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows


pip install --upgrade pip
pip install -r requirements.txt


* Adjust training parameters in configs.yaml


python3 run_pipeline.py   
1. Training checkpoints are saved in experiments/checkpoints.
2. Generated captions appear in results/generated_captions.json.


python3 test.py
1. Loads the latest checkpoint (e.g., ckpt_epoch50.pth).
2. Generates captions for images in the test set.
3. Saves results to results/generated_captions.json.


python3 visualize_captions.py
1. Random images are displayed with their predicted captions.
2. Generated captions are stored in results/generated_captions.json.
