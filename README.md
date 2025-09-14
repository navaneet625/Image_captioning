image-caption-simple/
├─ README.md                 # quick run + notes
├─ requirements.txt
├─ configs.yaml              # small set of hyperparams
├─ data/
│  ├─ images/                # put your JPG/PNG images here
│  └─ captions.json          # {"img1.jpg": ["a cat ..."], ...}
├─ features/                 # optional: pre-extracted features (.pt)
├─ src/
│  ├─ __init__.py
│  ├─ dataset.py             # tiny PyTorch Dataset (loads features or raw image)
│  ├─ encoder.py             # wrapper returning spatial features (14x14xC)
│  ├─ attention.py           # single-file attention module
│  ├─ decoder.py             # LSTM decoder + embedding + output layer
│  ├─ train.py               # minimal training loop + save checkpoint
│  └─ infer.py               # load checkpoint -> generate caption for one image
└─ example.jpg               # demo image for quick test
|__train.py
|
|___print_modelsummary.py
