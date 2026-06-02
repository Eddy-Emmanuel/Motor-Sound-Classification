# 🔊 Motor Sound Classifier

> **Six deep learning architectures — GRU, LSTM, LSTM-GRU, and three custom Transformer variants — trained from scratch to classify industrial motor audio into three operational states.**

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit)](https://motor-sound-classification-syoldbw2wrk6rmcb2ckeei.streamlit.app/)
[![HuggingFace Weights](https://img.shields.io/badge/🤗%20Weights-Motor__Sound__Classifier-orange)](https://huggingface.co/Eddy-Emmanuel/Motor_Sound_Classifier)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)

---

## 🧭 Overview

This project classifies the operational state of an industrial motor from raw `.wav` audio. Six architectures are trained and compared end-to-end on log-mel spectrogram features extracted at 16 kHz. A **live Streamlit app** supports single-model inference, configurable ensemble inference, and full audio visualization.

| Class | Label | Description |
|---|---|---|
| `engine1_good` | ✅ Good | Motor running normally |
| `engine2_broken` | ⚠️ Broken | Motor with fault or damage |
| `engine3_heavyload` | 🔴 Heavy Load | Motor under heavy mechanical load |

---

## 🖥️ Live Demo

The app is deployed on HuggingFace Spaces:

**👉 [motor-sound-classification-syoldbw2wrk6rmcb2ckeei.streamlit.app](https://motor-sound-classification-syoldbw2wrk6rmcb2ckeei.streamlit.app/)**

Upload any `.wav` file and get instant predictions from all six models plus an ensemble view.

---

## 🏗️ Model Architectures

Six models are trained and compared, split across two families.

### Recurrent Family

| Model | Architecture | Checkpoint Size |
|---|---|---|
| **GRU** | 2-layer bidirectional GRU + LayerNorm | 26 MB |
| **LSTM** | 2-layer bidirectional LSTM + LayerNorm | 35 MB |
| **LSTM-GRU** | Interleaved BiLSTM → BiGRU × 2 + LayerNorm | 73 MB |

All recurrent models operate on mel frames as a sequence `(T, 80)`, use hidden size 512, and pool via the final hidden state.

### Transformer Family (AudioModel)

A shared **AudioEncoder** backbone with three positional encoding strategies:

```
Input (B, 1, T, 80)
  → Linear projection  → (B, 1, T, 1024)
  → 12 × AudioAttentionBlock  [MHA + PE + residual]
  → LayerNorm + MLP (feed-forward)
  → mean pool over T
  → Linear classifier  → 3 logits
```

| Model | PE Strategy | Checkpoint Size |
|---|---|---|
| **AudioModel (Rotational PE)** | RoPE applied per-head to Q and K vectors | 236 MB |
| **AudioModel (Sinusoidal PE)** | Fixed sinusoidal encoding added to input | 244 MB |
| **AudioModel (Relative PE)** | Learnable relative bias added to attention logits | 252 MB |

**Transformer config:** `embed_dim=1024 · num_heads=16 · n_blocks=12 · max_seq_len=2048`

---

## 🎛️ Audio Processing Pipeline

```
WAV file  (any sample rate, mono or stereo)
    ↓  Resample  →  16 000 Hz
    ↓  Mono mix  (mean over channels if stereo)
    ↓  MelSpectrogram  (n_fft=1024, hop=512, n_mels=80)
    ↓  log1p  (dynamic range compression)
    ↓  z-score normalisation  (per clip, std + 1e-1 for stability)
    →  Tensor  (1, T, 80)  →  all six models
```

---

## 🗂️ Repository Structure

```
Motor-Sound-Classification/         ← HuggingFace Space (this repo)
├── app.py                          # Streamlit inference app
├── models/
│   └── load_model.py               # Downloads weights from HF Hub
├── inference-notebook.ipynb        # Step-by-step inference walkthrough
└── motor-sound-classification.ipynb  # Full training notebook

Eddy-Emmanuel/Motor_Sound_Classifier  ← Weights repo (separate HF repo)
├── gru_model.pt
├── lstm_model.pt
├── lstm_gru_model.pt
├── rotational_pe.pt
├── sinusodal_pe.pt
└── relative_pe.pt
```

---

## 🚀 Run Locally

### 1. Clone the Space repo

```bash
git clone https://huggingface.co/spaces/Eddy-Emmanuel/Motor-Sound-Classification
cd Motor-Sound-Classification
pip install -r requirements.txt
```

### 2. Set up your HuggingFace token

Create a `.env` file in the project root:

```bash
HF_TOKEN=hf_your_token_here
```

Model weights (~866 MB total) are downloaded automatically from `Eddy-Emmanuel/Motor_Sound_Classifier` on first run via `models/load_model.py`:

```python
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

local_dir = snapshot_download(
    repo_id="Eddy-Emmanuel/Motor_Sound_Classifier",
    allow_patterns="*.pt"
)
```

### 3. Launch

```bash
streamlit run app.py
```

---

## 🖼️ App Features

### 📊 Audio Visualization
Three-panel signal analysis for any uploaded WAV:
- **Waveform** — raw amplitude over time
- **STFT Spectrogram** — power in dB over linear frequency
- **Mel Spectrogram** — normalized log-mel energy over 80 mel bins

### 🔬 Individual Models
- Model selector dropdown with per-model prediction card
- Class probability bar chart
- Full comparison table across all six models
- Grouped bar chart showing per-class confidence for every model side by side

### 🤝 Ensemble
- Select any subset of the six loaded models
- Choose **uniform average** or **weighted average** (per-model weight sliders 0.1 → 3.0)
- Ensemble prediction card with aggregated confidence
- Individual model breakdown cards beneath the ensemble result

---

## 📦 Dependencies

```
torch
torchaudio
streamlit
numpy
matplotlib
soundfile
pandas
huggingface_hub
python-dotenv
```

---

## 🧠 Design Notes

**Why six models?**  
This is a comparative study across recurrent and attention-based architectures for audio classification. Training all six on identical features and evaluating them individually and in ensemble isolates the contribution of architecture choice vs. positional encoding strategy.

**Why RoPE, Sinusoidal, and Relative PE?**  
Three fundamentally different approaches to position information: RoPE encodes position in the rotation of Q/K vectors (relative, no added parameters); sinusoidal adds fixed absolute position to the input; relative PE learns a bias over pairwise token distances. Comparing them on the same encoder backbone isolates their effect.

**Why log1p + z-score normalisation?**  
`log1p` compresses the large dynamic range of raw mel energies. Per-clip z-score normalisation makes the model invariant to recording loudness, which varies significantly across motor operating conditions.

**Why bidirectional RNNs?**  
For classification (not streaming), the full temporal context is available at inference time. Bidirectional encoding gives each timestep access to both past and future frames, improving the quality of the final hidden state used for classification.

---

## 👤 Author

**Jimmy Edifon Emmanuel (Eddy)**  
AI & Computer Vision Engineer · Lagos, Nigeria

[![HuggingFace](https://img.shields.io/badge/🤗-Eddy--Emmanuel-yellow)](https://huggingface.co/Eddy-Emmanuel)
[![GitHub](https://img.shields.io/badge/GitHub-Eddy--Emmanuel-181717?logo=github)](https://github.com/Eddy-Emmanuel)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://linkedin.com/in/eddy-emmanuel)

---

## 📄 License

MIT License — see `LICENSE` for details.
