# Motor Sound Classifier — Streamlit App

A production-ready web interface for the **IDMT-ISA Electric Engine** motor sound classification models (GRU, LSTM, LSTM-GRU, and Transformer variants with Rotational / Sinusoidal / Relative positional encodings).

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features

| Tab | What it does |
|-----|-------------|
| **Single File Inference** | Upload one WAV → waveform, mel spectrogram, confidence bar chart |
| **Batch Evaluation** | Upload many WAVs → confusion matrix, per-class P/R/F1, confidence distribution, downloadable CSV |
| **Training History** | Upload `*_history.csv` files → learning curve comparison across all models |

## Checkpoint setup

Place your trained `.pt` files in any directory and point the app to it via the sidebar:

```
gru_model.pt
lstm_model.pt
lstm_gru_model.pt
rotational_pe.pt
sinusodal_pe.pt
relative_pe.pt
```

If a checkpoint is not found the model runs with random weights (useful for UI testing).

## Labelling batch files

For evaluation metrics in the **Batch** tab, include the class name in each filename:

```
motor_normal_001.wav
motor_defective_032.wav
motor_noload_017.wav
```

## Saving training histories

In the training notebook, after each `model_trainer.fit()` call:

```python
gru_result.to_csv("gru_model_history.csv", index=False)
lstm_result.to_csv("lstm_model_history.csv", index=False)
# etc.
```

Then upload these CSVs in the **Training History** tab.
