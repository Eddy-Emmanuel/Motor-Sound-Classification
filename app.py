import streamlit as st
import numpy as np
import math
import os
import io
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Motor Sound Classifier",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1117; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
.block-container { padding-top: 1.5rem; }

.pred-card {
    background: #1e2130;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid #4f8ef7;
}
.pred-card.winner {
    border-left: 4px solid #2ecc71;
    background: #162820;
}
.pred-label { font-size: 1.05rem; font-weight: 700; color: #e0e0e0; }
.pred-conf  { font-size: 0.85rem; color: #aaa; margin-top: 2px; }

.ensemble-box {
    background: linear-gradient(135deg, #1a1f36, #162820);
    border: 1px solid #2ecc71;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.ensemble-label { font-size: 2rem; font-weight: 800; color: #2ecc71; }
.ensemble-conf  { font-size: 1rem; color: #aaa; margin-top: 4px; }

.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #4f8ef7;
    border-bottom: 1px solid #2a2f45;
    padding-bottom: 4px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path(__file__).parent / "models"

class Config:
    n_fft              = 1024
    hop_length         = 512
    n_mels             = 80
    sampling_rate      = 16_000
    n_class            = 3
    input_size         = 80
    n_lstm_layer       = 2
    lstm_hidden_size   = 512
    n_gru_layer        = 2
    gru_hidden_size    = 512
    n_lstm_gru_layer   = 2
    lstm_gru_hidden_size = 512
    in_channel         = 1
    embed_dim          = 1024
    patch_size         = 16
    num_head           = 16
    n_blocks           = 12
    max_seq_len        = 2048
    device             = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

CLASS_NAMES  = ["engine1_good", "engine2_broken", "engine3_heavyload"]
CLASS_LABELS = {"engine1_good": "✅ Good", "engine2_broken": "⚠️ Broken", "engine3_heavyload": "🔴 Heavy Load"}
CLASS_COLORS = {"engine1_good": "#2ecc71", "engine2_broken": "#e74c3c", "engine3_heavyload": "#f39c12"}

MODEL_DISPLAY = {
    "gru_model":      "GRU",
    "lstm_model":     "LSTM",
    "lstm_gru_model": "LSTM-GRU",
    "rotational_pe":  "AudioModel (Rotational PE)",
    "sinusodal_pe":   "AudioModel (Sinusoidal PE)",
    "relative_pe":    "AudioModel (Relative PE)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────
class GRUModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(c.n_gru_layer):
            inp = c.input_size if i == 0 else 2 * c.gru_hidden_size
            self.layers.append(nn.GRU(inp, c.gru_hidden_size, batch_first=True, bidirectional=True))
        self.ln       = nn.LayerNorm(2 * c.gru_hidden_size)
        self.out_proj = nn.Linear(2 * c.gru_hidden_size, c.n_class)
    @property
    def name(self): return "gru_model"
    def forward(self, x):
        x = x.squeeze(1)
        for layer in self.layers:
            x, _ = layer(x); x = self.ln(x)
        return self.out_proj(x)[:, -1, :]

class LSTMModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(c.n_lstm_layer):
            inp = c.input_size if i == 0 else 2 * c.lstm_hidden_size
            self.layers.append(nn.LSTM(inp, c.lstm_hidden_size, batch_first=True, bidirectional=True))
        self.ln       = nn.LayerNorm(2 * c.lstm_hidden_size)
        self.out_proj = nn.Linear(2 * c.lstm_hidden_size, c.n_class)
    @property
    def name(self): return "lstm_model"
    def forward(self, x):
        x = x.squeeze(1)
        for layer in self.layers:
            x, _ = layer(x); x = self.ln(x)
        return self.out_proj(x)[:, -1, :]

class LSTMGRUModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(c.n_lstm_gru_layer):
            inp = c.input_size if i == 0 else 2 * c.lstm_gru_hidden_size
            self.layers.append(nn.LSTM(inp, c.lstm_gru_hidden_size, batch_first=True, bidirectional=True))
            self.layers.append(nn.GRU(2 * c.lstm_gru_hidden_size, c.lstm_gru_hidden_size, batch_first=True, bidirectional=True))
        self.ln       = nn.LayerNorm(2 * c.lstm_gru_hidden_size)
        self.out_proj = nn.Linear(2 * c.lstm_gru_hidden_size, c.n_class)
    @property
    def name(self): return "lstm_gru_model"
    def forward(self, x):
        x = x.squeeze(1)
        for layer in self.layers:
            x, _ = layer(x); x = self.ln(x)
        return self.out_proj(x)[:, -1, :]

class RotationalPE(nn.Module):
    def __init__(self, c):
        super().__init__()
        d_k = c.embed_dim // c.num_head
        pos = torch.arange(0, c.max_seq_len).float().unsqueeze(1)
        inv_freq = torch.exp((torch.arange(0, d_k, 2) * -math.log(10_000)) / d_k)
        mat_mul = pos * inv_freq
        self.register_buffer("cos", mat_mul.cos())
        self.register_buffer("sin", mat_mul.sin())
    @property
    def name(self): return "rotational_pe"
    def forward(self, x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        seq = x.shape[-2]
        cos = self.cos[:seq][None, None, None, :, :]
        sin = self.sin[:seq][None, None, None, :, :]
        return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1).flatten(-2)

class SinusoidalPE(nn.Module):
    def __init__(self, c):
        super().__init__()
        pe = torch.zeros(c.max_seq_len, c.embed_dim).float()
        pos = torch.arange(0, c.max_seq_len).float().unsqueeze(1)
        inv_freq = torch.exp((torch.arange(0, c.embed_dim, 2).float() * -math.log(10_000)) / c.embed_dim)
        mat_mul = pos * inv_freq
        pe[:, 0::2] = torch.sin(mat_mul); pe[:, 1::2] = torch.cos(mat_mul)
        self.register_buffer("pe", pe.unsqueeze(0).unsqueeze(0))
    @property
    def name(self): return "sinusodal_pe"
    def forward(self, x): return x + self.pe[:, :, :x.shape[2], :]

class RelativePE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.k = c.max_seq_len - 1
        self.rel_emb = nn.Embedding(2 * self.k + 1, c.embed_dim)
        self.proj    = nn.Linear(c.embed_dim, c.num_head, bias=False)
    @property
    def name(self): return "relative_pe"
    def forward(self, x):
        seq = x.shape[-2]
        pos = torch.arange(seq, device=x.device)
        rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(-self.k, self.k) + self.k
        return self.proj(self.rel_emb(rel)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

class AudioAttentionBlock(nn.Module):
    def __init__(self, c, pe):
        super().__init__()
        self.pe = pe; self.config = c
        self.ln  = nn.LayerNorm(c.embed_dim)
        self.qkv = nn.Linear(c.embed_dim, 3 * c.embed_dim, bias=False)
        self.out_proj = nn.Linear(c.embed_dim, c.embed_dim)
        self.d_k = c.embed_dim // c.num_head
    def _qkv(self, x):
        B, C, seq, _ = x.shape
        qkv = self.qkv(x).view(B, C, seq, 3, self.config.num_head, self.d_k)
        return qkv.permute(0, 1, 3, 4, 2, 5).unbind(2)
    def forward(self, x):
        B, C, seq, _ = x.shape
        residual = x; x = self.ln(x)
        if self.pe.name == "rotational_pe":
            q, k, v = self._qkv(x); q, k = self.pe(q), self.pe(k)
        elif self.pe.name == "sinusodal_pe":
            x = self.pe(x); q, k, v = self._qkv(x)
        else:
            q, k, v = self._qkv(x); bias = self.pe(x)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self.pe.name == "relative_pe": attn = attn + bias
        attn = torch.nn.functional.softmax(attn, dim=-1)
        out  = (attn @ v).transpose(2, 3).contiguous().view(B, C, seq, -1)
        return residual + self.out_proj(out)

class AudioEncoder(nn.Module):
    def __init__(self, c, pe):
        super().__init__()
        self.input_proj  = nn.Linear(c.n_mels, c.embed_dim, bias=False)
        self.attn_blocks = nn.ModuleList([AudioAttentionBlock(c, pe) for _ in range(c.n_blocks)])
        self.ln  = nn.LayerNorm(c.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(c.embed_dim, 4 * c.embed_dim), nn.GELU(),
            nn.Linear(4 * c.embed_dim, c.embed_dim),
        )
        self.dropout = nn.Dropout()
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.attn_blocks: x = block(x)
        return x + self.mlp(self.ln(x))

class AudioModel(nn.Module):
    def __init__(self, c, pe):
        super().__init__()
        self.pe = pe; self.ae = AudioEncoder(c, pe)
        self.out_proj = nn.Linear(c.embed_dim, c.n_class, bias=False)
    @property
    def name(self): return self.pe.name
    def forward(self, x):
        x = self.ae(x).squeeze(1)
        return self.out_proj(x).mean(dim=1)

# ─────────────────────────────────────────────────────────────────────────────
# Audio preprocessing
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_mel_transform():
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sampling_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
    )

def load_audio_bytes(audio_bytes: bytes):
    wave, sr = torchaudio.load(io.BytesIO(audio_bytes))
    if sr != cfg.sampling_rate:
        wave = torchaudio.transforms.Resample(sr, cfg.sampling_rate)(wave)
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    return wave, cfg.sampling_rate

def audio_to_mel(wave):
    mel = torch.log1p(get_mel_transform()(wave))
    mel_norm = (mel - mel.mean()) / (mel.std() + 1e-1)
    return mel_norm.transpose(1, 2)

def preprocess(audio_bytes, device):
    wave, sr = load_audio_bytes(audio_bytes)
    x = audio_to_mel(wave).unsqueeze(0).to(device)
    return x, wave

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def build_all_models():
    return {
        "gru_model":      GRUModel(cfg),
        "lstm_model":     LSTMModel(cfg),
        "lstm_gru_model": LSTMGRUModel(cfg),
        "rotational_pe":  AudioModel(cfg, RotationalPE(cfg)),
        "sinusodal_pe":   AudioModel(cfg, SinusoidalPE(cfg)),
        "relative_pe":    AudioModel(cfg, RelativePE(cfg)),
    }

@st.cache_resource
def load_models():
    all_models = build_all_models()
    loaded, errors = {}, []
    for name, model in all_models.items():
        ckpt = CHECKPOINT_DIR / f"{name}.pt"
        if not ckpt.exists():
            errors.append(f"{name}.pt not found in {CHECKPOINT_DIR}")
            continue
        try:
            model.load_state_dict(torch.load(ckpt, map_location=cfg.device))
            model.to(cfg.device).eval()
            loaded[name] = model
        except Exception as e:
            errors.append(f"{name}: {e}")
    return loaded, errors

# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_single(model, x):
    return torch.softmax(model(x), dim=-1).squeeze().cpu().numpy()

@torch.no_grad()
def run_ensemble(models: dict, x, weights: dict = None):
    all_probs, w_sum = [], 0.0
    for name, model in models.items():
        w = (weights or {}).get(name, 1.0)
        all_probs.append(run_single(model, x) * w)
        w_sum += w
    return np.sum(all_probs, axis=0) / w_sum

# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
_DARK_BG   = "#0e1117"
_PANEL_BG  = "#1a1d27"
_SPINE_COL = "#333"
_TICK_COL  = "#aaa"

def _style_ax(ax):
    ax.set_facecolor(_PANEL_BG)
    for sp in ax.spines.values(): sp.set_edgecolor(_SPINE_COL)
    ax.tick_params(colors=_TICK_COL)
    ax.xaxis.label.set_color(_TICK_COL)
    ax.yaxis.label.set_color(_TICK_COL)
    ax.title.set_color("#ddd")

def fig_audio_panels(wave_np, sr):
    wave_t   = torch.tensor(wave_np).unsqueeze(0)
    mel_np   = audio_to_mel(wave_t).squeeze().numpy().T

    stft     = torch.stft(
        torch.tensor(wave_np), n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        win_length=cfg.n_fft, window=torch.hann_window(cfg.n_fft), return_complex=True,
    )
    power_db = torchaudio.functional.amplitude_to_DB(
        stft.abs(), multiplier=10.0, amin=1e-10,
        db_multiplier=torch.log10(torch.tensor(1e-10)).item(), top_db=80.0,
    ).numpy()

    duration  = len(wave_np) / sr
    t         = np.linspace(0, duration, len(wave_np))
    freqs_khz = np.linspace(0, sr / 2 / 1000, power_db.shape[0])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8),
                             gridspec_kw={"height_ratios": [1, 1.4, 1.4]},
                             facecolor=_DARK_BG)
    for ax in axes: _style_ax(ax)

    axes[0].plot(t, wave_np, lw=0.6, color="#4f8ef7", alpha=0.9)
    axes[0].fill_between(t, wave_np, alpha=0.15, color="#4f8ef7")
    axes[0].axhline(0, color="#555", lw=0.5, ls="--")
    axes[0].set_xlim(0, duration); axes[0].set_ylabel("Amplitude"); axes[0].set_title("Waveform")

    im1 = axes[1].imshow(power_db, aspect="auto", origin="lower",
                          extent=[0, duration, freqs_khz[0], freqs_khz[-1]], cmap="inferno")
    cb1 = fig.colorbar(im1, ax=axes[1], pad=0.01)
    cb1.ax.yaxis.set_tick_params(color=_TICK_COL); cb1.set_label("Power (dB)", color=_TICK_COL)
    axes[1].set_ylabel("Frequency (kHz)"); axes[1].set_title("Spectrogram (STFT · linear freq)")

    im2 = axes[2].imshow(mel_np, aspect="auto", origin="lower",
                          extent=[0, duration, 0, cfg.n_mels], cmap="magma")
    cb2 = fig.colorbar(im2, ax=axes[2], pad=0.01)
    cb2.ax.yaxis.set_tick_params(color=_TICK_COL); cb2.set_label("Norm. log-mel energy", color=_TICK_COL)
    axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("Mel bin")
    axes[2].set_title(f"Mel Spectrogram ({cfg.n_mels} mel bins)")

    plt.tight_layout()
    return fig

def fig_prob_bars(probs_dict: dict, title=""):
    labels = list(probs_dict.keys())
    values = list(probs_dict.values())
    colors = [CLASS_COLORS.get(l, "#4f8ef7") for l in labels]

    fig, ax = plt.subplots(figsize=(6, 2.4), facecolor=_DARK_BG)
    _style_ax(ax)
    bars = ax.barh(labels, values, color=colors, alpha=0.85, height=0.5)
    for bar, val in zip(bars, values):
        ax.text(min(bar.get_width() + 0.02, 0.99), bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10, color="#ddd")
    ax.set_xlim(0, 1.14); ax.set_xlabel("Probability")
    ax.axvline(0.5, color="#555", ls="--", lw=0.8)
    if title: ax.set_title(title, color="#ddd", fontsize=10)
    plt.tight_layout()
    return fig

def fig_all_models_comparison(all_probs: dict):
    models = list(all_probs.keys())
    n      = len(models)
    x      = np.arange(n)
    width  = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n * 1.6), 4), facecolor=_DARK_BG)
    _style_ax(ax)
    for i, cls in enumerate(CLASS_NAMES):
        vals = [float(all_probs[m][CLASS_NAMES.index(cls)]) for m in models]
        ax.bar(x + i * width, vals, width, label=CLASS_LABELS[cls],
               color=CLASS_COLORS[cls], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models],
                        rotation=15, ha="right", fontsize=9, color="#ccc")
    ax.set_ylim(0, 1.12); ax.set_ylabel("Probability")
    ax.set_title("Per-Model Class Probabilities")
    ax.legend(fontsize=9, facecolor=_PANEL_BG, edgecolor="#444", labelcolor="#ddd")
    ax.axhline(1/3, color="#555", ls="--", lw=0.7, alpha=0.6)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Boot — load models once
# ─────────────────────────────────────────────────────────────────────────────
loaded_models, load_errors = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — status only
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔊 Motor Classifier")
    st.markdown("---")
    st.markdown(f"**Device:** `{'CUDA 🚀' if cfg.device == 'cuda' else 'CPU'}`")
    st.caption(f"SR: {cfg.sampling_rate} Hz · FFT: {cfg.n_fft} · Mels: {cfg.n_mels}")
    st.markdown(f"**Checkpoints:** `models/`")

    st.markdown("---")
    st.markdown("**Model status**")
    for name, label in MODEL_DISPLAY.items():
        ok = name in loaded_models
        st.markdown(f"{'✅' if ok else '❌'} `{label}`")

    if load_errors:
        with st.expander("⚠️ Load errors"):
            for e in load_errors: st.caption(e)

    st.markdown("---")
    st.markdown("**Classes**")
    for c in CLASS_NAMES:
        st.caption(f"{CLASS_LABELS[c]}  `{c}`")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🔊 Motor Sound Classifier")
st.markdown("Upload a `.wav` file to classify the motor state using individual models or an ensemble.")

if not loaded_models:
    st.error(f"No models loaded from `{CHECKPOINT_DIR}`. Make sure the `models/` folder is next to `app.py` and contains the `.pt` files.")
    st.stop()

uploaded = st.file_uploader("Upload a WAV audio file", type=["wav"], label_visibility="collapsed")

if not uploaded:
    st.info("👆 Upload a `.wav` file to get started.")
    st.stop()

audio_bytes = uploaded.read()
st.audio(audio_bytes, format="audio/wav")

with st.spinner("Preprocessing audio…"):
    x, wave  = preprocess(audio_bytes, cfg.device)
    wave_np  = wave.squeeze().numpy()

# ── Pre-run all models once so every tab can reuse results ────────────────────
@st.cache_data(show_spinner=False)
def run_all_models(_x_key, audio_hash):
    # _x_key is not serialisable; use audio_hash as cache key
    results = {}
    for name, model in loaded_models.items():
        results[name] = run_single(model, x)
    return results

audio_hash  = hash(audio_bytes)
all_probs   = run_all_models(None, audio_hash)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_viz, tab_single, tab_ensemble = st.tabs([
    "📊 Audio Visualization",
    "🔬 Individual Models",
    "🤝 Ensemble",
])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab_viz:
    st.markdown('<div class="section-header">Signal Analysis</div>', unsafe_allow_html=True)
    with st.spinner("Rendering plots…"):
        fig = fig_audio_panels(wave_np, cfg.sampling_rate)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    dur = len(wave_np) / cfg.sampling_rate
    c1, c2, c3 = st.columns(3)
    c1.metric("Duration",    f"{dur:.2f} s")
    c2.metric("Samples",     f"{len(wave_np):,}")
    c3.metric("Sample rate", f"{cfg.sampling_rate:,} Hz")

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab_single:
    st.markdown('<div class="section-header">Run inference with each loaded model</div>',
                unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Select model", list(loaded_models.keys()),
        format_func=lambda n: MODEL_DISPLAY.get(n, n),
    )

    probs    = all_probs[selected_model]
    pred_idx = int(np.argmax(probs))
    pred_cls = CLASS_NAMES[pred_idx]
    conf     = float(probs[pred_idx])

    col_res, col_chart = st.columns([1, 2])
    with col_res:
        st.markdown(f"""
        <div class="pred-card winner">
            <div class="pred-label">{CLASS_LABELS[pred_cls]}</div>
            <div style="font-size:0.8rem;color:#aaa;margin-top:4px;">{pred_cls}</div>
            <div style="font-size:1.8rem;font-weight:800;color:{CLASS_COLORS[pred_cls]};margin-top:8px;">
                {conf:.1%}
            </div>
            <div class="pred-conf">confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**All class probabilities**")
        for cls, p in zip(CLASS_NAMES, probs):
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
                f'<span style="color:#ccc">{CLASS_LABELS[cls]}</span>'
                f'<span style="color:{CLASS_COLORS[cls]};font-weight:700">{p:.1%}</span></div>',
                unsafe_allow_html=True,
            )

    with col_chart:
        fig2 = fig_prob_bars({c: float(p) for c, p in zip(CLASS_NAMES, probs)},
                              title=MODEL_DISPLAY[selected_model])
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    st.markdown("---")
    st.markdown('<div class="section-header">All Models at a Glance</div>', unsafe_allow_html=True)

    import pandas as pd
    rows = []
    for name, p in all_probs.items():
        pred = CLASS_NAMES[int(np.argmax(p))]
        rows.append({
            "Model":      MODEL_DISPLAY.get(name, name),
            "Prediction": CLASS_LABELS[pred],
            "Confidence": f"{float(p.max()):.1%}",
            **{CLASS_LABELS[c]: f"{float(p[i]):.1%}" for i, c in enumerate(CLASS_NAMES)},
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    fig3 = fig_all_models_comparison(all_probs)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

# ── Tab 3 ─────────────────────────────────────────────────────────────────────
with tab_ensemble:
    st.markdown('<div class="section-header">Build your ensemble</div>', unsafe_allow_html=True)

    col_sel, col_cfg = st.columns([1, 1])

    with col_sel:
        st.markdown("**Select models to include**")
        selected_for_ensemble = [
            name for name in loaded_models
            if st.checkbox(MODEL_DISPLAY.get(name, name), value=True, key=f"ens_{name}")
        ]

    with col_cfg:
        st.markdown("**Ensemble strategy**")
        strategy = st.radio("Strategy", ["Uniform average", "Weighted average"],
                            label_visibility="collapsed")
        weights = {}
        if strategy == "Weighted average" and selected_for_ensemble:
            st.markdown("**Weights** (higher = more influence)")
            for name in selected_for_ensemble:
                weights[name] = st.slider(MODEL_DISPLAY.get(name, name), 0.1, 3.0, 1.0,
                                           step=0.1, key=f"w_{name}")

    if not selected_for_ensemble:
        st.warning("Select at least one model.")
        st.stop()

    ens_models   = {n: loaded_models[n] for n in selected_for_ensemble}
    ens_probs    = run_ensemble(ens_models, x, weights if strategy == "Weighted average" else None)
    ens_pred_cls = CLASS_NAMES[int(np.argmax(ens_probs))]
    ens_conf     = float(ens_probs.max())

    st.markdown(f"""
    <div class="ensemble-box">
        <div style="color:#aaa;font-size:0.9rem;margin-bottom:8px;">
            Ensemble of {len(selected_for_ensemble)} model{'s' if len(selected_for_ensemble) > 1 else ''}
            &nbsp;·&nbsp; {strategy}
        </div>
        <div class="ensemble-label">{CLASS_LABELS[ens_pred_cls]}</div>
        <div style="font-size:0.85rem;color:#aaa;margin-top:2px;">{ens_pred_cls}</div>
        <div style="font-size:2.5rem;font-weight:900;color:#2ecc71;margin-top:10px;">{ens_conf:.1%}</div>
        <div class="ensemble-conf">ensemble confidence</div>
    </div>
    """, unsafe_allow_html=True)

    col_e1, col_e2 = st.columns([1, 2])
    with col_e1:
        st.markdown("**Ensemble probabilities**")
        for cls, p in zip(CLASS_NAMES, ens_probs):
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
                f'<span style="color:#ccc">{CLASS_LABELS[cls]}</span>'
                f'<span style="color:{CLASS_COLORS[cls]};font-weight:700">{p:.1%}</span></div>',
                unsafe_allow_html=True,
            )
    with col_e2:
        fig4 = fig_prob_bars({c: float(p) for c, p in zip(CLASS_NAMES, ens_probs)},
                              title=f"Ensemble ({len(selected_for_ensemble)} models · {strategy})")
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

    st.markdown("---")
    st.markdown('<div class="section-header">Individual model breakdown</div>', unsafe_allow_html=True)

    cols = st.columns(min(len(selected_for_ensemble), 3))
    for idx, name in enumerate(selected_for_ensemble):
        p    = all_probs[name]
        pred = CLASS_NAMES[int(np.argmax(p))]
        w_display = f"  ×{weights[name]:.1f}" if strategy == "Weighted average" else ""
        with cols[idx % 3]:
            st.markdown(
                f'<div class="pred-card">'
                f'<div class="pred-label">{MODEL_DISPLAY.get(name, name)}{w_display}</div>'
                f'<div style="color:{CLASS_COLORS[pred]};font-weight:700;font-size:1.1rem;margin-top:6px;">'
                f'{CLASS_LABELS[pred]}</div>'
                f'<div class="pred-conf">{float(p.max()):.1%} confidence</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            fig5 = fig_prob_bars({c: float(p[i]) for i, c in enumerate(CLASS_NAMES)})
            st.pyplot(fig5, use_container_width=True)
            plt.close(fig5)