#!/usr/bin/env python3
"""3D sphere viewer for rotation-scan data (sphere or ellipsoid).

Loads a rot2d_scan_*.npz or rot2d_ellipse_scan_*.npz file and renders
the (θ, φ) scan as colored points on a sphere. Select which token to
view from the sidebar. Shows decoded token text on hover.

Launch:
  streamlit run view_rot2d.py --server.address localhost --server.port 8502
"""
import sys, re
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Rotation Sphere", page_icon="🌀",
                   layout="wide", initial_sidebar_state="expanded")

MODEL = "Qwen/Qwen2-1.5B-Instruct"
MODEL_DIR = Path(__file__).resolve().parent / "models"

def discover_files():
    out = []
    for dname, pattern in [
        ("rot2d", "rot2d_scan_*.npz"),
        ("rot2d_ellipse", "rot2d_ellipse_scan_*.npz"),
    ]:
        p = Path(f"/home/nicolas/model-harness/151k_states/chunks/{dname}")
        if p.exists():
            out.extend(sorted(p.glob(pattern)))
    return out

def load(fpath):
    d = np.load(fpath)
    return d["theta"], d["phi"], d["top"], d["tids"]

def sphere_point(th_deg, ph_deg):
    th = np.deg2rad(th_deg)
    ph = np.deg2rad(ph_deg)
    return (np.cos(ph) * np.sin(th), np.sin(ph), np.cos(ph) * np.cos(th))

st.title("🌀 Rotation-Scan Sphere Viewer")
st.caption("Each point = hidden state rotated by (θ, φ). Color = top-1 token. Hover to see decoded text.")

if "mode_label" not in st.session_state:
    st.session_state.mode_label = "sphere"

files = discover_files()
if not files:
    st.error("No rot2d scan files found. Run scan_rotations_2d.py first.")
    st.stop()

fpath = st.sidebar.selectbox("Chunk file", files, format_func=lambda p: p.name)

if fpath:
    fname = fpath.name
    if "ellipse" in fname:
        st.session_state.mode_label = "ellipsoid"
    else:
        st.session_state.mode_label = "sphere"
    st.sidebar.caption(f"Geometry: {st.session_state.mode_label}")

    theta, phi, top, tids = load(fpath)
    n_tok, n_phi, n_theta, K = top.shape

    # Load tokenizer
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    except Exception:
        tok = None

    scan_idx = st.sidebar.selectbox("Token", range(n_tok),
                                    format_func=lambda i: f"tid={tids[i]}")

    pt_size = st.sidebar.slider("Point size", 1, 12, 5)
    opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.8, 0.05)
    show_grid = st.sidebar.checkbox("Wireframe grid", value=False)

    TH, PH = np.meshgrid(theta, phi)
    X, Y, Z = sphere_point(TH, PH)

    top1 = top[scan_idx, :, :, 0]
    unique_tokens = np.unique(top1)

    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    def hex_to_rgb(h):
        return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))
    token_colors = {t: hex_to_rgb(palette[i % len(palette)]) for i, t in enumerate(unique_tokens)}
    marker_colors = [token_colors[t] for t in top1.ravel()]

    # Decode token IDs to text for hover
    token_text_map = {}
    for tid in unique_tokens:
        tid_int = int(tid)
        if tok:
            token_text_map[tid] = tok.decode([tid_int]).strip()
        else:
            token_text_map[tid] = str(tid_int)

    hover_texts = [
        f"θ={int(TH.ravel()[i])}° φ={int(PH.ravel()[i])}°<br>"
        f"tok {top1.ravel()[i]} {token_text_map[top1.ravel()[i]]!r}"
        for i in range(len(X.ravel()))
    ]

    fig = go.Figure()

    if show_grid:
        u = np.linspace(0, 2*np.pi, 24)
        v = np.linspace(0, np.pi, 16)
        gu, gv = np.meshgrid(u, v)
        gx = np.cos(gv) * np.cos(gu)
        gy = np.sin(gv)
        gz = np.cos(gv) * np.sin(gu)
        fig.add_trace(go.Surface(x=gx, y=gy, z=gz, opacity=0.04,
                                 colorscale=[[0, '#444'], [1, '#444']],
                                 showscale=False, hoverinfo='skip',
                                 contours=dict(
                                     x=dict(show=True, color='#444', width=1),
                                     y=dict(show=True, color='#444', width=1),
                                     z=dict(show=True, color='#444', width=1))))

    fig.add_trace(go.Scatter3d(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        mode='markers',
        marker=dict(
            size=pt_size,
            color=[f'rgb({r},{g},{b})' for r, g, b in marker_colors],
            opacity=opacity,
        ),
        text=hover_texts,
        hovertemplate="<b>%{text}</b><extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        height=800,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            bgcolor='rgba(0,0,0,0)',
            aspectmode='cube',
            camera=dict(eye=dict(x=2.2, y=1.5, z=1.0)),
            xaxis=dict(visible=False, range=[-1.1, 1.1]),
            yaxis=dict(visible=False, range=[-1.1, 1.1]),
            zaxis=dict(visible=False, range=[-1.1, 1.1]),
        ),
    )

    st.plotly_chart(fig, width='stretch')

    # Token distribution table for this scan
    st.subheader(f"Token coverage for tid={tids[scan_idx]}")
    tok_counts = {int(t): int((top1 == t).sum()) for t in unique_tokens}
    top_tok = sorted(tok_counts.items(), key=lambda x: -x[1])[:8]
    rows = [{"token ID": tid, "text": repr(token_text_map[tid]),
              "% sphere": f"{100*count/(n_phi*n_theta):.1f}%"}
            for tid, count in top_tok]
    st.dataframe(rows, use_container_width=True, hide_index=True)