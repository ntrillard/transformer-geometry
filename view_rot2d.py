#!/usr/bin/env python3
"""Streamlit viewer for the 2D rotation-scan sphere (rot2d_scan_*.npz).

Plots every (θ, φ) as a point on a 3D sphere, colored by which token is
predicted (top-1) at that angle pair.  Hover to see the exact token.

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
BASE = Path(__file__).resolve().parent
DEFAULT_ROT2D = BASE / "151k_states" / "chunks" / "rot2d"

def discover_rot2d():
    files = sorted(DEFAULT_ROT2D.glob("rot2d_scan_*.npz"))
    if not files:
        files = sorted(Path("/home/nicolas/model-harness/151k_states/chunks/rot2d").glob("rot2d_scan_*.npz"))
    return files

def load(fpath):
    d = np.load(fpath)
    theta = d["theta"]
    phi = d["phi"]
    top = d["top"]       # (n, n_phi, n_theta, K)
    tids = d["tids"]
    return theta, phi, top, tids

def sphere_point(th_deg, ph_deg):
    """Map (θ, φ) in degrees to a 3D point on the unit sphere."""
    th = np.deg2rad(th_deg)
    ph = np.deg2rad(ph_deg)
    x = np.cos(ph) * np.sin(th)
    y = np.sin(ph)
    z = np.cos(ph) * np.cos(th)
    return x, y, z

st.title("🌀 2D Rotation-Scan Sphere")
st.caption("Each point = hidden state rotated by (θ, φ). "
           "Color = top-1 token. Hover to see the token.")

files = discover_rot2d()
if not files:
    st.error("No rot2d_scan_*.npz found. Run scan_rotations_2d.py first.")
    st.stop()

fpath = st.sidebar.selectbox("Chunk file", files,
                             format_func=lambda p: p.name)

if fpath:
    theta, phi, top, tids = load(fpath)
    n_tok, n_phi, n_theta, K = top.shape

    scan_idx = st.sidebar.selectbox("Token (scan index)",
                                    range(n_tok),
                                    format_func=lambda i: f"tid={tids[i]}")

    c = st.columns(3)
    c[0].metric("Tokens", n_tok)
    c[1].metric("θ angles", n_theta)
    c[2].metric("φ angles", n_phi)

    # Build the sphere surface
    TH, PH = np.meshgrid(theta, phi)
    X, Y, Z = sphere_point(TH, PH)

    # Top-1 token at each (θ, φ) for the selected scan
    top1 = top[scan_idx, :, :, 0]  # (n_phi, n_theta)

    # Color: use token ID as integer, map to categorical color
    unique_tokens = np.unique(top1)
    # Create a colormap via Plotly's categorical palette
    n_u = len(unique_tokens)
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    token_colors = {t: palette[i % len(palette)] for i, t in enumerate(unique_tokens)}
    color_map = np.array([token_colors[t] for t in top1.ravel()])

    # Token text labels for hover
    token_text = np.array([str(t) for t in top1.ravel()])

    show_grid = st.sidebar.checkbox("Show wireframe grid", value=True)
    point_size = st.sidebar.slider("Point size", 1, 8, 3)
    opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.85, 0.05)

    fig = go.Figure()

    # Wireframe sphere
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
                                     x=dict(show=True, color='#444', width=0.3),
                                     y=dict(show=True, color='#444', width=0.3),
                                     z=dict(show=True, color='#444', width=0.3))))

    # Points on the sphere
    marker_colors = color_map.reshape(-1, 3)
    hover_texts = [f"θ={int(TH.ravel()[i])}° φ={int(PH.ravel()[i])}°<br>token={token_text[i]}"
                   for i in range(len(token_text))]

    fig.add_trace(go.Scatter3d(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        mode='markers',
        marker=dict(
            size=point_size,
            color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
                   for c in marker_colors],
            opacity=opacity,
        ),
        text=hover_texts,
        hovertemplate="<b>%{text}</b><extra></extra>",
        showlegend=False,
    ))

    # Legend: which token ID maps to which color
    fig2 = go.Figure()
    for tok, col in token_colors.items():
        fig2.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=5, color=col),
            name=f"tid={tok}",
        ))
    fig2.update_layout(
        legend=dict(
            x=1.02, y=1, xanchor='left', yanchor='top',
            font=dict(size=10),
            itemsizing='constant',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        height=1, margin=dict(l=0, r=0, t=0, b=0),
    )

    fig.update_layout(
        height=800,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            bgcolor='rgba(0,0,0,0)',
            aspectmode='cube',
            camera=dict(eye=dict(x=2.2, y=1.5, z=1.0)),
            xaxis=dict(showspikes=False, showgrid=False, zeroline=False, visible=False,
                       range=[-1.1, 1.1]),
            yaxis=dict(showspikes=False, showgrid=False, zeroline=False, visible=False,
                       range=[-1.1, 1.1]),
            zaxis=dict(showspikes=False, showgrid=False, zeroline=False, visible=False,
                       range=[-1.1, 1.1]),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    st.subheader("Token distribution over the sphere")
    tok_counts = {int(t): int((top1 == t).sum()) for t in unique_tokens}
    top_tok = sorted(tok_counts.items(), key=lambda x: -x[1])[:10]
    cols = st.columns(len(top_tok))
    for i, (tid, count) in enumerate(top_tok):
        cols[i].metric(f"tid={tid}", f"{count}", f"{100*count/(n_phi*n_theta):.0f}%")

    # Show at key angles
    st.subheader("Top-1 at cardinal angles")
    card = [(0, 0, "true h"), (90, 0, "self-tangent"), (180, 0, "antipode -h"),
            (0, 90, "target-tangent"), (90, 90, "mixed"), (180, 90, "antipode + target"),
            (0, 180, "antipode -h (via φ)"), (180, 180, "back to h")]
    card_data = []
    for th_deg, ph_deg, label in card:
        ti_idx = np.argmin(np.abs(theta - th_deg))
        pi_idx = np.argmin(np.abs(phi - ph_deg))
        tid = int(top1[pi_idx, ti_idx])
        card_data.append({"θ": f"{th_deg}°", "φ": f"{ph_deg}°", "label": label, "top-1": f"tid={tid}"})
    st.dataframe(card_data, use_container_width=True)