# DESIGN.md — sphere_walker.html

Research tool: interactive navigation of a transformer's layer "topic spheres".
Minimal, scientific, dark. Single-file app; tokens below are the contract the
file's inline CSS/JS uses.

## 1. Tokens

### Color
| Token | Value | Use |
|---|---|---|
| `--bg` | `#0a0e14` | scene clear color / page background |
| `--fg` | `#d7e0ea` | primary text (HUD) |
| `--muted` | `#7d8ea3` | secondary text, hints |
| `--accent` | `#6ec1ff` | state dot halo, trail, interactive highlights |
| `--ring` | `#5aa9e6` | equator ring (glowing torus) |
| `--pole` | `#8a97a8` | pole axis + caps |
| `--grid` | `rgba(140,170,210,0.14)` | wireframe sphere |
| `--panel` | `rgba(10,14,20,0.72)` | HUD / legend panel fill |
| `--panel-border` | `rgba(110,193,255,0.22)` | HUD / legend panel border |

Topic palette (6 distinct, readable on dark, colorblind-tolerant hue spread):
`food #ff7a6b` · `animal #ffb454` · `color #ffd166` · `city #5aa9ff` ·
`nature #6fd08c` · `number #b48cff`

### Typography
- Stack: `ui-monospace, SFMono-Regular, Menlo, Consolas, monospace` (scientific, tabular).
- Scale: HUD label 11px / value 14px bold / hint 11px muted / legend 11px.
- Degree labels: `lat 56.2°`, `az 211.4°` — one decimal, no leading zeros.

### Spacing
- 4px grid. HUD panel padding 12px, gap 4px; legend gap 6px; corner inset 16px.

## 2. Primitives
- **HUD panel** (top-left, fixed, `pointer-events:none`): model, `sphere i/6 · layer L`,
  lat/az, topic, token, transitions, hint.
- **Legend** (top-right, fixed, `pointer-events:none`): 6 swatches + topic names.
- **Sphere** (radius 1, origin): wireframe mesh (opacity 0.14) + glowing equator
  torus + pole axis line with caps.
- **Topic marker**: 0.035-radius sphere at `(cos az, sin az, 0)` + canvas-texture
  sprite label (pill + topic name) at radius 1.14, always camera-facing.
- **State dot**: 0.06-radius bright sphere + additive halo; position from
  `theta = acos(u·h)`, `phi = atan2(h·r2, h·r1)`.
- **Trail**: `THREE.Line`, additive blending, vertex-color fade (bright→dim),
  last 120 positions.

## 3. Motion
- Movement: geodesic rotate-toward, STEP 2.5° (formulas copied verbatim from the
  validated CLI walker — see task spec). No animation easing; discrete steps.
- Camera: OrbitControls with damping; GPU-composited only (no layout animation).

## 4. Responsive
- Renderer resizes with window; `pixelRatio` capped at 2. HUD/legend are fixed
  overlays that never reflow.

## 5. Accessibility
- All text ≥ 11px, contrast ≥ 4.5:1 on `--bg` (muted `#7d8ea3` on `#0a0e14` ≈ 5.4:1).
- Topic identity conveyed by color AND text label (never color alone).
- `prefers-reduced-motion`: no continuous animation exists beyond camera damping.

## 6. Accepted debt
- No touch controls (keyboard-first research tool); no WebGL fallback message.
- CDN dependency (three@0.160) requires network; acceptable for a local tool.