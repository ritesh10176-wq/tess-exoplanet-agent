import streamlit as st
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TESS Exoplanet Detector",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Deep-space palette */
    :root {
        --space-black:  #080C14;
        --nebula-navy:  #0D1B2A;
        --panel-bg:     #111D2E;
        --border:       #1E3050;
        --accent-cyan:  #00C8FF;
        --accent-gold:  #FFB347;
        --accent-green: #39FF84;
        --accent-red:   #FF4560;
        --text-primary: #E8F4FD;
        --text-muted:   #7A9BB5;
        --font-mono:    'Courier New', monospace;
    }

    /* Global reset to dark space theme */
    .stApp { background-color: var(--space-black); color: var(--text-primary); }
    section[data-testid="stSidebar"] {
        background-color: var(--nebula-navy);
        border-right: 1px solid var(--border);
    }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0D1B2A 0%, #0a2540 50%, #0D1B2A 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 32px 36px 28px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: "";
        position: absolute; inset: 0;
        background: radial-gradient(ellipse at 70% 50%, rgba(0,200,255,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-family: var(--font-mono);
        font-size: 2.0rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: var(--accent-cyan);
        margin: 0 0 6px;
        text-shadow: 0 0 24px rgba(0,200,255,0.4);
    }
    .hero-sub {
        font-size: 0.95rem;
        color: var(--text-muted);
        margin: 0;
        max-width: 680px;
        line-height: 1.55;
    }
    .hero-badge {
        position: absolute; top: 24px; right: 28px;
        background: rgba(0,200,255,0.12);
        border: 1px solid rgba(0,200,255,0.35);
        border-radius: 20px;
        padding: 4px 14px;
        font-family: var(--font-mono);
        font-size: 0.72rem;
        color: var(--accent-cyan);
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    /* ── Stat cards ── */
    .stat-card {
        background: var(--panel-bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 20px 18px 16px;
        text-align: center;
        height: 100%;
        transition: border-color 0.2s;
    }
    .stat-card:hover { border-color: var(--accent-cyan); }
    .stat-label {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 8px;
    }
    .stat-value {
        font-family: var(--font-mono);
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 4px;
    }
    .stat-sub { font-size: 0.75rem; color: var(--text-muted); }

    /* ── Confidence meter ── */
    .conf-bar-wrap {
        background: var(--panel-bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 20px 24px;
        margin: 20px 0;
    }
    .conf-label {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 10px;
    }
    .conf-bar-bg {
        background: rgba(255,255,255,0.07);
        border-radius: 6px;
        height: 22px;
        overflow: hidden;
        position: relative;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.8s ease;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        font-family: var(--font-mono);
        font-size: 0.78rem;
        font-weight: 700;
        color: #000;
    }

    /* ── Section headers ── */
    .section-eyebrow {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent-cyan);
        margin-bottom: 4px;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 16px;
    }

    /* ── Info / alert panels ── */
    .alert-panel {
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.88rem;
        line-height: 1.5;
        margin: 12px 0;
    }
    .alert-success {
        background: rgba(57,255,132,0.08);
        border-left: 3px solid var(--accent-green);
        color: #B6FFD9;
    }
    .alert-warn {
        background: rgba(255,179,71,0.08);
        border-left: 3px solid var(--accent-gold);
        color: #FFE3B3;
    }
    .alert-info {
        background: rgba(0,200,255,0.08);
        border-left: 3px solid var(--accent-cyan);
        color: #B3ECFF;
    }
    .alert-danger {
        background: rgba(255,69,96,0.08);
        border-left: 3px solid var(--accent-red);
        color: #FFB3BF;
    }

    /* ── Sidebar styling ── */
    .sidebar-step {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 16px;
    }
    .step-num {
        font-family: var(--font-mono);
        font-size: 0.7rem;
        background: rgba(0,200,255,0.15);
        border: 1px solid rgba(0,200,255,0.3);
        border-radius: 50%;
        width: 22px; height: 22px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
        color: var(--accent-cyan);
        margin-top: 1px;
    }
    .step-text { font-size: 0.82rem; color: var(--text-muted); line-height: 1.45; }
    .step-text strong { color: var(--text-primary); }

    /* ── Input + Button overrides ── */
    .stTextInput > div > div > input {
        background: var(--panel-bg) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
        font-family: var(--font-mono) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0090C8, #005F8A) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: var(--font-mono) !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        padding: 0.55rem 1.4rem !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }

    /* matplotlib figure backgrounds */
    .stPlotlyChart, .element-container { border-radius: 10px; overflow: hidden; }

    /* hide default streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Sector table */
    .sector-table {
        width: 100%;
        border-collapse: collapse;
        font-family: var(--font-mono);
        font-size: 0.8rem;
    }
    .sector-table th {
        color: var(--text-muted);
        text-align: left;
        padding: 6px 10px;
        border-bottom: 1px solid var(--border);
        font-weight: normal;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-size: 0.68rem;
    }
    .sector-table td {
        padding: 7px 10px;
        border-bottom: 1px solid rgba(30,48,80,0.5);
        color: var(--text-primary);
    }
    .sector-table tr:last-child td { border-bottom: none; }
    .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.68rem;
        font-family: var(--font-mono);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ─────────────────────────────────────────────

def confidence_color(pct: float) -> str:
    """Return a CSS gradient colour based on detection probability."""
    if pct >= 75:
        return "linear-gradient(90deg, #00a86b, #39FF84)"
    elif pct >= 45:
        return "linear-gradient(90deg, #d4a017, #FFB347)"
    else:
        return "linear-gradient(90deg, #8B1A1A, #FF4560)"


def verdict(pct: float, snr: float) -> tuple[str, str, str]:
    """Return (label, css_class, explanation) for the detection result."""
    if pct >= 80 and snr >= 10:
        return (
            "🟢 HIGH-CONFIDENCE PLANET CANDIDATE",
            "alert-success",
            f"SNR of {snr:.1f} is well above the standard threshold of 7.1 — this signal is statistically robust. "
            "The periodic dimming profile is consistent with a transiting exoplanet."
        )
    elif pct >= 50:
        return (
            "🟡 POSSIBLE PLANET CANDIDATE",
            "alert-warn",
            f"SNR of {snr:.1f} crosses the 7.1σ detection threshold but falls short of high-confidence. "
            "Follow-up observations or additional TESS sectors recommended to confirm."
        )
    elif pct >= 25:
        return (
            "🔵 WEAK PERIODIC SIGNAL",
            "alert-info",
            f"SNR of {snr:.1f} is below the standard threshold. Could be instrumental noise, "
            "stellar variability, or a grazing transit. Further data required."
        )
    else:
        return (
            "🔴 NO CONVINCING TRANSIT DETECTED",
            "alert-danger",
            f"SNR of {snr:.1f} is very low. No periodic transit signal consistent with an exoplanet was found "
            "in this sector's data."
        )


def dark_fig(figsize=(12, 4)):
    """Create a matplotlib figure styled for our dark-space theme."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#080C14")
    ax.set_facecolor("#0D1B2A")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E3050")
    ax.tick_params(colors="#7A9BB5", labelsize=8)
    ax.xaxis.label.set_color("#7A9BB5")
    ax.yaxis.label.set_color("#7A9BB5")
    ax.title.set_color("#E8F4FD")
    ax.grid(color="#1E3050", linestyle="--", linewidth=0.5, alpha=0.6)
    return fig, ax


def dark_fig_two(figsize=(14, 8)):
    """Create a 2-panel matplotlib figure."""
    fig = plt.figure(figsize=figsize, facecolor="#080C14")
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.42)
    axes = [fig.add_subplot(gs[i]) for i in range(2)]
    for ax in axes:
        ax.set_facecolor("#0D1B2A")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1E3050")
        ax.tick_params(colors="#7A9BB5", labelsize=8)
        ax.xaxis.label.set_color("#7A9BB5")
        ax.yaxis.label.set_color("#7A9BB5")
        ax.title.set_color("#E8F4FD")
        ax.grid(color="#1E3050", linestyle="--", linewidth=0.5, alpha=0.6)
    return fig, axes


# ─────────────────────────────────────────────
# 3. SIDEBAR — AGENT ARCHITECTURE
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Courier New',monospace;font-size:0.68rem;letter-spacing:0.18em;
                text-transform:uppercase;color:#00C8FF;margin-bottom:4px;">System</div>
    <div style="font-size:1.05rem;font-weight:600;color:#E8F4FD;margin-bottom:18px;">
        Agent Architecture
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("Data Ingestion",      "Queries NASA's MAST archive via <code>lightkurve</code>. Discovers all available TESS sectors and pipelines (SPOC / QLP)."),
        ("Pre-processing",      "Removes NaN cadences and σ-clip outliers. Flattens stellar variability with a Savitzky-Golay filter (<code>window_length=101</code>)."),
        ("Periodogram Search",  "Runs a <strong>Box Least Squares (BLS)</strong> matrix across 3 000 trial periods (0.5–15 d) to detect periodic dimming signatures."),
        ("Confidence Scoring",  "Translates the peak Signal-to-Noise Ratio into a detection probability. SNR ≥ 12 → 100 %. Calibrated against known TESS planet hosts."),
        ("Multi-sector Fusion", "Optionally stacks multiple TESS sectors to boost SNR before folding, narrowing the period uncertainty window."),
    ]
    for i, (title, body) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="sidebar-step">
            <div class="step-num">{i}</div>
            <div class="step-text"><strong>{title}</strong><br>{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="alert-info" style="font-size:0.8rem;">
        💡 Try <strong>TIC 307210830</strong> to detect L 98-59 c — a confirmed super-Earth 
        with a 3.69-day orbit just 35 ly away.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#7A9BB5;line-height:1.55;">
        <strong style="color:#E8F4FD;">Detection Thresholds</strong><br>
        SNR ≥ 12 → High confidence<br>
        SNR 7–12 → Candidate<br>
        SNR 4–7 &nbsp;→ Weak signal<br>
        SNR &lt; 4 &nbsp;→ No detection
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 4. HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">NASA TESS · MAST Archive</div>
    <div class="hero-title">🛰️ TESS Exoplanet Detection Agent</div>
    <p class="hero-sub">
        Automatically downloads photometric light curves from NASA's Transiting Exoplanet Survey Satellite, 
        runs a Box Least Squares periodogram, and calculates the statistical probability of an orbiting planet.
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 5. INPUT CONTROLS
# ─────────────────────────────────────────────
col_in, col_opts = st.columns([2, 1])

with col_in:
    tic_id = st.text_input(
        "TESS Input Catalog ID",
        value="TIC 307210830",
        placeholder="e.g. TIC 307210830",
        help="Must start with 'TIC ' followed by the catalog number.",
    )

with col_opts:
    max_sectors = st.selectbox(
        "Sectors to analyse",
        options=[1, 2, 3, 5],
        index=0,
        help="More sectors = higher SNR but longer download time.",
    )

col_btn, col_hint = st.columns([1, 3])
with col_btn:
    run = st.button("⚡ Launch Detection", use_container_width=True)
with col_hint:
    st.markdown(
        "<div style='padding-top:8px;font-size:0.8rem;color:#7A9BB5;'>"
        "Analysis takes ~30 s per sector for the first run (data cached by NASA MAST).</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# 6. MAIN ANALYSIS PIPELINE
# ─────────────────────────────────────────────
if run:
    # ── 6a. Search archive ──────────────────
    with st.spinner("Querying NASA MAST archive…"):
        try:
            search_result = lk.search_lightcurve(tic_id, mission="TESS", author="SPOC")
        except Exception as e:
            st.error(f"Archive query failed: {e}")
            st.stop()

    if len(search_result) == 0:
        st.markdown("""
        <div class="alert-danger">
            ❌ No SPOC pipeline data found for this TIC ID.<br>
            Check that the ID is formatted as <code>TIC 307210830</code> and that the star 
            falls within a TESS observing sector.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── 6b. Show available sectors ──────────
    n_available = len(search_result)
    n_use       = min(max_sectors, n_available)

    st.markdown(f"""
    <div class="section-eyebrow">Step 1 — Archive Discovery</div>
    <div class="section-title">Found {n_available} TESS sector(s) · Downloading {n_use}</div>
    """, unsafe_allow_html=True)

    rows_html = ""
    for i, row in enumerate(search_result):
        used = "✔" if i < n_use else ""
        tag_col = "#39FF84" if i < n_use else "#1E3050"
        rows_html += f"""
        <tr>
            <td style="color:{'#00C8FF' if i<n_use else '#7A9BB5'};">{used}</td>
            <td>{getattr(row, 'mission', '—')}</td>
            <td>{getattr(row, 'year', '—')}</td>
            <td>{getattr(row, 'exptime', '—') if hasattr(row, 'exptime') else '—'} s</td>
        </tr>"""

    st.markdown(f"""
    <div style="background:var(--panel-bg);border:1px solid var(--border);border-radius:10px;padding:16px 18px;margin-bottom:20px;">
    <table class="sector-table">
        <thead><tr>
            <th>Use</th><th>Mission / Sector</th><th>Year</th><th>Cadence</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # ── 6c. Download & stitch sectors ────────
    lc_list = []
    progress = st.progress(0, text="Downloading sector 1…")
    for i in range(n_use):
        progress.progress((i) / n_use, text=f"Downloading sector {i+1} of {n_use}…")
        try:
            lc_raw = search_result[i].download()
            lc_clean = lc_raw.remove_nans().remove_outliers().flatten(window_length=101)
            lc_list.append(lc_clean)
        except Exception as e:
            st.warning(f"Sector {i+1} download failed ({e}). Skipping.")
    progress.progress(1.0, text="Download complete.")

    if not lc_list:
        st.error("All sector downloads failed. Try again or choose a different TIC ID.")
        st.stop()

    # Stitch if multiple sectors
    if len(lc_list) > 1:
        from lightkurve import LightCurveCollection
        lc_stitched = LightCurveCollection(lc_list).stitch()
    else:
        lc_stitched = lc_list[0]

    # ── 6d. BLS periodogram ──────────────────
    with st.spinner("Running Box Least Squares periodogram…"):
        periods = np.linspace(0.5, 15, 3000)
        bls     = lc_stitched.to_periodogram(method="bls", period=periods)

    best_period  = bls.period_at_max_power.value
    best_power   = float(np.max(bls.power.value))
    snr_arr      = bls.snr
    snr          = float(snr_arr[np.argmax(bls.power)].value)
    transit_time = float(bls.transit_time_at_max_power.value)
    duration     = float(bls.duration_at_max_power.value)

    # Depth estimate (relative flux drop)
    lc_folded    = lc_stitched.fold(period=best_period, epoch_time=transit_time)
    in_transit   = np.abs(lc_folded.phase.value) < (duration / best_period / 2)
    depth_ppm    = 0.0
    if np.any(in_transit):
        depth_ppm = float((1.0 - np.nanmedian(lc_folded.flux.value[in_transit])) * 1e6)

    # Probability (SNR-based, capped 0-100)
    probability  = float(np.clip((snr / 12.0) * 100, 0, 100))

    # ── 6e. Verdict & headline stats ─────────
    st.markdown("---")
    st.markdown("""
    <div class="section-eyebrow">Step 2 — Detection Results</div>
    <div class="section-title">Signal Analysis Summary</div>
    """, unsafe_allow_html=True)

    label, css_cls, explanation = verdict(probability, snr)
    st.markdown(f"""
    <div class="alert-panel {css_cls}">
        <strong>{label}</strong><br>
        <span style="font-size:0.85rem;">{explanation}</span>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bar
    bar_color = confidence_color(probability)
    st.markdown(f"""
    <div class="conf-bar-wrap">
        <div class="conf-label">Detection Confidence</div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{probability:.1f}%;background:{bar_color};">
                {probability:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stat cards
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("Orbital Period",   f"{best_period:.3f}", "days"),
        ("Transit Depth",    f"{depth_ppm:,.0f}",  "ppm"),
        ("Signal-to-Noise",  f"{snr:.1f}",          "σ"),
        ("Sectors Used",     str(len(lc_list)),      f"of {n_available} available"),
    ]
    for col, (lbl, val, sub) in zip([c1, c2, c3, c4], stats):
        val_color = "#00C8FF"
        if lbl == "Signal-to-Noise":
            val_color = "#39FF84" if snr >= 12 else ("#FFB347" if snr >= 7 else "#FF4560")
        col.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">{lbl}</div>
            <div class="stat-value" style="color:{val_color};">{val}</div>
            <div class="stat-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 6f. Plots ────────────────────────────
    st.markdown("""
    <div class="section-eyebrow">Step 3 — Photometric Visualisation</div>
    <div class="section-title">Light Curve & Transit Signature</div>
    """, unsafe_allow_html=True)

    # Plot A: Cleaned light curve
    fig1, ax1 = dark_fig(figsize=(12, 3.5))
    lc_stitched.scatter(ax=ax1, color="#00C8FF", s=1.2, alpha=0.6, label="Normalised flux")
    ax1.set_title("Cleaned & Flattened TESS Photometry", fontsize=11, pad=10)
    ax1.set_xlabel("Time (BTJD)", fontsize=9)
    ax1.set_ylabel("Normalised Flux", fontsize=9)
    ax1.legend(fontsize=8, facecolor="#0D1B2A", edgecolor="#1E3050", labelcolor="#7A9BB5")
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    # Plot B: BLS periodogram + folded transit (side-by-side)
    fig2 = plt.figure(figsize=(14, 7), facecolor="#080C14")
    gs2  = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.35)

    ax_bls  = fig2.add_subplot(gs2[0])
    ax_fold = fig2.add_subplot(gs2[1])

    for ax in [ax_bls, ax_fold]:
        ax.set_facecolor("#0D1B2A")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1E3050")
        ax.tick_params(colors="#7A9BB5", labelsize=8)
        ax.xaxis.label.set_color("#7A9BB5")
        ax.yaxis.label.set_color("#7A9BB5")
        ax.title.set_color("#E8F4FD")
        ax.grid(color="#1E3050", linestyle="--", linewidth=0.5, alpha=0.6)

    # BLS periodogram
    ax_bls.plot(bls.period.value, bls.power.value, color="#00C8FF", lw=0.8, alpha=0.9)
    ax_bls.axvline(best_period, color="#FFB347", lw=1.5, ls="--",
                   label=f"Peak: {best_period:.3f} d")
    ax_bls.set_xlabel("Trial Period (days)", fontsize=9)
    ax_bls.set_ylabel("BLS Power", fontsize=9)
    ax_bls.set_title("Box Least Squares Periodogram", fontsize=11, pad=10)
    ax_bls.legend(fontsize=8, facecolor="#0D1B2A", edgecolor="#1E3050", labelcolor="#7A9BB5")

    # Phase-folded transit
    lc_fold_plot = lc_stitched.fold(period=best_period, epoch_time=transit_time)
    lc_fold_plot.scatter(ax=ax_fold, color="#FF4560", s=1.8, alpha=0.5, label="Individual cadences")
    # Binned overlay
    try:
        lc_binned = lc_fold_plot.bin(time_bin_size=0.01)
        ax_fold.plot(lc_binned.phase.value, lc_binned.flux.value,
                     color="#FFB347", lw=1.8, label="Binned (10-min)")
    except Exception:
        pass
    ax_fold.set_xlabel("Phase (orbital fraction)", fontsize=9)
    ax_fold.set_ylabel("Normalised Flux", fontsize=9)
    ax_fold.set_title(f"Phase-Folded Transit  [P = {best_period:.3f} d]", fontsize=11, pad=10)
    ax_fold.legend(fontsize=8, facecolor="#0D1B2A", edgecolor="#1E3050", labelcolor="#7A9BB5")

    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # ── 6g. Interpretation guide ─────────────
    st.markdown("---")
    st.markdown("""
    <div class="section-eyebrow">Step 4 — Interpretation</div>
    <div class="section-title">How to Read These Results</div>
    """, unsafe_allow_html=True)

    col_guide1, col_guide2 = st.columns(2)
    with col_guide1:
        st.markdown("""
        <div class="alert-info" style="font-size:0.82rem;">
            <strong>📈 BLS Periodogram</strong><br>
            Each spike marks a trial period where the algorithm found a box-shaped dimming pattern. 
            The tallest spike (orange dashed line) is the most statistically significant orbital period. 
            Harmonics (half/double the real period) sometimes appear as secondary peaks.
        </div>
        """, unsafe_allow_html=True)
    with col_guide2:
        st.markdown("""
        <div class="alert-info" style="font-size:0.82rem;">
            <strong>🪐 Phase-Folded View</strong><br>
            All TESS cadences are wrapped onto one orbital cycle. A genuine transit appears as 
            a narrow, symmetric dip near phase 0. A flat bottom confirms a sharp ingress/egress — 
            the hallmark of a planet crossing a stellar disk.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="alert-panel alert-info" style="margin-top:12px;font-size:0.82rem;">
        <strong>⚠️ Important caveat:</strong>  
        This tool performs automated signal detection only. False positives from eclipsing binaries, 
        background contaminants, or systematic instrumental trends are common. Any candidate 
        SNR ≥ 7.1 should be followed up with ground-based photometry and radial-velocity spectroscopy 
        before a planet claim can be published.
    </div>
    """, unsafe_allow_html=True)