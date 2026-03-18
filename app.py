"""
LiftCV — Streamlit GUI
Wraps analyze_video() in a web interface. No pipeline files are modified.

Usage:
    streamlit run app.py
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from scripts.analyze import analyze_video
from src.io.output import AnalysisResult
from src.config import CONFIDENCE_HIGH, VBT_ERROR_PCT

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="LiftCV", page_icon="🏋️", layout="centered")
st.title("LiftCV — Squat Analysis")

# ── Sidebar inputs ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Parameters")

    uploaded = st.file_uploader("Squat video", type=["mp4", "mov", "avi"])

    bar_weight_kg = st.number_input(
        "Bar weight (kg)", min_value=0.0, step=2.5, value=60.0
    )

    height_raw = st.number_input(
        "Athlete height (m or cm)", min_value=0.0, step=0.01, value=1.75
    )
    height_m = height_raw / 100.0 if height_raw > 3.0 else height_raw
    if height_raw > 3.0:
        st.caption(f"→ {height_m:.2f} m")

    debug = st.checkbox("Debug mode")

    missing = uploaded is None or bar_weight_kg == 0.0 or height_m == 0.0
    run = st.button("Analyze", disabled=missing)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_criterion(val: Optional[bool]) -> str:
    if val is None:
        return "N/A"
    return "✓" if val else "✗"


def _show_results(result: AnalysisResult) -> None:
    # Case A — rejected
    if result.rejection_reason:
        st.error(f"Unreliable analysis\n\n{result.rejection_reason}")
        return

    # Case B — valid result

    # 1. Main verdict
    if result.valid and result.confidence >= CONFIDENCE_HIGH:
        st.success("VALID")
    elif result.valid:
        st.warning(f"BORDERLINE (confidence {result.confidence:.0%})")
    else:
        st.error("INVALID")

    # 2. KO criteria table
    st.subheader("KO Criteria")
    depth_detail = ""
    if result.depth_angle_deg is not None:
        depth_detail = f"{result.depth_angle_deg:+.1f} cm above knee"
    ko_data = {
        "Criterion": ["Depth", "Initial lockout", "Final lockout", "Feet"],
        "Result": [
            _fmt_criterion(result.depth_ok),
            _fmt_criterion(result.initial_lockout_ok),
            _fmt_criterion(result.final_lockout_ok),
            "N/A",
        ],
        "Detail": [depth_detail, "", "", ""],
    }
    st.table(ko_data)

    # 3. Kinematic metrics
    st.subheader("Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("ROM", f"{result.rom_m:.2f} m" if result.rom_m is not None else "N/A")
    with c2:
        st.metric(
            "Avg velocity",
            f"{result.avg_concentric_ms:.2f} m/s"
            if result.avg_concentric_ms is not None
            else "N/A",
        )
    with c3:
        st.metric(
            "Peak velocity",
            f"{result.peak_concentric_ms:.2f} m/s"
            if result.peak_concentric_ms is not None
            else "N/A",
        )
    with c4:
        st.metric(
            "Bar deviation",
            f"{result.bar_deviation_cm:.1f} cm"
            if result.bar_deviation_cm is not None
            else "N/A",
        )

    # 4. Intensity estimate
    st.subheader("Intensity Estimate")
    if result.bar_weight_kg is None:
        st.write("Bar weight not provided — %1RM estimate disabled.")
    elif result.estimated_1rm_pct is None:
        st.write("%1RM: not computable (velocity data unavailable).")
    else:
        st.write(f"**Declared weight:** {result.bar_weight_kg:.0f} kg")
        st.write(
            f"**Estimated %1RM:** {result.estimated_1rm_pct:.0f}%  "
            f"(±{VBT_ERROR_PCT}% — generic curve, not calibrated)"
        )

    # 5. Declared limits
    st.info(
        "**Declared limitations**\n"
        "- Single lateral camera analysis.\n"
        "- Knee valgus: not observable, excluded.\n"
        "- Symmetry: estimate — not a direct measurement.\n"
        f"- %1RM: generic curve, expected error ±{VBT_ERROR_PCT}%.\n"
        "- Feet: detection unreliable on lateral view (feature disabled)."
    )


# ── Analysis execution ────────────────────────────────────────────────────────

if run:
    tmp_path = None
    try:
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Analysis in progress..."):
            result = analyze_video(tmp_path, bar_weight_kg, height_m, debug)

        st.session_state["result"] = result

    except (FileNotFoundError, ValueError) as e:
        st.error(f"Video error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        if debug:
            st.exception(e)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Display stored result (persists across sidebar interactions)
if "result" in st.session_state:
    _show_results(st.session_state["result"])
