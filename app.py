import streamlit as st
import cv2
import numpy as np
import time
import os
from PIL import Image
from streamlit_image_comparison import image_comparison
from engine import run_luna_align, run_sift_match

st.set_page_config(
    page_title="LunaAlign | Chandrayaan-2 Registration Demo",
    page_icon="🛰️",
    layout="wide"
)

# Custom Styling for Navigation
st.markdown("""
    <style>
    .mobile-nav-banner {
        background: linear-gradient(90deg, #0f172a, #1e3a8a);
        color: #38bdf8;
        padding: 12px 16px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 20px;
        border-left: 5px solid #38bdf8;
    }
    div[data-testid="stSidebar"] {
        border-right: 2px solid #1e3a8a;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="mobile-nav-banner">📱 On mobile? Tap the <b>&gt; arrow</b> in the top-left corner to switch evaluation modes.</div>',
    unsafe_allow_html=True
)

st.title("🛰️ LunaAlign: Automated Lunar Image Registration")
st.markdown("Sub-pixel alignment across extreme illumination variations, sensor scale disparities, and lunar surface geometries.")

def resolve_image_path(filename):
    """Finds image files across root and subfolders regardless of casing or formatting."""
    if os.path.exists(filename):
        return filename
    
    clean_target = filename.lower().replace(" ", "").replace("_", "").replace("-", "")
    for root, _, files in os.walk("."):
        for f in files:
            if f.lower() == filename.lower():
                return os.path.join(root, f)
            clean_f = f.lower().replace(" ", "").replace("_", "").replace("-", "")
            if clean_f == clean_target:
                return os.path.join(root, f)
    return None

# Sidebar Navigation
st.sidebar.header("Evaluation Navigation")
eval_mode = st.sidebar.radio(
    "Choose Evaluation Mode:",
    [
        "1. Preliminary Testing (SIFT vs. LoFTR)",
        "2. LunaAlign Multi-Sensor Pipeline"
    ]
)

# -------------------------------------------------------------
# MODE 1: PRELIMINARY TESTING (Root SIFT vs. LoFTR Benchmark)
# -------------------------------------------------------------
if eval_mode == "1. Preliminary Testing (SIFT vs. LoFTR)":
    st.subheader("Preliminary Testing — SIFT (Classical) vs. LoFTR (Transformer)")
    st.markdown(
        "Demonstrating why classical feature detection fails under changing solar illumination and why "
        "detector-free attention models are necessary for lunar terrain."
    )

    ref_bench_name = "WhatsApp Image 2026-09-29 at 22.09.14.jpeg"
    target_bench_name = "isro_payload.JPG.jpeg"

    ref_bench = resolve_image_path(ref_bench_name)
    target_bench = resolve_image_path(target_bench_name)

    if ref_bench and target_bench:
        img_ref_b = cv2.imread(ref_bench, cv2.IMREAD_GRAYSCALE)
        img_tgt_b = cv2.imread(target_bench, cv2.IMREAD_GRAYSCALE)
        h_std, w_std = 480, 480
        img_ref_b_std = cv2.resize(img_ref_b, (w_std, h_std))
        img_tgt_b_std = cv2.resize(img_tgt_b, (w_std, h_std))

        c1, c2 = st.columns(2)
        with c1:
            st.image(img_ref_b_std, caption=f"Basemap: {os.path.basename(ref_bench)}")
        with c2:
            st.image(img_tgt_b_std, caption=f"Target: {os.path.basename(target_bench)}")

        if st.button("⚡ Run Comparative Benchmark", type="primary"):
            with st.spinner("Executing SIFT baseline and LoFTR transformer sequentially..."):
                vis_sift, sift_total, sift_inliers = run_sift_match(img_ref_b_std, img_tgt_b_std)
                aligned_b, blend_b, vis_loftr, loftr_inliers, loftr_ratio, rmse_b = run_luna_align(ref_bench, target_bench)

            col_sift, col_loftr = st.columns(2)

            with col_sift:
                st.subheader("SIFT — Classical Baseline")
                st.image(vis_sift, caption=f"SIFT Inliers: {sift_inliers} verified matches")
                st.markdown(f"""
                * **Verified Matches:** **{sift_inliers} pts**
                * **Characteristics:** Sparse, clustered correspondences
                * **Failure Mode:** Relies entirely on local pixel gradient extrema. Misses matches across smooth lunar plains and drops features when shadows invert.
                """)

            with col_loftr:
                st.subheader("LoFTR — Deep Attention Matcher")
                st.image(vis_loftr, caption=f"LoFTR Inliers: {loftr_inliers:,} verified matches")
                st.markdown(f"""
                * **Verified Matches:** **{loftr_inliers:,} pts**
                * **Characteristics:** Uniform, dense surface coverage
                * **Advantage:** Eliminates the local detector step. Self- and cross-attention correlate shapes globally, keeping tie-points intact across low-texture regions.
                """)

            st.divider()

            ratio_mult = round(loftr_inliers / max(1, sift_inliers), 1)
            st.info(
                f"**Empirical Finding:** LoFTR recovered **{ratio_mult}× more valid correspondences** "
                "than SIFT under identical illumination shifts. This validates our choice of a detector-free "
                "deep matching architecture for the LunaAlign core pipeline."
            )
    else:
        st.error(f"Could not locate `{ref_bench_name}` or `{target_bench_name}` in root repository.")

# -------------------------------------------------------------
# MODE 2: LUNAALIGN MULTI-SENSOR REGISTRATION (Sector Pairs)
# -------------------------------------------------------------
else:
    st.sidebar.subheader("Select Dataset Pair")
    
    pair_choice = st.sidebar.selectbox(
        "Choose Orbital Terrain Pair:",
        [
            "Primary Benchmark: Sector Tile 000001 (High Contrast & Distinct Morphology)",
            "Boundary Stress Test: Sector Tile 000007 (Extreme Repetitive Texture & High Regolith Noise)"
        ]
    )

    if pair_choice.startswith("Primary Benchmark"):
        ref_name = "reference_tile_000001.png"
        tgt_name = "source_tile_000001.png"
        pair_desc = "Optimal operating conditions: distinct crater rims provide strong structural context for transformer cross-attention."
    else:
        ref_name = "reference_tile_000007.png"
        tgt_name = "source_tile_000007.png"
        pair_desc = "Challenging edge-case scenario: uniform micro-crater fields without distinct landmarks test the safety rejection boundary."

    st.info(f"**Evaluation Focus:** {pair_desc}")

    ref_path = resolve_image_path(ref_name)
    target_path = resolve_image_path(tgt_name)

    if ref_path and target_path:
        img_ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        img_tgt = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Base Reference Frame")
            st.image(img_ref, caption=f"Basemap: {os.path.basename(ref_path)}")
        with c2:
            st.subheader("Target Frame to Register")
            st.image(img_tgt, caption=f"Target: {os.path.basename(target_path)}")

        if st.button("⚡ Run Registration Pipeline", type="primary"):
            aligned = None
            blend = None
            matches_plot = None
            inliers = 0
            ratio = 0.0
            rmse = 0.0

            with st.spinner("Executing: Tile Normalization (CLAHE) -> LoFTR Transformer -> MAGSAC++..."):
                t0 = time.time()
                try:
                    aligned, blend, matches_plot, inliers, ratio, rmse = run_luna_align(ref_path, target_path)
                    latency = round(time.time() - t0, 2)
                except Exception as e:
                    st.error(f"Execution Error: {e}")

            if aligned is not None and inliers >= 15:
                st.success(f"Alignment converged successfully in {latency}s with {inliers:,} verified tie-points.")

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Verified Inliers", f"{inliers:,} pts")
                m2.metric("Inlier Consistency", f"{ratio:.1f}%")
                m3.metric("Reprojection RMSE", f"{rmse} px")
                m4.metric("Engine Latency", f"{latency} s")

                st.divider()

                with st.expander("ℹ️ How the engine executed this alignment", expanded=True):
                    st.markdown("""
                    1. **Tile Normalization (CLAHE):** Balances contrast across pitch-black shadows and bright crater rims on local $8 \\times 8$ tiles.
                    2. **Dense Attention (LoFTR):** Correlates structural terrain landmarks contextually, maintaining tie-points across inverted shadow angles.
                    3. **MAGSAC++ Verification:** Discards false matches from shifting shadows and computes an optimal $3 \\times 3$ projective matrix.
                    4. **Sub-Pixel Warping:** Re-projects the target raster to align with the reference basemap.
                    """)

                st.divider()

                # Visualizations
                st.subheader("1. Landmark Correspondence Map")
                st.image(matches_plot, caption="Green vectors indicate geometrically consistent tie-points.")

                st.subheader("2. 50/50 Blended Overlay")
                st.image(blend, caption="Combined view: crisp crater rims without double edges confirm sub-pixel alignment.")

                st.subheader("3. Interactive Split-Screen Slider")
                st.write("Drag the handle to inspect how crater edges match between frames:")
                image_comparison(
                    img1=Image.fromarray(cv2.resize(img_ref, (480, 480))),
                    img2=Image.fromarray(aligned),
                    label1="Reference Basemap",
                    label2="Registered Target"
                )
            else:
                st.warning(
                    f"⚠️ **Safe Registration Threshold Triggered:** Pipeline recovered only {inliers} inliers "
                    "(below the operational threshold of 15 tie-points)."
                )
                
                with st.expander("🔍 Scientific Failure Analysis (Why this pair was rejected)", expanded=True):
                    st.markdown("""
                    * **Identical Micro-Topography:** The tile consists of isotropic, uniform micro-craters without unique structural anchors.
                    * **Ambiguous Attention Tokens:** LoFTR correctly assigns diffuse cross-attention across repetitive gravel textures rather than false-positive tie points.
                    * **Fail-Safe Integrity:** MAGSAC++ actively rejects false correspondences to prevent distorted homography warps or hallucinated basemap alignments.
                    * **Operational Recommendation:** In a real flight pipeline, this trigger prompts the orchestrator to widen the bounding sector or fuse low-resolution DEM priors.
                    """)
    else:
        st.error(f"Could not locate image files `{ref_name}` or `{tgt_name}` in root repository.")
