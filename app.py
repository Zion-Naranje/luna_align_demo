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

# Custom Styling for Judges & Mobile Screens
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

def get_all_images():
    """Scans root and subfolders for any image file."""
    image_paths = []
    for root, _, files in os.walk("."):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in [".png", ".jpg", ".jpeg", ".webp"]:
                rel_path = os.path.relpath(os.path.join(root, f), ".")
                image_paths.append(rel_path)
    return sorted(image_paths)

all_images = get_all_images()

# Sidebar Navigation
st.sidebar.header("Evaluation Navigation")
eval_mode = st.sidebar.radio(
    "Choose Evaluation Mode:",
    [
        "1. LunaAlign Multi-Sensor Pipeline",
        "2. Preliminary Testing (SIFT vs. LoFTR)"
    ]
)

if not all_images:
    st.error("No images found in the repository root or subfolders. Please upload your images to GitHub.")
    st.stop()

# -------------------------------------------------------------
# MODE 1: LUNAALIGN MULTI-SENSOR REGISTRATION
# -------------------------------------------------------------
if eval_mode == "1. LunaAlign Multi-Sensor Pipeline":
    st.sidebar.subheader("Select Image Pair")
    
    # Try to set sensible defaults if matching files exist
    default_ref_idx = 0
    default_tgt_idx = min(1, len(all_images) - 1)
    
    for idx, path in enumerate(all_images):
        low = path.lower()
        if "lorc" in low or "ref" in low:
            default_ref_idx = idx
        elif "isro" in low or "target" in low or "aligned" in low:
            default_tgt_idx = idx

    ref_path = st.sidebar.selectbox("Base Reference Image:", all_images, index=default_ref_idx)
    target_path = st.sidebar.selectbox("Target Image to Align:", all_images, index=default_tgt_idx)

    img_ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    img_tgt = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

    if img_ref is not None and img_tgt is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Base Reference Frame")
            st.image(img_ref, caption=f"File: {os.path.basename(ref_path)} ({img_ref.shape[1]}x{img_ref.shape[0]})", use_container_width=True)
        with c2:
            st.subheader("Target Frame to Register")
            st.image(img_tgt, caption=f"File: {os.path.basename(target_path)} ({img_tgt.shape[1]}x{img_tgt.shape[0]})", use_container_width=True)

        if st.button("⚡ Run Registration Pipeline", type="primary", use_container_width=True):
            aligned = None
            blend = None
            matches_plot = None
            inliers = 0
            ratio = 0.0
            rmse = 0.0

            with st.spinner("Executing: Local CLAHE Normalization -> LoFTR Transformer -> MAGSAC++..."):
                t0 = time.time()
                try:
                    aligned, blend, matches_plot, inliers, ratio, rmse = run_luna_align(ref_path, target_path)
                    latency = round(time.time() - t0, 2)
                except Exception as e:
                    st.error(f"Execution Error: {e}")

            if aligned is not None and inliers >= 4:
                st.success(f"Alignment converged in {latency}s with {inliers:,} verified tie-points.")

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
                st.image(matches_plot, caption="Green vectors indicate geometrically consistent tie-points.", use_container_width=True)

                st.subheader("2. 50/50 Blended Overlay")
                st.image(blend, caption="Combined view: crisp crater rims without double edges confirm sub-pixel alignment.", use_container_width=True)

                st.subheader("3. Interactive Split-Screen Slider")
                st.write("Drag the handle to inspect how crater edges match between frames:")
                image_comparison(
                    img1=Image.fromarray(cv2.resize(img_ref, (640, 640))),
                    img2=Image.fromarray(aligned),
                    label1="Reference Basemap",
                    label2="Registered Target"
                )
            else:
                st.warning(f"Registration threshold not reached (found {inliers} inliers). Ensure the two selected images share overlapping terrain.")
    else:
        st.error("Failed to decode the selected image files.")

# -------------------------------------------------------------
# MODE 2: PRELIMINARY TESTING (SIFT vs. LoFTR Benchmark)
# -------------------------------------------------------------
else:
    st.subheader("Preliminary Testing — SIFT (Classical) vs. LoFTR (Transformer)")
    st.markdown(
        "Demonstrating why classical feature detection fails under changing solar illumination and why "
        "detector-free attention models are necessary for lunar terrain."
    )

    st.sidebar.subheader("Select Benchmark Pair")
    default_ref_b = 0
    default_tgt_b = min(1, len(all_images) - 1)
    for idx, path in enumerate(all_images):
        low = path.lower()
        if "lorc" in low:
            default_ref_b = idx
        elif "isro" in low:
            default_tgt_b = idx

    ref_bench = st.sidebar.selectbox("Reference Frame:", all_images, index=default_ref_b, key="bench_ref")
    target_bench = st.sidebar.selectbox("Target Frame:", all_images, index=default_tgt_b, key="bench_tgt")

    img_ref_b = cv2.imread(ref_bench, cv2.IMREAD_GRAYSCALE)
    img_tgt_b = cv2.imread(target_bench, cv2.IMREAD_GRAYSCALE)

    if img_ref_b is not None and img_tgt_b is not None:
        h_std, w_std = 640, 640
        img_ref_b_std = cv2.resize(img_ref_b, (w_std, h_std))
        img_tgt_b_std = cv2.resize(img_tgt_b, (w_std, h_std))

        if st.button("⚡ Run Comparative Benchmark", type="primary", use_container_width=True):
            with st.spinner("Executing SIFT baseline and LoFTR transformer sequentially..."):
                vis_sift, sift_total, sift_inliers = run_sift_match(img_ref_b_std, img_tgt_b_std)
                aligned_b, blend_b, vis_loftr, loftr_inliers, loftr_ratio, rmse_b = run_luna_align(ref_bench, target_bench)

            col_sift, col_loftr = st.columns(2)

            with col_sift:
                st.subheader("SIFT — Classical Baseline")
                st.image(vis_sift, caption=f"SIFT Inliers: {sift_inliers} verified matches", use_container_width=True)
                st.markdown(f"""
                * **Verified Matches:** **{sift_inliers} pts**
                * **Characteristics:** Sparse, clustered correspondences
                * **Failure Mode:** Relies entirely on local pixel gradient extrema. Misses matches across smooth lunar plains and drops features when shadows invert.
                """)

            with col_loftr:
                st.subheader("LoFTR — Deep Attention Matcher")
                st.image(vis_loftr, caption=f"LoFTR Inliers: {loftr_inliers:,} verified matches", use_container_width=True)
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
        st.error("Failed to decode the selected benchmark images.")
