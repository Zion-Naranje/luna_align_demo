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

def get_images_in_dir(directory):
    """Return all supported image files in a folder recursively."""
    if not os.path.isdir(directory):
        return []

    image_paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in [".png", ".jpg", ".jpeg", ".webp", ".JPG", ".JPEG"]:
                rel_path = os.path.relpath(os.path.join(root, f), ".")
                image_paths.append(rel_path)
    return sorted(image_paths)


def get_pair_images(pair_name):
    """Return the image pair for a named pair folder, using the first two images found."""
    pair_dir = os.path.join("pairs", pair_name)
    images = get_images_in_dir(pair_dir)
    return images[:2]


# Sidebar Navigation
st.sidebar.header("Evaluation Navigation")
eval_mode = st.sidebar.radio(
    "Choose Evaluation Mode:",
    [
        "1. LunaAlign Multi-Sensor Pipeline",
        "2. Preliminary Testing (SIFT vs. LoFTR)"
    ]
)

pair_labels = [f"pair{i}" for i in range(1, 5)]
preliminary_images = get_images_in_dir("preliminaryTesting")

if not preliminary_images:
    st.warning("No images found in preliminaryTesting/. Add the benchmark images there before running the comparison test.")

# -------------------------------------------------------------
# MODE 1: LUNAALIGN MULTI-SENSOR REGISTRATION
# -------------------------------------------------------------
if eval_mode == "1. LunaAlign Multi-Sensor Pipeline":
    st.sidebar.subheader("Select Pair")
    selected_pair = st.sidebar.radio("Choose Pair:", pair_labels, index=0)
    pair_images = get_pair_images(selected_pair)

    if len(pair_images) < 2:
        st.error(f"No valid image pair found in pairs/{selected_pair}. Add the reference and target images for this pair folder.")
        st.stop()

    ref_path, target_path = pair_images[0], pair_images[1]
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

    if len(preliminary_images) < 2:
        st.error("The preliminaryTesting folder must contain at least two image files for the benchmark.")
        st.stop()

    ref_bench, target_bench = preliminary_images[0], preliminary_images[1]
    img_ref_b = cv2.imread(ref_bench, cv2.IMREAD_GRAYSCALE)
    img_tgt_b = cv2.imread(target_bench, cv2.IMREAD_GRAYSCALE)

    if img_ref_b is not None and img_tgt_b is not None:
        h_std, w_std = 640, 640
        img_ref_b_std = cv2.resize(img_ref_b, (w_std, h_std))
        img_tgt_b_std = cv2.resize(img_tgt_b, (w_std, h_std))

        if st.button("⚡ Run Comparative Benchmark", type="primary", use_container_width=True):
            with st.spinner("Executing SIFT baseline and LoFTR transformer sequentially..."):
                vis_sift, sift_total, sift_inliers = run_sift_match(img_ref_b_std, img_tgt_b_std)

                try:
                    aligned_b, blend_b, vis_loftr, loftr_inliers, loftr_ratio, rmse_b = run_luna_align(ref_bench, target_bench)
                    loftr_available = True
                except Exception as exc:
                    loftr_available = False
                    vis_loftr = cv2.cvtColor(np.hstack((img_ref_b_std, img_tgt_b_std)), cv2.COLOR_GRAY2RGB)
                    loftr_error = str(exc)

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
                if loftr_available:
                    st.image(vis_loftr, caption=f"LoFTR Inliers: {loftr_inliers:,} verified matches", use_container_width=True)
                    st.markdown(f"""
                    * **Verified Matches:** **{loftr_inliers:,} pts**
                    * **Characteristics:** Uniform, dense surface coverage
                    * **Advantage:** Eliminates the local detector step. Self- and cross-attention correlate shapes globally, keeping tie-points intact across low-texture regions.
                    """)
                else:
                    st.image(vis_loftr, caption="LoFTR unavailable in this environment", use_container_width=True)
                    st.warning(
                        "LoFTR could not run because the pretrained model weights could not be downloaded. "
                        "This is usually caused by local SSL certificate validation issues. "
                        "The SIFT benchmark still works and can be used for comparison."
                    )

            if loftr_available:
                st.divider()
                ratio_mult = round(loftr_inliers / max(1, sift_inliers), 1)
                st.info(
                    f"**Empirical Finding:** LoFTR recovered **{ratio_mult}× more valid correspondences** "
                    "than SIFT under identical illumination shifts. This validates our choice of a detector-free "
                    "deep matching architecture for the LunaAlign core pipeline."
                )
    else:
        st.error("Failed to decode the selected benchmark images.")
