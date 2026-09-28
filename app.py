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

# Custom CSS for high-contrast mobile navigation and styling
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
    '<div class="mobile-nav-banner">📱 On mobile? Tap the <b>&gt; arrow</b> in the top-left corner to switch dataset pairs and evaluation modes.</div>',
    unsafe_allow_html=True
)

st.title("🛰️ LunaAlign: Automated Lunar Image Registration")
st.markdown("Sub-pixel alignment across extreme illumination variations, sensor scale disparities, and pushbroom sensor geometries.")

# Sidebar Navigation
st.sidebar.header("Evaluation Navigation")
eval_mode = st.sidebar.radio(
    "Choose Evaluation Mode:",
    [
        "1. LunaAlign Multi-Sensor Pipeline",
        "2. Preliminary Testing (SIFT vs. LoFTR)"
    ]
)

# -------------------------------------------------------------
# MODE 1: LUNAALIGN MULTI-SENSOR REGISTRATION (Real Mission Pairs)
# -------------------------------------------------------------
if eval_mode == "1. LunaAlign Multi-Sensor Pipeline":
    st.sidebar.subheader("Select Mission Dataset Pair")
    dataset_choice = st.sidebar.selectbox(
        "Choose Real Chandrayaan-2 Pair:",
        [
            "Mission Pair 1: South Pole Ridge (TMC vs. IIRS)",
            "Mission Pair 2: Highland Basin Complex (TMC vs. IIRS)",
            "Mission Pair 3: Mare Plain Surface (TMC vs. IIRS)"
        ]
    )

    if dataset_choice == "Mission Pair 1: South Pole Ridge (TMC vs. IIRS)":
        ref_path = "1_ref_TMC (1).jpg"
        target_path = "2_aligned_IIRS (1).jpg"
        pair_desc = "Testing alignment across high-contrast polar topography between optical TMC and hyperspectral IIRS sensors."
    elif dataset_choice == "Mission Pair 2: Highland Basin Complex (TMC vs. IIRS)":
        ref_path = "1_ref_TMC (2).jpg"
        target_path = "2_aligned_IIRS (2).jpg"
        pair_desc = "Highland rugged terrain with oblique camera pointing and shadow boundary variation."
    else:
        ref_path = "1_ref_TMC (3).jpg"
        target_path = "2_aligned_IIRS (3).jpg"
        pair_desc = "Mare surface sector exhibiting sensor streaking and significant null-data boundary padding."

    st.info(f"**Dataset Focus:** {pair_desc}")

    if os.path.exists(ref_path) and os.path.exists(target_path):
        img_ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        img_tgt = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Base Reference Frame (TMC-2)")
            st.image(img_ref, caption=f"File: {ref_path}", use_container_width=True)
        with c2:
            st.subheader("Target Frame (IIRS Hyperspectral)")
            st.image(img_tgt, caption=f"File: {target_path}", use_container_width=True)

        if st.button("⚡ Run Registration Pipeline", type="primary", use_container_width=True):
            aligned = None
            blend = None
            matches_plot = None
            inliers = 0
            ratio = 0.0
            rmse = 0.0

            with st.spinner("Executing: Void Masking -> CLAHE Normalization -> LoFTR Attention -> MAGSAC++..."):
                t0 = time.time()
                try:
                    aligned, blend, matches_plot, inliers, ratio, rmse = run_luna_align(ref_path, target_path)
                    latency = round(time.time() - t0, 2)
                except Exception as e:
                    st.error(f"Execution Error: {e}")

            if aligned is not None:
                st.success(f"Alignment converged in {latency}s with {inliers:,} verified tie-points.")

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Verified Inliers", f"{inliers:,} pts")
                m2.metric("Inlier Consistency", f"{ratio:.1f}%")
                m3.metric("Reprojection RMSE", f"{rmse} px")
                m4.metric("Engine Latency", f"{latency} s")

                st.divider()

                with st.expander("ℹ️ How the engine handled this dataset", expanded=True):
                    st.markdown("""
                    1. **Mask-Aware Filtering:** Raw IIRS products feature diagonal null-data padding (black borders). The pipeline creates an eroded valid-data mask, filtering out edge artifacts before computing consensus.
                    2. **Tile Normalization (CLAHE):** Balances dynamic range across dark crater floors and bright highland slopes on local $8 \\times 8$ tiles.
                    3. **Dense Attention (LoFTR):** Relies on global image context rather than fragile corner detectors, linking terrain even across cross-sensor resolution disparities.
                    4. **MAGSAC++ Verification:** Discards inconsistent vectors and calculates the $3 \\times 3$ projective homography matrix.
                    """)

                st.divider()

                # 1. Matches
                st.subheader("1. Landmark Correspondence Map")
                st.image(matches_plot, caption="Green vectors represent verified tie-points on illuminated terrain.", use_container_width=True)

                # 2. 50/50 Blend
                st.subheader("2. 50/50 Blended Overlay")
                st.image(blend, caption="Overlay of reference basemap and registered target image.", use_container_width=True)

                # 3. Interactive Split Slider
                st.subheader("3. Interactive Split Slider")
                st.write("Drag the handle horizontally to inspect crater alignment across frames:")
                image_comparison(
                    img1=Image.fromarray(cv2.resize(img_ref, (640, 640))),
                    img2=Image.fromarray(aligned),
                    label1="Reference Basemap (TMC)",
                    label2="Registered Target (IIRS)"
                )
            else:
                st.error(f"Alignment did not reach convergence threshold (found {inliers} inliers). Try an adjacent sector or the Preliminary Testing benchmark.")
    else:
        st.error(f"Image files not found in root: `{ref_path}` or `{target_path}`.")

# -------------------------------------------------------------
# MODE 2: PRELIMINARY TESTING (SIFT vs. LoFTR Benchmark)
# -------------------------------------------------------------
else:
    st.subheader("Preliminary Testing — SIFT (Classical) vs. LoFTR (Transformer)")
    st.markdown(
        "Demonstrating why classical feature detection fails under changing solar illumination and why "
        "detector-free attention models are necessary for lunar terrain."
    )

    ref_bench = "lorc.JPG.jpeg"
    target_bench = "isro_target.jpeg"

    if os.path.exists(ref_bench) and os.path.exists(target_bench):
        img_ref_b = cv2.imread(ref_bench, cv2.IMREAD_GRAYSCALE)
        img_tgt_b = cv2.imread(target_bench, cv2.IMREAD_GRAYSCALE)
        h_std, w_std = 640, 640
        img_ref_b_std = cv2.resize(img_ref_b, (w_std, h_std))
        img_tgt_b_std = cv2.resize(img_tgt_b, (w_std, h_std))

        if st.button("⚡ Run Comparative Benchmark", type="primary", use_container_width=True):
            with st.spinner("Executing SIFT baseline and LoFTR transformer sequentially..."):
                # Run SIFT
                vis_sift, sift_total, sift_inliers = run_sift_match(img_ref_b_std, img_tgt_b_std)

                # Run LoFTR
                aligned_b, blend_b, vis_loftr, loftr_inliers, loftr_ratio, rmse_b = run_luna_align(ref_bench, target_bench)

            col_sift, col_loftr = st.columns(2)

            with col_sift:
                st.subheader("SIFT — Classical Baseline")
                st.image(vis_sift, caption=f"SIFT Inliers: {sift_inliers} verified matches", use_container_width=True)
                st.markdown(f"""
                * **Verified Matches:** **{sift_inliers} pts**
                * **Characteristics:** Sparse, clustered correspondences
                * **Failure Mode:** Relies entirely on local pixel gradient extrema. It misses matches across smooth lunar plains and drops features when shadows invert.
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
        st.error(f"Benchmark images `{ref_bench}` or `{target_bench}` not found in repository root.")
