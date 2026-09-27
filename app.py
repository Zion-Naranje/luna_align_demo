import streamlit as st
import cv2
import numpy as np
import time
import os
from PIL import Image
from streamlit_image_comparison import image_comparison
from engine import run_luna_align

st.set_page_config(
    page_title="LunaAlign | Chandrayaan-2 Registration Demo",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ LunaAlign: Automated Lunar Image Registration")
st.markdown("Sub-pixel alignment of lunar orbital imagery under varying solar incidence angles and sensor scale disparities.")

# Sidebar Dataset Selector (Pre-loaded benchmark pairs only)
st.sidebar.header("Select Test Dataset")
pair_choice = st.sidebar.radio(
    "Choose a lunar terrain pair to evaluate:",
    [
        "Pair 1",
        "Pair 2"
    ]
)

if pair_choice == "Pair 1":
    ref_path = "ref_pair1.jpeg"
    target_path = "target_pair1.jpeg"
    pair_description = (
        "Variations in crater rim shadows caused by different solar incidence angles."
    )
else:
    ref_path = "ref_pair2.png"
    target_path = "target_pair2.png"
    pair_description = (
        "Testing against substantial illumination drift and local terrain shadow inversions."
    )

st.info(pair_description)

# Ensure images exist on disk
if os.path.exists(ref_path) and os.path.exists(target_path):
    img_ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    img_tgt = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Base Reference Frame")
        st.image(img_ref, caption="Static reference basemap (LROC)", use_container_width=True)
    with c2:
        st.subheader("Unregistered Target Frame")
        st.image(img_tgt, caption="New target tile to register (ISRO)", use_container_width=True)

    if st.button(" Run Test", type="primary", use_container_width=True):
        aligned = None
        blend = None
        matches_plot = None
        inliers = 0
        ratio = 0.0
        rmse = 0.0

        with st.spinner("Processing: Normalizing contrast -> Extracting LoFTR features -> Running MAGSAC++..."):
            t0 = time.time()
            try:
                aligned, blend, matches_plot, inliers, ratio, rmse = run_luna_align(ref_path, target_path)
                latency = round(time.time() - t0, 2)
            except Exception as e:
                st.error(f"Execution Error: {e}")

        if aligned is not None:
            st.success(f"Alignment converged in {latency} seconds with {inliers:,} verified tie-points.")

            # Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Verified Inliers", f"{inliers:,} pts")
            m2.metric("Inlier Consistency", f"{ratio:.1f}%")
            m3.metric("Reprojection RMSE", f"{rmse} px")
            m4.metric("Engine Latency", f"{latency} s")

            st.divider()

            # Technical Workflow Breakdown
            with st.expander("ℹ️ How the pipeline processed these images", expanded=True):
                st.markdown("""
                1. **Tile Normalization (CLAHE):** Applied localized histogram equalization to balance extreme contrast between pitch-black crater shadows and bright rims.
                2. **Feature Extraction (LoFTR Transformer):** Detected structural landmark points using self- and cross-attention, correlating crater shapes even where shadow directions changed.
                3. **Outlier Filtering (MAGSAC++):** Identified and discarded inconsistent match points caused by moving shadows, keeping only geometrically stable tie-points.
                4. **Homography Warping:** Applied a 3x3 projective transformation matrix to warp the target image into exact coordinate alignment with the reference basemap.
                """)

            st.divider()

            # 1. Matching lines
            st.subheader("1. Landmark Correspondence Map")
            st.image(
                matches_plot,
                caption="Green lines show valid tie-points connecting identical crater rim features between images.",
                use_container_width=True
            )

            # 2. 50/50 Blend
            st.subheader("2. 50/50 Blended Overlay")
            st.image(
                blend,
                caption="Combined view of both images. Sharp crater rims without double edges or blur confirm proper alignment.",
                use_container_width=True
            )

            # 3. Interactive Split Slider
            st.subheader("3. Interactive Split-Screen Slider")
            st.write("Drag the handle left and right to inspect how crater edges match between the two frames:")
            image_comparison(
                img1=Image.fromarray(cv2.resize(img_ref, (640, 640))),
                img2=Image.fromarray(aligned),
                label1="Reference Basemap",
                label2="Registered Target"
            )
        else:
            st.error("Registration failed. The selected tiles did not yield enough consistent landmarks.")
else:
    st.error("Image assets not found. Ensure the image files are uploaded to your repository.")
