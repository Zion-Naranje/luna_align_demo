# luna_align_demo
Lunar image registration engine for Chandrayaan-2 payloads (OHRC, TMC-2, IIRS) invariant to extreme Sun angles and scale disparities. [SIH 2026 / PS: SIH26166]

+---------------------+     +--------------------+     +---------------------+     +--------------------+
|  Orbital Rasters    | --> | Tile-Level CLAHE   | --> | Dense Transformer   | --> | MAGSAC++           |
| (TMC / OHRC / LROC) |     | Normalization      |     | Matching (LoFTR)    |     | Homography & Warp  |
+---------------------+     +--------------------+     +---------------------+     +--------------------+


1. **Illumination Normalization:** Tile-level CLAHE dynamic range scaling enhances morphological structure inside crater shadow basins while preserving bright rim features.
2. **Dense Feature Correspondence:** Employs Local Feature Transformers (LoFTR) with self- and cross-attention backbones to correlate multi-scale crater geometries across severe solar incidence shifts.
3. **Robust Geometric Verification:** Uses MAGSAC++ to discard false correspondence vectors from moving shadows and estimate a sub-pixel projective homography matrix ($H$).
4. **Sub-Pixel Warping & Verification:** Resamples target imagery to the reference coordinate system, validated via dynamic RMSE computation and interactive visual overlays.

---

## 🚀 Live Demo Deployment

The live interactive application is deployed on Streamlit Community Cloud:
👉 **[Access the LunaAlign Live Engine](https://lunaaligndemo.streamlit.app/)**

---

## 🛠️ Local Development & Execution

# Clone the repository
git clone [https://github.com/your-username/lunaalign-demo.git](https://github.com/your-username/luna_align_demo.git)
cd luna_align_demo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit interface
streamlit run app.py
