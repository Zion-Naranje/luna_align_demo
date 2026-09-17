c_u1, c_u2 = st.columns(2)
    with c_u1:
        f1 = st.file_uploader("Upload Reference Tile (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if f1 is not None:
            ref_path = os.path.join(temp_dir, "ref_upload.png")
            with open(ref_path, "wb") as f:
                f.write(f1.getvalue())  # Uses getvalue() so data persists across button clicks
    with c_u2:
        f2 = st.file_uploader("Upload Target Tile (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if f2 is not None:
            target_path = os.path.join(temp_dir, "tgt_upload.png")
            with open(target_path, "wb") as f:
                f.write(f2.getvalue())  # Uses getvalue() so data persists across button clicks
