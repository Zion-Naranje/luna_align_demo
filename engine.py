import os
import cv2
import torch
import numpy as np
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MATCHER = None

def _get_matcher():
    global MATCHER
    if MATCHER is None:
        MATCHER = KF.LoFTR(pretrained="outdoor").to(DEVICE).eval()
    return MATCHER

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def run_luna_align(ref_path, target_path):
    try:
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        if ref_img is None or target_img is None:
            return None, None, None, 0, 0.0

        # Resize to standard 640x640 for uniform coordinate mapping & fast compute
        h_std, w_std = 640, 640
        ref_img = cv2.resize(ref_img, (w_std, h_std))
        target_img = cv2.resize(target_img, (w_std, h_std))

        # 1. Illumination Normalization
        ref_clahe = apply_clahe(ref_img)
        target_clahe = apply_clahe(target_img)

        # 2. Tensor Conversion
        t_ref = torch.from_numpy(ref_clahe).float().unsqueeze(0).unsqueeze(0) / 255.0
        t_target = torch.from_numpy(target_clahe).float().unsqueeze(0).unsqueeze(0) / 255.0

        t_ref = t_ref.to(DEVICE)
        t_target = t_target.to(DEVICE)

        # 3. Dense LoFTR Matching
        matcher = _get_matcher()
        with torch.inference_mode():
            corrs = matcher({"image0": t_ref, "image1": t_target})

        pts0 = corrs["keypoints0"].cpu().numpy()  # Ref (x, y)
        pts1 = corrs["keypoints1"].cpu().numpy()  # Target (x, y)

        total_matches = len(pts0)
        if total_matches < 15:
            return None, None, None, total_matches, 0.0

        # 4. Robust Homography via MAGSAC++
        H, mask = cv2.findHomography(pts1, pts0, cv2.USAC_MAGSAC, 3.0)
        if H is None:
            return None, None, None, total_matches, 0.0

        inliers = mask.ravel() == 1
        inlier_count = int(np.sum(inliers))
        inlier_ratio = (inlier_count / total_matches) * 100

        # Guard against degenerate homographies
        if inlier_count < 15:
            return None, None, None, inlier_count, inlier_ratio

        # 5. Warp Target to Align with Reference
        aligned_target = cv2.warpPerspective(target_img, H, (w_std, h_std))

        # 6. Generate 50/50 Blended Overlay
        overlay_blend = cv2.addWeighted(ref_img, 0.5, aligned_target, 0.5, 0)

        # 7. Render Side-by-Side Green Matching Lines
        fig, ax = plt.subplots(figsize=(10, 4), dpi=130)
        canvas = np.hstack((ref_img, target_img))
        ax.imshow(canvas, cmap="gray")
        
        inlier_pts0 = pts0[inliers]
        inlier_pts1 = pts1[inliers]
        step = max(1, len(inlier_pts0) // 50)  # Draw up to 50 crisp lines
        
        for i in range(0, len(inlier_pts0), step):
            x0, y0 = inlier_pts0[i]
            x1, y1 = inlier_pts1[i]
            ax.plot([x0, x1 + w_std], [y0, y1], color="#00FF00", linewidth=0.8, alpha=0.8)
            ax.scatter([x0, x1 + w_std], [y0, y1], color="red", s=4)

        ax.axis("off")
        plt.tight_layout()
        fig.canvas.draw()
        vis_matches = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_matches = vis_matches.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return aligned_target, overlay_blend, vis_matches, inlier_count, inlier_ratio

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, 0, 0.0
