import os
import cv2
import torch
import numpy as np
import kornia as K
import kornia.feature as KF

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MATCHER = None

def _get_matcher():
    global MATCHER
    if MATCHER is None:
        MATCHER = KF.LoFTR(pretrained="outdoor").to(DEVICE).eval()
    return MATCHER

def apply_clahe(img):
    """Enhance local shadow-invariant contrast across crater terrains."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def compute_rmse(pts_target, pts_ref, H, inliers):
    """
    Computes mathematical Root Mean Square Error (RMSE) 
    in pixels for verified inlier correspondences.
    """
    tgt_inliers = pts_target[inliers]
    ref_inliers = pts_ref[inliers]

    # Convert target inliers to homogeneous coordinates (N, 3)
    tgt_homo = np.hstack([tgt_inliers, np.ones((tgt_inliers.shape[0], 1))])
    
    # Project target points using estimated Homography matrix H
    projected = (H @ tgt_homo.T).T
    projected = projected[:, :2] / projected[:, 2:]

    # Calculate Euclidean distance between projected target points and reference points
    errors = np.linalg.norm(projected - ref_inliers, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return round(rmse, 2)

def run_luna_align(ref_path, target_path):
    """
    LunaAlign Core Registration Pipeline:
    1. Loads grayscale image rasters
    2. Applies local tile-level CLAHE dynamic range scaling
    3. Extracts deep structural correspondences via LoFTR Attention
    4. Computes projective homography via MAGSAC++
    5. Calculates dynamic reprojection RMSE
    6. Warps target frame, creates 50/50 blend, and renders vector match lines
    """
    if not os.path.exists(ref_path) or not os.path.exists(target_path):
        raise FileNotFoundError(f"Image files missing: {ref_path} or {target_path}")

    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or target_img is None:
        raise ValueError("OpenCV failed to decode image inputs. Ensure valid JPEG/PNG formats.")

    # Standardize dimensions for stable tensor processing & sub-2s execution
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

    # 3. Dense Transformer Feature Correspondence
    matcher = _get_matcher()
    with torch.inference_mode():
        corrs = matcher({"image0": t_ref, "image1": t_target})

    pts0 = corrs["keypoints0"].cpu().numpy()  # Base Reference Points
    pts1 = corrs["keypoints1"].cpu().numpy()  # Target Points

    total_matches = len(pts0)
    if total_matches < 4:
        return None, None, None, total_matches, 0.0, 0.0

    # 4. Geometric Verification & Robust Homography via MAGSAC++
    H, mask = cv2.findHomography(pts1, pts0, cv2.USAC_MAGSAC, 3.0)
    if H is None:
        return None, None, None, total_matches, 0.0, 0.0

    inliers = mask.ravel() == 1
    inlier_count = int(np.sum(inliers))
    inlier_ratio = (inlier_count / total_matches) * 100

    if inlier_count < 4:
        return None, None, None, inlier_count, inlier_ratio, 0.0

    # 5. Dynamic RMSE Computation
    rmse_value = compute_rmse(pts1, pts0, H, inliers)

    # 6. Warp Target Image to Base Reference Geometry
    aligned_target = cv2.warpPerspective(target_img, H, (w_std, h_std))

    # 7. Generate 50/50 Blended Overlay for Inspection
    overlay_blend = cv2.addWeighted(ref_img, 0.5, aligned_target, 0.5, 0)

    # 8. Render Side-by-Side Match Vectors with OpenCV (Buffer crash-proof)
    canvas = np.hstack((ref_img, target_img))
    vis_matches = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    inlier_pts0 = pts0[inliers]
    inlier_pts1 = pts1[inliers]
    step = max(1, len(inlier_pts0) // 40)  # Render up to 40 uncluttered lines

    for i in range(0, len(inlier_pts0), step):
        pt0 = (int(round(inlier_pts0[i][0])), int(round(inlier_pts0[i][1])))
        pt1 = (int(round(inlier_pts1[i][0] + w_std)), int(round(inlier_pts1[i][1])))

        cv2.circle(vis_matches, pt0, 3, (255, 50, 50), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis_matches, pt1, 3, (255, 50, 50), -1, lineType=cv2.LINE_AA)
        cv2.line(vis_matches, pt0, pt1, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    return aligned_target, overlay_blend, vis_matches, inlier_count, inlier_ratio, rmse_value
