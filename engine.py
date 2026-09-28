import os
import cv2
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MATCHER = None

def _get_matcher():
    """Lazy-load LoFTR matcher to prevent Streamlit Cloud boot-time ImportErrors."""
    global MATCHER
    if MATCHER is None:
        try:
            from kornia.feature import LoFTR
        except ImportError:
            import kornia.feature as KF
            LoFTR = KF.LoFTR
            
        MATCHER = LoFTR(pretrained="outdoor").to(DEVICE).eval()
    return MATCHER

def apply_clahe(img):
    """Enhance local contrast across shadow-cast lunar basins."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def compute_rmse(pts_target, pts_ref, H, inliers):
    """Computes Root Mean Square Error (RMSE) in pixels for verified inliers."""
    tgt_inliers = pts_target[inliers]
    ref_inliers = pts_ref[inliers]

    tgt_homo = np.hstack([tgt_inliers, np.ones((tgt_inliers.shape[0], 1))])
    projected = (H @ tgt_homo.T).T
    projected = projected[:, :2] / projected[:, 2:]

    errors = np.linalg.norm(projected - ref_inliers, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return round(rmse, 2)

def run_sift_match(ref_img, target_img):
    """
    Classical SIFT baseline with Lowe's ratio test and RANSAC filtering.
    """
    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(ref_img, None)
    kp1, des1 = sift.detectAndCompute(target_img, None)

    if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
        canvas = np.hstack((ref_img, target_img))
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB), 0, 0

    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des0, des1, k=2)
    good_matches = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    total_good = len(good_matches)
    if total_good < 4:
        canvas = np.hstack((ref_img, target_img))
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB), total_good, 0

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good_matches])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good_matches])

    H, mask = cv2.findHomography(pts1, pts0, cv2.RANSAC, 3.0)
    inlier_count = 0
    canvas = np.hstack((ref_img, target_img))
    vis_sift = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    w_std = ref_img.shape[1]

    if mask is not None:
        inliers = mask.ravel() == 1
        inlier_count = int(np.sum(inliers))
        step = max(1, inlier_count // 35)
        for i in range(0, len(pts0), step):
            if inliers[i]:
                p0 = (int(round(pts0[i][0])), int(round(pts0[i][1])))
                p1 = (int(round(pts1[i][0] + w_std)), int(round(pts1[i][1])))
                cv2.circle(vis_sift, p0, 3, (255, 60, 60), -1)
                cv2.circle(vis_sift, p1, 3, (255, 60, 60), -1)
                cv2.line(vis_sift, p0, p1, (0, 255, 255), 1, lineType=cv2.LINE_AA)

    return vis_sift, total_good, inlier_count

def run_luna_align(ref_path, target_path):
    """
    Mask-aware LoFTR registration engine:
    - Filters out black boundary voids in raw IIRS rasters
    - Normalizes local terrain dynamics with CLAHE
    - Establishes dense transformer correspondences
    - Solves homography via MAGSAC++
    """
    if not os.path.exists(ref_path) or not os.path.exists(target_path):
        raise FileNotFoundError(f"Missing file: {ref_path} or {target_path}")

    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or target_img is None:
        raise ValueError("Could not decode image files. Ensure valid JPEG/PNG formats.")

    # Standardize dimensions for consistent scale and fast CPU runtime
    h_std, w_std = 640, 640
    ref_img = cv2.resize(ref_img, (w_std, h_std))
    target_img = cv2.resize(target_img, (w_std, h_std))

    # Mask out null-data/black padding areas (intensity <= 15)
    kernel = np.ones((5, 5), np.uint8)
    mask_ref = cv2.erode((ref_img > 15).astype(np.uint8), kernel)
    mask_tgt = cv2.erode((target_img > 15).astype(np.uint8), kernel)

    # Contrast balancing
    ref_clahe = apply_clahe(ref_img)
    target_clahe = apply_clahe(target_img)

    # Tensor conversion
    t_ref = torch.from_numpy(ref_clahe).float().unsqueeze(0).unsqueeze(0) / 255.0
    t_target = torch.from_numpy(target_clahe).float().unsqueeze(0).unsqueeze(0) / 255.0
    t_ref = t_ref.to(DEVICE)
    t_target = t_target.to(DEVICE)

    matcher = _get_matcher()
    with torch.inference_mode():
        corrs = matcher({"image0": t_ref, "image1": t_target})

    pts0 = corrs["keypoints0"].cpu().numpy()
    pts1 = corrs["keypoints1"].cpu().numpy()

    # Filter out candidate matches that fall inside black margins
    valid_pts = []
    for i in range(len(pts0)):
        x0, y0 = int(round(pts0[i][0])), int(round(pts0[i][1]))
        x1, y1 = int(round(pts1[i][0])), int(round(pts1[i][1]))

        x0 = min(max(x0, 0), w_std - 1)
        y0 = min(max(y0, 0), h_std - 1)
        x1 = min(max(x1, 0), w_std - 1)
        y1 = min(max(y1, 0), h_std - 1)

        if mask_ref[y0, x0] == 1 and mask_tgt[y1, x1] == 1:
            valid_pts.append(i)

    if len(valid_pts) < 4:
        return None, None, None, len(valid_pts), 0.0, 0.0

    pts0 = pts0[valid_pts]
    pts1 = pts1[valid_pts]
    total_valid_matches = len(pts0)

    # Geometric verification via MAGSAC++
    H, mask = cv2.findHomography(pts1, pts0, cv2.USAC_MAGSAC, 3.5)
    if H is None:
        return None, None, None, total_valid_matches, 0.0, 0.0

    inliers = mask.ravel() == 1
    inlier_count = int(np.sum(inliers))
    inlier_ratio = (inlier_count / total_valid_matches) * 100

    if inlier_count < 4:
        return None, None, None, inlier_count, inlier_ratio, 0.0

    rmse_value = compute_rmse(pts1, pts0, H, inliers)

    # Warp target to reference frame
    aligned_target = cv2.warpPerspective(target_img, H, (w_std, h_std))
    overlay_blend = cv2.addWeighted(ref_img, 0.5, aligned_target, 0.5, 0)

    # Draw tie-point match vectors
    canvas = np.hstack((ref_img, target_img))
    vis_matches = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    inlier_pts0 = pts0[inliers]
    inlier_pts1 = pts1[inliers]
    step = max(1, len(inlier_pts0) // 40)

    for i in range(0, len(inlier_pts0), step):
        p0 = (int(round(inlier_pts0[i][0])), int(round(inlier_pts0[i][1])))
        p1 = (int(round(inlier_pts1[i][0] + w_std)), int(round(inlier_pts1[i][1])))

        cv2.circle(vis_matches, p0, 3, (255, 50, 50), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis_matches, p1, 3, (255, 50, 50), -1, lineType=cv2.LINE_AA)
        cv2.line(vis_matches, p0, p1, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    return aligned_target, overlay_blend, vis_matches, inlier_count, inlier_ratio, rmse_value
