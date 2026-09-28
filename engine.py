import os
import cv2
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MATCHER = None

def _get_matcher():
    """Lazy-load LoFTR matcher to prevent Streamlit boot errors."""
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
    """Normalize extreme shadow/rim contrast."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def compute_rmse(pts_target, pts_ref, H, inliers):
    """Calculates actual RMSE in pixels for verified inliers."""
    tgt_inliers = pts_target[inliers]
    ref_inliers = pts_ref[inliers]

    tgt_homo = np.hstack([tgt_inliers, np.ones((tgt_inliers.shape[0], 1))])
    projected = (H @ tgt_homo.T).T
    projected = projected[:, :2] / projected[:, 2:]

    errors = np.linalg.norm(projected - ref_inliers, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return round(rmse, 2)

def run_sift_match(ref_img, target_img):
    """Classical SIFT baseline comparison."""
    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(ref_img, None)
    kp1, des1 = sift.detectAndCompute(target_img, None)

    if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
        canvas = np.hstack((ref_img, target_img))
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB), 0, 0

    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des0, des1, k=2)
    good_matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

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

def _match_single_orientation(t_ref, t_tgt, target_img_rot, ref_img_std, w_std, h_std):
    matcher = _get_matcher()
    with torch.inference_mode():
        corrs = matcher({"image0": t_ref, "image1": t_tgt})

    pts0 = corrs["keypoints0"].cpu().numpy()
    pts1 = corrs["keypoints1"].cpu().numpy()

    # Discard matches landing on pure black void (< 15 intensity)
    valid_idx = []
    for i in range(len(pts0)):
        x0, y0 = int(round(pts0[i][0])), int(round(pts0[i][1]))
        x1, y1 = int(round(pts1[i][0])), int(round(pts1[i][1]))
        if (0 <= x0 < w_std and 0 <= y0 < h_std and 0 <= x1 < w_std and 0 <= y1 < h_std):
            if ref_img_std[y0, x0] > 15 and target_img_rot[y1, x1] > 15:
                valid_idx.append(i)

    if len(valid_idx) < 4:
        return 0, 0.0, 0.0, None, None, None

    pts0_v = pts0[valid_idx]
    pts1_v = pts1[valid_idx]

    H, mask = cv2.findHomography(pts1_v, pts0_v, cv2.USAC_MAGSAC, 4.0)
    if H is None:
        return 0, 0.0, 0.0, None, None, None

    inliers = mask.ravel() == 1
    inlier_count = int(np.sum(inliers))
    ratio = (inlier_count / len(valid_idx)) * 100
    rmse = compute_rmse(pts1_v, pts0_v, H, inliers)

    return inlier_count, ratio, rmse, H, pts0_v[inliers], pts1_v[inliers]

def run_luna_align(ref_path, target_path):
    """
    Robust Multi-Sensor Registration:
    - Automatically tests 4 orbital rotations (0, 90, 180, 270) to counter along-track sensor tilt
    - Rejects black background void pixels
    - Selects the optimal orientation that yields maximum inliers
    """
    if not os.path.exists(ref_path) or not os.path.exists(target_path):
        raise FileNotFoundError(f"Missing file: {ref_path} or {target_path}")

    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or target_img is None:
        raise ValueError("Could not decode image files.")

    h_std, w_std = 640, 640
    ref_img_std = cv2.resize(ref_img, (w_std, h_std))
    ref_clahe = apply_clahe(ref_img_std)
    t_ref = torch.from_numpy(ref_clahe).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0

    rotations = [
        (0, None),
        (90, cv2.ROTATE_90_CLOCKWISE),
        (180, cv2.ROTATE_180),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    best_inliers = -1
    best_res = None
    best_tgt_img = None

    for angle, rot_flag in rotations:
        tgt_rot = target_img.copy()
        if rot_flag is not None:
            tgt_rot = cv2.rotate(tgt_rot, rot_flag)
        
        tgt_rot_std = cv2.resize(tgt_rot, (w_std, h_std))
        tgt_clahe = apply_clahe(tgt_rot_std)
        t_tgt = torch.from_numpy(tgt_clahe).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0

        inliers, ratio, rmse, H, p0_in, p1_in = _match_single_orientation(
            t_ref, t_tgt, tgt_rot_std, ref_img_std, w_std, h_std
        )

        if inliers > best_inliers:
            best_inliers = inliers
            best_res = (inliers, ratio, rmse, H, p0_in, p1_in)
            best_tgt_img = tgt_rot_std

    if best_inliers < 12 or best_res[3] is None:
        return None, None, None, max(0, best_inliers), 0.0, 0.0

    inlier_count, inlier_ratio, rmse_val, H, in_pts0, in_pts1 = best_res

    # Warp optimal target frame
    aligned_target = cv2.warpPerspective(best_tgt_img, H, (w_std, h_std))
    overlay_blend = cv2.addWeighted(ref_img_std, 0.5, aligned_target, 0.5, 0)

    # Draw correspondence vectors
    canvas = np.hstack((ref_img_std, best_tgt_img))
    vis_matches = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    step = max(1, len(in_pts0) // 40)

    for i in range(0, len(in_pts0), step):
        p0 = (int(round(in_pts0[i][0])), int(round(in_pts0[i][1])))
        p1 = (int(round(in_pts1[i][0] + w_std)), int(round(in_pts1[i][1])))
        cv2.circle(vis_matches, p0, 3, (255, 50, 50), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis_matches, p1, 3, (255, 50, 50), -1, lineType=cv2.LINE_AA)
        cv2.line(vis_matches, p0, p1, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    return aligned_target, overlay_blend, vis_matches, inlier_count, inlier_ratio, rmse_val
