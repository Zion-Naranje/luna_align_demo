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
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def run_luna_align(ref_path, target_path):
    try:
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        if ref_img is None or target_img is None:
            return None, 0, 0.0

        h_ref, w_ref = ref_img.shape
        ref_clahe = apply_clahe(ref_img)
        target_clahe = apply_clahe(target_img)

        t_ref = torch.from_numpy(ref_clahe).float().unsqueeze(0).unsqueeze(0) / 255.0
        t_target = torch.from_numpy(target_clahe).float().unsqueeze(0).unsqueeze(0) / 255.0

        if t_ref.shape != t_target.shape:
            t_target = K.geometry.transform.resize(t_target, (t_ref.shape[2], t_ref.shape[3]))

        t_ref = t_ref.to(DEVICE)
        t_target = t_target.to(DEVICE)

        matcher = _get_matcher()
        with torch.inference_mode():
            corrs = matcher({"image0": t_ref, "image1": t_target})

        pts0 = corrs["keypoints0"].cpu().numpy()
        pts1 = corrs["keypoints1"].cpu().numpy()

        if len(pts0) < 4:
            return None, 0, 0.0

        H, mask = cv2.findHomography(pts1, pts0, cv2.USAC_MAGSAC, 3.0)
        if H is None:
            return None, 0, 0.0

        inliers = mask.ravel() == 1
        inlier_count = int(np.sum(inliers))
        inlier_ratio = (inlier_count / len(pts0)) * 100

        aligned_target = cv2.warpPerspective(target_img, H, (w_ref, h_ref))
        return aligned_target, inlier_count, inlier_ratio

    except Exception:
        return None, 0, 0.0
