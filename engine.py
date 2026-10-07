"""
LunaAlign Engine v3 -- Contextual Area Verification Pipeline
====================================================================
Core Methodology (Local Dot-Judging → Contextual Area Verification):

1. Let LoFTR Gather Freely
   Dense matching across the full tile. NO local brightness gate, NO hard
   confidence pre-filter. conf >= 0.35 and dedup=1px are the only guards.
   All surviving candidates flow directly to MAGSAC++.

2. Global Geometry Context Check (MAGSAC++)
   MAGSAC++ tests all candidates against a single coherent spatial transform
   (Similarity → Affine → Homography). Non-overlapping/hallucinated pairs
   are rejected by three region-level signals:
     • total inliers < 6
     • inlier ratio  < 1.0%
     • SVD-derived scale outside (0.01×, 100×) OR aspect ratio > 50

3. Feature-Anchored Proof Scoring (post-geometry only)
   Geometry-verified inliers are cross-referenced with native-resolution
   Canny/Sobel crater edges via terrain_gradient_score(). Shadow-adjacent
   matches are flagged (not dropped) for human review.

4. Human-in-the-Loop Validation
   Results are passed to the Streamlit inspector. The user reviews the global
   layout and raw/CLAHE/rim-overlay proofs before final sign-off.

Changes from v2:
- Removed conf pre-filter gate: candidates are no longer dropped before MAGSAC.
- Removed lit_filter pre-filter gate: shadow-adjacent matches survive if geometry agrees.
- Fixed scale check: uses SVD singular values (not det, which is area not scale).
- Indoor retry: replaced conf+lit gates with padded-zone rejection only.
- Added terrain_gradient_score() for clean, standalone Sobel proof scoring.
- Funnel now tracks: total_raw, total_padzone_ok, total_geom_inliers.
"""

import os, time
import cv2
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MATCHERS = {}

# Tunable constants
LOFTR_SIZE      = 640
LOFTR_CONF      = 0.35   # lowered: lunar low-contrast matches often 0.35-0.49
LIT_FRAC        = 0.08   # lowered: crater interiors are darker; 8% of median
LIT_PATCH       = 5      # half-side of lit-check patch (px, loftr-space)
MAGSAC_THRESH   = 8.0    # fixed reprojection threshold (px, loftr-space)
DEDUP_RADIUS    = 1.0    # lowered: 2px was killing tight crater-cluster matches
CROP_PX         = 64
DISPLAY_LONG    = 640
MIN_EDGE_LEN    = 8
RIM_DILATE      = 3      # lowered: reduce over-dilation in dense crater fields
MAX_RIM_FRAC    = 0.55   # raised: dense crater fields need higher tolerance
MIN_INLIERS_HOM = 12


# --- Model (cached) -----------------------------------------------------------

def _get_matcher(pretrained="outdoor"):
    global _MATCHERS
    if pretrained not in _MATCHERS:
        try:
            from kornia.feature import LoFTR
        except ImportError:
            import kornia.feature as KF
            LoFTR = KF.LoFTR
        m = LoFTR(pretrained=pretrained).to(DEVICE).eval()
        _MATCHERS[pretrained] = m
    return _MATCHERS[pretrained]


# --- Preprocessing ------------------------------------------------------------

def apply_clahe(img, clip=4.0, grid=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(img)


def preprocess(img, size=LOFTR_SIZE):
    """
    Aspect-ratio preserving resize -> CONSTANT-zero pad -> CLAHE.

    Returns
    -------
    clahe_img, raw_padded, scale_x, scale_y, pad_left, pad_top, new_w, new_h
    """
    h, w = img.shape[:2]
    sc   = size / max(h, w)
    new_h = max(32, (int(h*sc)//32)*32)
    new_w = max(32, (int(w*sc)//32)*32)
    sx, sy = new_w/w, new_h/h

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ph = size - new_h;  pt = ph//2
    pw = size - new_w;  pl = pw//2

    raw_pad = cv2.copyMakeBorder(
        resized, pt, ph-pt, pl, pw-pl, cv2.BORDER_CONSTANT, value=0)
    return apply_clahe(raw_pad, 4.0, 8), raw_pad, sx, sy, pl, pt, new_w, new_h


def _in_padded_zone(pts, pad_l, pad_t, new_w, new_h):
    x, y = pts[:,0], pts[:,1]
    return (x < pad_l)|(x >= pad_l+new_w)|(y < pad_t)|(y >= pad_t+new_h)


# --- Duplicate removal --------------------------------------------------------

def dedup_matches(pts0, pts1, conf, radius=DEDUP_RADIUS):
    if len(pts0) == 0:
        return pts0, pts1, conf
    order = np.argsort(-conf)
    pts0, pts1, conf = pts0[order], pts1[order], conf[order]
    keep = np.ones(len(pts0), bool)
    for i in range(len(pts0)):
        if not keep[i]:
            continue
        diff  = pts0[i+1:] - pts0[i]
        dists = np.sqrt((diff**2).sum(1))
        keep[i+1+np.where(dists < radius)[0]] = False
    return pts0[keep], pts1[keep], conf[keep]


# --- Crater edge mask (fixed) -------------------------------------------------

def crater_edge_mask(raw_img):
    """
    Detect crater rims at native resolution.
    1. Light Gaussian blur.
    2. Otsu-based auto Canny thresholds.
    3. Keep only long connected edges (>= MIN_EDGE_LEN px).
    4. Small dilation (RIM_DILATE px).
    5. Sanity: if mask > MAX_RIM_FRAC of image, return empty.
    """
    H, W = raw_img.shape[:2]
    blurred = cv2.GaussianBlur(raw_img, (5,5), 1.0)

    g32 = blurred.astype(np.float32)
    gx  = cv2.Sobel(g32, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(g32, cv2.CV_32F, 0, 1, ksize=3)
    mag8 = np.clip(cv2.magnitude(gx, gy), 0, 255).astype(np.uint8)
    ov, _ = cv2.threshold(mag8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    lo, hi = max(10.0, ov*0.5), min(200.0, ov*1.5)

    edges = cv2.Canny(blurred, lo, hi)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    long_e = np.zeros_like(edges)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= MIN_EDGE_LEN:
            long_e[labels==lbl] = 255

    kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*RIM_DILATE+1, 2*RIM_DILATE+1))
    rim = cv2.dilate(long_e, kern)

    if rim.astype(bool).sum()/(H*W) > MAX_RIM_FRAC:
        return np.zeros((H, W), np.uint8)
    return rim.astype(np.uint8)


# --- Lit filter (patch mean, relative threshold) ------------------------------

def _patch_mean(raw_img, pts, half=LIT_PATCH):
    H, W = raw_img.shape[:2]
    means = np.empty(len(pts), np.float32)
    for k, (x, y) in enumerate(pts):
        cx = int(np.clip(round(x), 0, W-1))
        cy = int(np.clip(round(y), 0, H-1))
        x0,x1 = max(0,cx-half), min(W,cx+half+1)
        y0,y1 = max(0,cy-half), min(H,cy+half+1)
        means[k] = float(raw_img[y0:y1, x0:x1].mean())
    return means


def lit_filter(pts0, pts1, conf, ref_pad, tgt_pad, lit_frac=LIT_FRAC):
    """
    Returns (pts0, pts1, conf, warning_str).
    Shadow = patch mean < lit_frac * per-image median.
    If <4 lit matches: returns unfiltered + warning (never silently skips).
    """
    rt = float(np.median(ref_pad)) * lit_frac
    tt = float(np.median(tgt_pad)) * lit_frac
    m0 = _patch_mean(ref_pad, pts0)
    m1 = _patch_mean(tgt_pad, pts1)
    lit = (m0 >= rt) & (m1 >= tt)
    if lit.sum() < 4:
        warn = (f"Only {int(lit.sum())} lit matches "
                f"(ref_thr={rt:.1f}, tgt_thr={tt:.1f}). "
                "Shadow filter bypassed -- verify manually.")
        return pts0, pts1, conf, warn
    return pts0[lit], pts1[lit], conf[lit], ""


# --- Terrain gradient scoring (Stage 3 — Feature-Anchored Proof) --------------

def terrain_gradient_score(raw_img, x, y, patch=24):
    """
    Compute mean Sobel gradient magnitude at native resolution around (x, y).
    High gradient = crater rim / wall = strong structural anchor.
    Used by classify_match_point after geometry is already confirmed.

    Returns
    -------
    score : int  0-100
    gradient_mean : float  (raw Sobel mean in pixel units)
    """
    H, W = raw_img.shape[:2]
    cx = int(np.clip(round(x), 0, W-1))
    cy = int(np.clip(round(y), 0, H-1))
    x0, x1 = max(0, cx-patch), min(W, cx+patch)
    y0, y1 = max(0, cy-patch), min(H, cy+patch)
    rp = raw_img[y0:y1, x0:x1].astype(np.float32)
    if rp.size == 0:
        return 0, 0.0
    gx = cv2.Sobel(rp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(rp, cv2.CV_32F, 0, 1, ksize=3)
    gm = float(np.mean(cv2.magnitude(gx, gy)))
    return min(int(gm * 2), 100), gm


# --- Match-point classification (post-geometry only) --------------------------

def classify_match_point(raw_img, clahe_img, x, y, rim_mask,
                          lit_thresh=0.0, patch=24):
    """
    Classify a geometry-verified inlier point as a terrain feature type.
    Called AFTER MAGSAC++ — this is proof annotation, not filtering.
    """
    H, W = raw_img.shape[:2]
    cx = int(np.clip(round(x), 0, W-1))
    cy = int(np.clip(round(y), 0, H-1))
    x0,x1 = max(0,cx-patch), min(W,cx+patch)
    y0,y1 = max(0,cy-patch), min(H,cy+patch)
    rp = raw_img[y0:y1, x0:x1].astype(np.float32)
    cp = clahe_img[y0:y1, x0:x1].astype(np.float32)
    rm = float(rp.mean()) if rp.size else 0.0

    if rm < lit_thresh:
        sc = float(cp.std()) if cp.size else 0.0
        if sc > 18:
            return "shadow (CLAHE proof)", True, int(min(sc*2, 60))
        return "shadow (no structure)", True, 0

    on_rim = bool(rim_mask[cy, cx] > 0)
    # Use terrain_gradient_score instead of inline Sobel block
    _, gm = terrain_gradient_score(raw_img, x, y, patch)
    ls = float(rp.std()) if rp.size else 0.0

    near_shad = any(
        0<=cx+dx<W and 0<=cy+dy<H and raw_img[cy+dy, cx+dx] < lit_thresh
        for dy in range(-20,21,5) for dx in range(-20,21,5)
    )
    if on_rim and gm > 20:
        return "crater rim",  False, min(80+int(gm), 100)
    if gm > 15 and ls > 12:
        return "crater wall", False, min(60+int(gm),  90)
    if near_shad:
        return "near shadow", False, 50
    return "lit terrain",    False, 70


# --- LoFTR passes -------------------------------------------------------------

def _loftr_pass(ref_c, tgt_c, pretrained="outdoor"):
    m = _get_matcher(pretrained)
    def _t(x):
        return torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(DEVICE)/255.
    with torch.inference_mode():
        out = m({"image0": _t(ref_c), "image1": _t(tgt_c)})
    p0 = out["keypoints0"].cpu().numpy()
    p1 = out["keypoints1"].cpu().numpy()
    cf = out.get("confidence", torch.ones(len(p0))).cpu().numpy()
    return p0, p1, cf


def _tiled_loftr_pass(ref_c, tgt_c, pretrained="outdoor",
                       tile_size=480, overlap=0.25):
    H, W   = ref_c.shape
    stride = int(tile_size*(1-overlap))
    all_p0, all_p1, all_cf = [], [], []
    m = _get_matcher(pretrained)
    def _t(x):
        return torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(DEVICE)/255.

    for y0 in range(0, H-tile_size//2, stride):
        for x0 in range(0, W-tile_size//2, stride):
            y1,x1 = min(y0+tile_size,H), min(x0+tile_size,W)
            th,tw = y1-y0, x1-x0
            if th < tile_size//2 or tw < tile_size//2:
                continue
            ph = ((th+31)//32)*32;  pw = ((tw+31)//32)*32
            rp = cv2.copyMakeBorder(ref_c[y0:y1,x0:x1], 0,ph-th,0,pw-tw,
                                    cv2.BORDER_CONSTANT, value=0)
            tp = cv2.copyMakeBorder(tgt_c[y0:y1,x0:x1], 0,ph-th,0,pw-tw,
                                    cv2.BORDER_CONSTANT, value=0)
            try:
                with torch.inference_mode():
                    out = m({"image0": _t(rp), "image1": _t(tp)})
                lp0 = out["keypoints0"].cpu().numpy()
                lp1 = out["keypoints1"].cpu().numpy()
                lc  = out.get("confidence", torch.ones(len(lp0))).cpu().numpy()
            except Exception:
                continue
            if len(lp0) == 0:
                continue
            valid = (lp0[:,0]<tw)&(lp0[:,1]<th)&(lp1[:,0]<tw)&(lp1[:,1]<th)
            lp0,lp1,lc = lp0[valid],lp1[valid],lc[valid]
            if len(lp0) == 0:
                continue
            lp0 = lp0.copy(); lp1 = lp1.copy()
            lp0[:,0]+=x0; lp0[:,1]+=y0
            lp1[:,0]+=x0; lp1[:,1]+=y0
            all_p0.append(lp0); all_p1.append(lp1); all_cf.append(lc)

    if not all_p0:
        return np.zeros((0,2),np.float32), np.zeros((0,2),np.float32), np.zeros(0,np.float32)
    return np.vstack(all_p0), np.vstack(all_p1), np.concatenate(all_cf)


# --- Transform fitting --------------------------------------------------------

def _hom_sane(H, w, h):
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    m       = cv2.perspectiveTransform(corners, H).reshape(-1,2)
    return np.all(np.abs(m) < max(w,h)*2) and np.linalg.det(H[:2,:2]) > 0.1


def fit_transform(pts0, pts1, cw, ch, thr=MAGSAC_THRESH):
    """
    similarity -> affine -> homography (homography gated on sanity).
    Returns (H_3x3, inlier_bool_mask, model_name) or (None, None, "failed").
    """
    best_H, best_mask, best_cnt, best_name = None, None, 0, "failed"

    if len(pts0) >= 3:
        A, mask = cv2.estimateAffinePartial2D(pts1, pts0,
            method=cv2.RANSAC, ransacReprojThreshold=thr,
            confidence=0.999, maxIters=10000)
        if A is not None:
            cnt = int(mask.ravel().sum())
            if cnt > best_cnt:
                best_H, best_mask, best_cnt, best_name = (
                    np.vstack([A,[0,0,1]]), mask.ravel()==1, cnt, "similarity")

    if len(pts0) >= 4:
        A, mask = cv2.estimateAffine2D(pts1, pts0,
            method=cv2.RANSAC, ransacReprojThreshold=thr,
            confidence=0.999, maxIters=10000)
        if A is not None:
            cnt = int(mask.ravel().sum())
            if cnt > best_cnt:
                best_H, best_mask, best_cnt, best_name = (
                    np.vstack([A,[0,0,1]]), mask.ravel()==1, cnt, "affine")

    if len(pts0) >= MIN_INLIERS_HOM:
        Hom, mask = cv2.findHomography(pts1, pts0, cv2.USAC_MAGSAC, thr,
                                        confidence=0.999, maxIters=10000)
        if Hom is not None:
            cnt = int(mask.ravel().sum())
            if cnt >= MIN_INLIERS_HOM and _hom_sane(Hom, cw, ch) and cnt > best_cnt:
                best_H, best_mask, best_cnt, best_name = (
                    Hom, mask.ravel()==1, cnt, "homography")

    if best_H is None:
        return None, None, "failed"
    return best_H, best_mask, best_name


# --- RMSE (held-out 20%) ------------------------------------------------------

def compute_rmse(pts_tgt, pts_ref, H, inliers, px_per_m=None):
    inl_idx = np.where(inliers)[0]
    if len(inl_idx) < 5:
        check = inl_idx
    else:
        rng   = np.random.default_rng(42)
        check = rng.choice(inl_idx, max(1, len(inl_idx)//5), replace=False)
    ti   = pts_tgt[check];  ri = pts_ref[check]
    th   = np.hstack([ti, np.ones((len(ti),1))])
    proj = (H @ th.T).T
    proj = proj[:,:2]/proj[:,2:]
    rmse = float(np.sqrt(np.mean(np.sum((proj-ri)**2, axis=1))))
    return round(rmse,2), (round(rmse/px_per_m,3) if px_per_m else None), len(check)


# --- Spatial coverage ---------------------------------------------------------

def spatial_coverage(pts, w, h, bins=4):
    if len(pts) == 0:
        return 0.0
    col = np.clip((pts[:,0]/w*bins).astype(int), 0, bins-1)
    row = np.clip((pts[:,1]/h*bins).astype(int), 0, bins-1)
    occ = len(set(zip(row.tolist(), col.tolist())))
    return round(occ/(bins*bins), 3)


# --- SIFT benchmark -----------------------------------------------------------

def run_sift_match(ref_img, target_img):
    """Preserves aspect ratio; uses same MAGSAC++ threshold as LoFTR pipeline."""
    def _rkpar(im):
        h,w = im.shape[:2]; sc = DISPLAY_LONG/max(h,w)
        return cv2.resize(im, (int(w*sc), int(h*sc)), cv2.INTER_AREA)
    rs = _rkpar(ref_img); ts = _rkpar(target_img)

    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02, edgeThreshold=10)
    kp0,des0 = sift.detectAndCompute(rs, None)
    kp1,des1 = sift.detectAndCompute(ts, None)
    if des0 is None or des1 is None or len(kp0)<4 or len(kp1)<4:
        return cv2.cvtColor(np.hstack((rs,ts)), cv2.COLOR_GRAY2RGB), 0, 0

    good = [m for m,n in cv2.BFMatcher().knnMatch(des0,des1,k=2)
            if m.distance < 0.75*n.distance]
    if len(good) < 4:
        return cv2.cvtColor(np.hstack((rs,ts)), cv2.COLOR_GRAY2RGB), len(good), 0

    p0 = np.float32([kp0[m.queryIdx].pt for m in good])
    p1 = np.float32([kp1[m.trainIdx].pt for m in good])
    Hs, mask = cv2.findHomography(p1, p0, cv2.USAC_MAGSAC, MAGSAC_THRESH,
                                   confidence=0.999, maxIters=10000)
    mh = max(rs.shape[0], ts.shape[0])
    def _pad(im):
        bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        return cv2.copyMakeBorder(bgr, 0, mh-im.shape[0], 0,0,
                                  cv2.BORDER_CONSTANT, value=0)
    vis = np.hstack((_pad(rs), _pad(ts)))
    w0  = rs.shape[1]; inlier_count = 0
    if mask is not None:
        inls = mask.ravel()==1; inlier_count = int(inls.sum())
        for i,(a,b) in enumerate(zip(p0,p1)):
            if inls[i]:
                cv2.circle(vis,(int(a[0]),int(a[1])),3,(255,60,60),-1)
                cv2.circle(vis,(int(b[0])+w0,int(b[1])),3,(255,60,60),-1)
                cv2.line(vis,(int(a[0]),int(a[1])),
                          (int(b[0])+w0,int(b[1])),(0,255,255),1,cv2.LINE_AA)
    return vis, len(good), inlier_count


# --- Visualization helpers ----------------------------------------------------

def _make_palette(n):
    colors = []
    for i in range(max(n,1)):
        hue = int(180*i/max(n,1))
        bgr = cv2.cvtColor(np.uint8([[[hue,230,220]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]),int(bgr[1]),int(bgr[2])))
    return colors

def _spread_pts(pts, n, conf=None):
    """
    Spatial-grid sampling: divide the point cloud into an n-cell grid,
    pick the highest-confidence match in each occupied cell.
    Falls back to index list of all points when n >= m.
    This ensures crater clusters (many close points) are represented,
    not just the spatially spread-out outliers.
    """
    m = len(pts)
    if m <= n:
        return list(range(m))
    if conf is None:
        conf = np.ones(m, np.float32)

    # Determine grid size that yields ~n cells
    grid_side = max(1, int(np.ceil(np.sqrt(n))))
    xs = pts[:, 0]; ys = pts[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = max(x_max - x_min, 1)
    y_range = max(y_max - y_min, 1)

    col = np.clip(((xs - x_min) / x_range * grid_side).astype(int), 0, grid_side - 1)
    row = np.clip(((ys - y_min) / y_range * grid_side).astype(int), 0, grid_side - 1)

    cells = {}
    for i in range(m):
        key = (int(row[i]), int(col[i]))
        if key not in cells or conf[i] > conf[cells[key]]:
            cells[key] = i

    sel = list(cells.values())

    # If we have more cells than n, trim to highest-conf n
    if len(sel) > n:
        sel = sorted(sel, key=lambda i: -conf[i])[:n]
    # If we have fewer than n, fill remaining by conf order (avoid duplicates)
    elif len(sel) < n:
        existing = set(sel)
        extras = sorted([i for i in range(m) if i not in existing],
                        key=lambda i: -conf[i])
        sel += extras[:n - len(sel)]

    return sel

def _crop_patch(img, cx, cy, half):
    H,W = img.shape[:2]
    x0,x1 = max(0,cx-half), min(W,cx+half)
    y0,y1 = max(0,cy-half), min(H,cy+half)
    c = img[y0:y1, x0:x1].copy()
    pb = (2*half)-c.shape[0]; pr = (2*half)-c.shape[1]
    if pb>0 or pr>0:
        c = cv2.copyMakeBorder(c,0,max(0,pb),0,max(0,pr),cv2.BORDER_CONSTANT,value=0)
    return c[:2*half, :2*half]

def _corner_brackets(img, cx, cy, arm=16, gap=7, color=(255,255,255), t=2):
    for sx,sy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
        cv2.line(img,(cx+sx*gap,cy+sy*gap),(cx+sx*(gap+arm),cy+sy*gap),color,t,cv2.LINE_AA)
        cv2.line(img,(cx+sx*gap,cy+sy*gap),(cx+sx*gap,cy+sy*(gap+arm)),color,t,cv2.LINE_AA)

def _label_color(label):
    if "crater rim"    in label: return (0,255,100)
    if "crater wall"   in label: return (0,200,255)
    if "near shadow"   in label: return (0,165,255)
    if "shadow (CLAHE" in label: return (100,100,255)
    if "shadow (no"    in label: return (50,50,200)
    return (200,200,200)


# --- Match proof builder ------------------------------------------------------

def build_match_proof(ref_disp, tgt_disp,
                      ref_clahe_disp, tgt_clahe_disp,
                      all_pts0, all_pts1,
                      ref_rim_disp, tgt_rim_disp,
                      palette, match_meta, selected_idx):
    W_ref = ref_disp.shape[1]
    mh    = max(ref_disp.shape[0], tgt_disp.shape[0])
    half  = CROP_PX

    def _ph(img, h):
        if img.shape[0] < h:
            return cv2.copyMakeBorder(img,0,h-img.shape[0],0,0,
                                      cv2.BORDER_CONSTANT,value=(10,10,10))
        return img

    comp = np.hstack((_ph(cv2.cvtColor(ref_disp,cv2.COLOR_GRAY2BGR),mh),
                      _ph(cv2.cvtColor(tgt_disp,cv2.COLOR_GRAY2BGR),mh)))

    for i,(p0,p1) in enumerate(zip(all_pts0,all_pts1)):
        if i==selected_idx: continue
        cv2.line(comp,(int(round(p0[0])),int(round(p0[1]))),
                      (int(round(p1[0]))+W_ref,int(round(p1[1]))),(22,22,22),1,cv2.LINE_AA)
        cv2.circle(comp,(int(round(p0[0])),int(round(p0[1]))),2,(45,45,45),-1)
        cv2.circle(comp,(int(round(p1[0]))+W_ref,int(round(p1[1]))),2,(45,45,45),-1)

    color = palette[selected_idx % len(palette)]
    sx0 = int(round(all_pts0[selected_idx][0])); sy0 = int(round(all_pts0[selected_idx][1]))
    sx1 = int(round(all_pts1[selected_idx][0]))+W_ref; sy1 = int(round(all_pts1[selected_idx][1]))
    cv2.line(comp,(sx0,sy0),(sx1,sy1),color,2,cv2.LINE_AA)
    _corner_brackets(comp,sx0,sy0,arm=12,gap=6,color=color,t=2)
    _corner_brackets(comp,sx1,sy1,arm=12,gap=6,color=color,t=2)

    tx = int(round(all_pts1[selected_idx][0])); ty = int(round(all_pts1[selected_idx][1]))

    def _zoom(gray, cx, cy):
        bgr = cv2.cvtColor(_crop_patch(gray, cx, cy, half), cv2.COLOR_GRAY2BGR)
        return cv2.resize(bgr, (half*4, half*4), interpolation=cv2.INTER_NEAREST)

    raw_ref   = _zoom(ref_disp, sx0, sy0)
    raw_tgt   = _zoom(tgt_disp, tx, ty)
    clahe_ref = _zoom(ref_clahe_disp, sx0, sy0)
    clahe_tgt = _zoom(tgt_clahe_disp, tx, ty)

    def _rim_ov(cc, rm, cx, cy):
        ru = cv2.resize(_crop_patch(rm,cx,cy,half),(half*4,half*4),cv2.INTER_NEAREST)
        o  = cc.copy(); o[ru>0] = (0,220,80); return o

    rim_ref_crop = _rim_ov(clahe_ref, ref_rim_disp, sx0, sy0)
    rim_tgt_crop = _rim_ov(clahe_tgt, tgt_rim_disp, tx, ty)

    cz = half*2
    for c in (raw_ref, raw_tgt):
        _corner_brackets(c,cz,cz,arm=14,gap=6,color=color,t=2)
    for c in (clahe_ref, clahe_tgt, rim_ref_crop, rim_tgt_crop):
        _corner_brackets(c,cz,cz,arm=14,gap=6,color=(0,255,180),t=2)

    meta = dict(match_meta[selected_idx])
    meta["raw_brightness_ref"] = int(ref_disp[min(sy0,ref_disp.shape[0]-1),
                                               min(sx0,ref_disp.shape[1]-1)])
    meta["raw_brightness_tgt"] = int(tgt_disp[min(ty,tgt_disp.shape[0]-1),
                                               min(tx,tgt_disp.shape[1]-1)])
    return comp, raw_ref, raw_tgt, clahe_ref, clahe_tgt, rim_ref_crop, rim_tgt_crop, meta


# --- Main pipeline ------------------------------------------------------------

def run_luna_align(ref_path, target_path,
                   ref_px_per_m=None, tgt_px_per_m=None):
    """
    Returns a dict.  New keys vs old version:
      rmse_px, rmse_m, n_checkpts, spatial_coverage,
      model_used, lit_warning, timing, funnel.
    """
    if not os.path.exists(ref_path) or not os.path.exists(target_path):
        raise FileNotFoundError(f"Missing: {ref_path} or {target_path}")

    timing = {}
    t0_total = time.perf_counter()

    ref_raw = cv2.imread(ref_path,    cv2.IMREAD_GRAYSCALE)
    tgt_raw = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    if ref_raw is None or tgt_raw is None:
        raise ValueError("Could not decode images.")

    # Step 1: Preprocess
    t0 = time.perf_counter()
    ref_clahe, ref_raw_pad, ref_sx, ref_sy, ref_pl, ref_pt, ref_nw, ref_nh = \
        preprocess(ref_raw, LOFTR_SIZE)
    tgt_clahe, tgt_raw_pad, tgt_sx, tgt_sy, tgt_pl, tgt_pt, tgt_nw, tgt_nh = \
        preprocess(tgt_raw, LOFTR_SIZE)
    timing["preprocess_ms"] = round((time.perf_counter()-t0)*1000)

    def _disp(img):
        h,w = img.shape[:2]; sc = DISPLAY_LONG/max(h,w)
        return cv2.resize(img,(int(w*sc),int(h*sc)),cv2.INTER_AREA), sc

    ref_disp, r_dsc = _disp(ref_raw)
    tgt_disp, t_dsc = _disp(tgt_raw)
    ref_c_disp = cv2.resize(apply_clahe(ref_disp,4.0,8),(ref_disp.shape[1],ref_disp.shape[0]))
    tgt_c_disp = cv2.resize(apply_clahe(tgt_disp,4.0,8),(tgt_disp.shape[1],tgt_disp.shape[0]))

    # Step 2: Crater masks
    t0 = time.perf_counter()
    ref_rim_full = crater_edge_mask(ref_raw_pad)
    tgt_rim_full = crater_edge_mask(tgt_raw_pad)
    timing["crater_edge_ms"] = round((time.perf_counter()-t0)*1000)

    ref_rim_disp = cv2.resize(ref_rim_full,(ref_disp.shape[1],ref_disp.shape[0]),cv2.INTER_NEAREST)
    tgt_rim_disp = cv2.resize(tgt_rim_full,(tgt_disp.shape[1],tgt_disp.shape[0]),cv2.INTER_NEAREST)

    # Step 3: LoFTR -- run once; tile only when needed
    t0 = time.perf_counter()
    p0f, p1f, cf = _loftr_pass(ref_clahe, tgt_clahe, "outdoor")
    timing["loftr_full_ms"] = round((time.perf_counter()-t0)*1000)

    if (cf >= LOFTR_CONF).sum() < 60:  # raised threshold: tile more aggressively
        t0 = time.perf_counter()
        p0t, p1t, ct = _tiled_loftr_pass(ref_clahe, tgt_clahe, "outdoor", 480, 0.25)
        timing["loftr_tiled_ms"] = round((time.perf_counter()-t0)*1000)
        if len(p0f)>0 and len(p0t)>0:
            pts0_all = np.vstack([p0f,p0t]); pts1_all = np.vstack([p1f,p1t])
            conf_all = np.concatenate([cf,ct])
        elif len(p0t)>0:
            pts0_all, pts1_all, conf_all = p0t, p1t, ct
        else:
            pts0_all, pts1_all, conf_all = p0f, p1f, cf
    else:
        pts0_all, pts1_all, conf_all = p0f, p1f, cf

    # Stage 1 — Step 4: Dedup (1px radius, confidence-ordered)
    pts0_all, pts1_all, conf_all = dedup_matches(pts0_all, pts1_all, conf_all)
    total_raw = len(pts0_all)

    # Stage 1 — Step 5 REMOVED: confidence pre-filter abolished.
    # All deduped candidates flow directly to MAGSAC++ (Stage 2).
    # The local conf >= LOFTR_CONF threshold is used only for post-geometry
    # metadata annotation — it is NOT a gate on who enters the geometry check.

    # Stage 1 — Step 6: Padded-zone rejection (geometry-neutral; removes
    # points that land in the black constant-border padding added by preprocess)
    if len(pts0_all) > 0:
        bad = (_in_padded_zone(pts0_all, ref_pl, ref_pt, ref_nw, ref_nh) |
               _in_padded_zone(pts1_all, tgt_pl, tgt_pt, tgt_nw, tgt_nh))
        ok = ~bad
        if ok.sum() >= 4:
            pts0_all, pts1_all, conf_all = pts0_all[ok], pts1_all[ok], conf_all[ok]
    total_padzone_ok = len(pts0_all)

    # Stage 3 prep — lit thresholds computed here for post-geometry proof scoring.
    # lit_filter is NO LONGER called as a candidate gate before MAGSAC.
    # Shadow-adjacent matches that pass global geometry are preserved and
    # annotated with is_shadow=True in match_meta for the human inspector.
    t0 = time.perf_counter()
    lit_warning = ""
    ref_lit_thr = float(np.median(ref_raw_pad)) * LIT_FRAC
    tgt_lit_thr = float(np.median(tgt_raw_pad)) * LIT_FRAC
    timing["filter_ms"] = round((time.perf_counter()-t0)*1000)

    pts0, pts1 = pts0_all, pts1_all
    total = len(pts0)

    # Step 8: Fit transform
    t0 = time.perf_counter()
    best_H, best_mask, model_name = None, None, "failed"
    if total >= 4:
        best_H, best_mask, model_name = fit_transform(pts0, pts1,
                                                       LOFTR_SIZE, LOFTR_SIZE,
                                                       MAGSAC_THRESH)
    best_cnt = int(best_mask.sum()) if best_mask is not None else 0

    # Indoor retry — consistent with Stage 1: no conf or lit pre-filters.
    # Only padded-zone rejection applied (geometry-neutral).
    if best_cnt < 20:
        p0i, p1i, ci = _loftr_pass(ref_clahe, tgt_clahe, "indoor")
        p0i, p1i, ci = dedup_matches(p0i, p1i, ci)
        if len(p0i) > 0:
            bad_i = (_in_padded_zone(p0i, ref_pl, ref_pt, ref_nw, ref_nh) |
                     _in_padded_zone(p1i, tgt_pl, tgt_pt, tgt_nw, tgt_nh))
            ok_i = ~bad_i
            if ok_i.sum() >= 4:
                p0i, p1i, ci = p0i[ok_i], p1i[ok_i], ci[ok_i]
        if len(p0i) >= 4:
            Hi, maski, namei = fit_transform(p0i, p1i, LOFTR_SIZE, LOFTR_SIZE, MAGSAC_THRESH)
            if Hi is not None and int(maski.sum()) > best_cnt:
                best_H, best_mask, model_name = Hi, maski, namei+"_indoor"
                pts0, pts1 = p0i, p1i
                total = len(p0i)
                conf_all = ci

    best_cnt = int(best_mask.sum()) if best_mask is not None else 0
    timing["fit_ms"] = round((time.perf_counter()-t0)*1000)

    EMPTY = dict(
        aligned_target=None, overlay_blend=None, vis_overview=None,
        inlier_pts0=np.zeros((0,2),np.float32),
        inlier_pts1=np.zeros((0,2),np.float32),
        palette=[], ref_disp=ref_disp, tgt_disp=tgt_disp,
        ref_clahe_disp=ref_c_disp, tgt_clahe_disp=tgt_c_disp,
        ref_rim_mask=ref_rim_disp, tgt_rim_mask=tgt_rim_disp,
        match_meta=[], inlier_count=best_cnt, inlier_ratio=0.0,
        rmse_px=0.0, rmse_m=None, n_checkpts=0,
        spatial_coverage=0.0, model_used=model_name,
        lit_warning=lit_warning, timing=timing,
        funnel=dict(total_raw=total_raw,
                    total_padzone_ok=total_padzone_ok,
                    total_geom_inliers=best_cnt),
    )
    if best_H is None or best_cnt < 4:
        return EMPTY

    inlier_ratio = best_cnt / max(1, total) * 100

    # Stage 2: Contextual region sanity — "does the transform tell a coherent story?"
    # SVD singular values give true per-axis scale (det² is signed area, not scale).
    sv = (np.linalg.svd(best_H[:2, :2], compute_uv=False)
          if best_H is not None else np.array([1.0, 1.0]))
    sane_scale  = (0.1 < sv[0] < 10.0) and (0.1 < sv[1] < 10.0)
    sane_aspect = (sv[0] / max(sv[1], 1e-9)) < 5.0  # degenerate shear/collapse
    is_hallucinated = ((best_cnt < 8) or (inlier_ratio < 1.0)
                       or (not sane_scale) or (not sane_aspect))

    if is_hallucinated:
        return EMPTY

    # Step 9: RMSE on held-out points
    rmse_px, rmse_m, n_chk = compute_rmse(pts1, pts0, best_H, best_mask, ref_px_per_m)

    # Stage 3 — Feature-Anchored Proof Scoring (post-geometry only)
    inl_pts0 = pts0[best_mask]; inl_pts1 = pts1[best_mask]
    # Use boolean mask for conf alignment (not a fixed slice)
    conf_inl = (conf_all[best_mask]
                if len(conf_all) == len(pts0)
                else np.zeros(best_cnt, np.float32))

    match_meta = []
    for i in range(best_cnt):
        l0,s0,p0v = classify_match_point(
            ref_raw_pad, ref_clahe, inl_pts0[i,0], inl_pts0[i,1],
            ref_rim_full, ref_lit_thr)
        l1,s1,p1v = classify_match_point(
            tgt_raw_pad, tgt_clahe, inl_pts1[i,0], inl_pts1[i,1],
            tgt_rim_full, tgt_lit_thr)
        match_meta.append(dict(label_ref=l0, label_tgt=l1,
                                is_shadow=s0 or s1,
                                proof_score=min(p0v,p1v),
                                loftr_conf=float(conf_inl[i])))

    # Post-geometry shadow summary (informational — no points dropped)
    n_shadow = sum(1 for m in match_meta if m["is_shadow"])
    if n_shadow > 0:
        lit_warning = (f"{n_shadow}/{best_cnt} geometry-verified inliers are "
                       f"shadow-adjacent. CLAHE proofs available in the inspector.")

    # Step 11: Display-space coords
    # LoFTR coords include pad. Subtract pad, divide by true scale, times display scale.
    def _to_disp(pts, pl, pt, sx, sy, dsc):
        p = pts.copy()
        p[:,0] = (p[:,0]-pl)/sx*dsc
        p[:,1] = (p[:,1]-pt)/sy*dsc
        return p

    pts0_d = _to_disp(inl_pts0, ref_pl, ref_pt, ref_sx, ref_sy, r_dsc)
    pts1_d = _to_disp(inl_pts1, tgt_pl, tgt_pt, tgt_sx, tgt_sy, t_dsc)
    pts0_d[:,0] = np.clip(pts0_d[:,0], 0, ref_disp.shape[1]-1)
    pts0_d[:,1] = np.clip(pts0_d[:,1], 0, ref_disp.shape[0]-1)
    pts1_d[:,0] = np.clip(pts1_d[:,0], 0, tgt_disp.shape[1]-1)
    pts1_d[:,1] = np.clip(pts1_d[:,1], 0, tgt_disp.shape[0]-1)

    # Step 12: Warp -- correct two-sided scale
    # H maps tgt_loftr -> ref_loftr.
    # H_disp = S_tgt_disp_to_loftr @ H @ S_ref_loftr_to_disp
    # S_ref: ref_disp -> ref_loftr  (multiply by sx/r_dsc, add pad)
    def _S(sx, sy, pl, pt, dsc):
        return np.array([[sx/dsc, 0,      pl],
                         [0,      sy/dsc, pt],
                         [0,      0,      1.0]], np.float64)

    S_ref     = _S(ref_sx, ref_sy, ref_pl, ref_pt, r_dsc)
    S_tgt_inv = np.linalg.inv(_S(tgt_sx, tgt_sy, tgt_pl, tgt_pt, t_dsc))
    H_disp    = S_tgt_inv @ best_H @ S_ref

    out_h, out_w = ref_disp.shape[:2]
    aligned_target = cv2.warpPerspective(
        tgt_disp, H_disp, (out_w, out_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    overlay_blend = np.clip(
        ref_disp.astype(np.float32)*0.5 + aligned_target.astype(np.float32)*0.5,
        0,255).astype(np.uint8)

    # Step 13: Overview
    MAX_VIS = min(100, best_cnt)
    # Use boolean mask to align confidences (safe even after indoor retry swap)
    inl_conf = (conf_all[best_mask]
                if len(conf_all) == len(pts0)
                else np.ones(best_cnt, np.float32))
    spread  = _spread_pts(pts0_d, MAX_VIS, conf=inl_conf)
    vis_pts0 = pts0_d[spread]; vis_pts1 = pts1_d[spread]
    vis_meta = [match_meta[i] for i in spread]
    palette  = _make_palette(len(spread))
    cov      = spatial_coverage(pts0_d, ref_disp.shape[1], ref_disp.shape[0])

    mh2 = max(ref_disp.shape[0], tgt_disp.shape[0])
    def _phd(g, h):
        bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if bgr.shape[0] < h:
            bgr = cv2.copyMakeBorder(bgr,0,h-bgr.shape[0],0,0,
                                     cv2.BORDER_CONSTANT,value=(10,10,10))
        return bgr

    overview = np.hstack((_phd(ref_disp,mh2), _phd(tgt_disp,mh2)))
    W_rd     = ref_disp.shape[1]
    for p0,p1,vm in zip(vis_pts0,vis_pts1,vis_meta):
        col  = _label_color(vm["label_ref"])
        px0  = int(round(p0[0])); py0 = int(round(p0[1]))
        px1  = int(round(p1[0]))+W_rd; py1 = int(round(p1[1]))
        cv2.line(overview,(px0,py0),(px1,py1),col,1,cv2.LINE_AA)
        cv2.circle(overview,(px0,py0),3,col,-1,cv2.LINE_AA)
        cv2.circle(overview,(px1,py1),3,col,-1,cv2.LINE_AA)

    leg = np.zeros((32, overview.shape[1], 3), np.uint8); leg[:] = (20,20,30)
    cv2.putText(leg,
        f"  {best_cnt} inliers | {model_name} | RMSE={rmse_px}px | "
        f"cov={cov*100:.0f}% | GREEN=rim CYAN=wall ORANGE=shadow-edge GREY=lit",
        (6,21), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (200,220,200), 1, cv2.LINE_AA)
    overview = np.vstack([overview, leg])

    timing["total_ms"] = round((time.perf_counter()-t0_total)*1000)

    return dict(
        aligned_target=aligned_target, overlay_blend=overlay_blend,
        vis_overview=overview,
        inlier_pts0=vis_pts0, inlier_pts1=vis_pts1, palette=palette,
        ref_disp=ref_disp, tgt_disp=tgt_disp,
        ref_clahe_disp=ref_c_disp, tgt_clahe_disp=tgt_c_disp,
        ref_rim_mask=ref_rim_disp, tgt_rim_mask=tgt_rim_disp,
        match_meta=[match_meta[i] for i in spread],
        inlier_count=best_cnt, inlier_ratio=round(inlier_ratio,1),
        rmse_px=rmse_px, rmse_m=rmse_m, n_checkpts=n_chk,
        spatial_coverage=cov, model_used=model_name,
        lit_warning=lit_warning, timing=timing,
        funnel=dict(total_raw=total_raw,
                    total_padzone_ok=total_padzone_ok,
                    total_geom_inliers=best_cnt),
    )
