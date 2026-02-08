# eval_sdonly_full.py
# ------------------------------------------------------------
# SD-only 결과 폴더를 평가: IS, FID, KID, CLIP, (옵션)BG PSNR/SSIM/LPIPS
# ------------------------------------------------------------
import os, glob, gc, math, warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from scipy.linalg import sqrtm

# ====== (옵션) LPIPS / SSIM ======
try:
    import lpips
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False

try:
    from skimage.metrics import structural_similarity as ssim_sk
    HAS_SSIM = True
except Exception:
    HAS_SSIM = False

# ====== (옵션) CLIP ======
_HAS_OPEN_CLIP = False
_HAS_ORIG_CLIP = False
try:
    import open_clip
    _HAS_OPEN_CLIP = True
except Exception:
    pass
if not _HAS_OPEN_CLIP:
    try:
        import clip as orig_clip
        _HAS_ORIG_CLIP = True
    except Exception:
        pass

# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# 원본/결과 루트
REAL_DIR = r"E:\KSH\DGIST\DIP\real"
OUT_ROOT = r"E:\KSH\DGIST\DIP\experiments\GS_results_baseline_SDonly_regions_3"

# 결과 폴더 & 접미사 후보(유연 매칭)
TASKS = {
    "upper_to_formal": {
        "dir": os.path.join(OUT_ROOT, "baseline_upper_to_formal"),
        # 결과 파일 접미사 후보 (길이가 긴 것부터 우선 매칭)
        "suffix_candidates": ["_upper_to_formal_SD", "_upper_to_formal_DE", "_upper_to_formal", ""],
        # (옵션) 마스크 폴더: 편집영역을 뺀 배경 품질평가 (없으면 생략)
        "mask_dir": None,   # 예: r"E:\...\masks\upper_to_formal"
    },
    "lower_to_casual": {
        "dir": os.path.join(OUT_ROOT, "baseline_lower_to_casual"),
        "suffix_candidates": ["_lower_to_casual_SD", "_lower_to_casual_DE", "_lower_to_casual", ""],
        "mask_dir": None,
    },
    "remove_helmet": {
        "dir": os.path.join(OUT_ROOT, "baseline_remove_helmet"),
        "suffix_candidates": ["_remove_helmet_SD", "_remove_helmet_DE", "_remove_helmet", ""],
        "mask_dir": None,
    },
    "uniform_navy": {
        "dir": os.path.join(OUT_ROOT, "baseline_uniform_navy"),
        "suffix_candidates": ["_uniform_navy_SD", "_uniform_navy_DE", "_uniform_navy", ""],
        "mask_dir": None,
    },
}

# 배치/전처리
BATCH_SIZE_IS  = 50
BATCH_SIZE_FID = 32
IMAGE_SIZE_299 = (299, 299)  # Inception v3 권장 크기

# =========================
# Utils: 파일/매칭
# =========================
def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    paths = glob.glob(os.path.join(folder, "*"))
    return [p for p in paths if os.path.splitext(p)[1] in ALLOWED_EXTS]

def parse_base_name_flexible(result_path: str, suffix_candidates: List[str]) -> Optional[str]:
    """결과 파일에서 여러 접미사 후보 중 일치하는 걸 떼고 base 반환."""
    stem, _ = os.path.splitext(os.path.basename(result_path))
    for suf in sorted(suffix_candidates, key=len, reverse=True):
        if suf and stem.endswith(suf):
            return stem[: -len(suf)]
    # 빈 접미사 후보("")가 있으면 그냥 stem
    return stem if ("" in suffix_candidates) else None

def find_real_by_base(real_dir: str, base: str) -> Optional[str]:
    for ext in ALLOWED_EXTS:
        p = os.path.join(real_dir, base + ext)
        if os.path.exists(p):
            return p
    # 확장자 다른 경우 패턴
    cands = glob.glob(os.path.join(real_dir, base + ".*"))
    cands = [c for c in cands if os.path.splitext(c)[1] in ALLOWED_EXTS]
    return cands[0] if cands else None

def collect_matched_pairs_flexible(result_dir: str, suffix_candidates: List[str], real_dir: str) -> Tuple[List[str], List[str]]:
    gen_paths_all = list_images(result_dir)
    gen_paths, real_paths = [], []
    skipped = 0
    for g in gen_paths_all:
        base = parse_base_name_flexible(g, suffix_candidates)
        if base is None:
            skipped += 1; continue
        r = find_real_by_base(real_dir, base)
        if r is None:
            skipped += 1; continue
        gen_paths.append(g); real_paths.append(r)
    if skipped > 0:
        print(f"[WARN] {os.path.basename(result_dir)}: 매칭 실패 {skipped}개 제외")
    print(f"[INFO] {os.path.basename(result_dir)}: 매칭 쌍 {len(gen_paths)}")
    return gen_paths, real_paths

# =========================
# Inception models (load once)
# =========================
def get_inception_for_is():
    m = models.inception_v3(pretrained=True, transform_input=False).eval().to(DEVICE)
    return m

def get_inception_for_fid_kid():
    m = models.inception_v3(pretrained=True, transform_input=False).eval().to(DEVICE)
    m.fc = torch.nn.Identity()  # 2048-d
    return m

transform_299 = transforms.Compose([
    transforms.Resize(IMAGE_SIZE_299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def load_batch_tensors(paths: List[str], start: int, end: int) -> torch.Tensor:
    imgs = []
    for p in paths[start:end]:
        img = Image.open(p).convert("RGB")
        imgs.append(transform_299(img).unsqueeze(0))
    return torch.cat(imgs, dim=0).to(DEVICE)

# =========================
# IS
# =========================
@torch.no_grad()
def inception_score(folder: str, model_is, batch_size=BATCH_SIZE_IS):
    paths = list_images(folder)
    if len(paths) == 0:
        print(f"[IS] '{folder}' 비어있음"); return None
    preds_list = []
    for i in range(0, len(paths), batch_size):
        batch = load_batch_tensors(paths, i, i + batch_size)
        logits = model_is(batch)
        preds = F.softmax(logits, dim=1).detach().cpu().numpy()
        preds_list.append(preds)
        del batch, logits
        if DEVICE == "cuda": torch.cuda.empty_cache()
        gc.collect()
    preds_all = np.concatenate(preds_list, axis=0)
    py = np.mean(preds_all, axis=0)
    kl = preds_all * (np.log(preds_all + 1e-10) - np.log(py + 1e-10))
    is_score = float(np.exp(np.mean(np.sum(kl, axis=1))))
    print(f"[IS] {os.path.basename(folder)} -> {is_score:.4f} (N={len(paths)})")
    return is_score

# =========================
# FID
# =========================
@torch.no_grad()
def get_inception_features(paths: List[str], model_fid, batch_size=BATCH_SIZE_FID):
    if len(paths) == 0:
        return np.empty((0, 2048), dtype=np.float32)
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = load_batch_tensors(paths, i, i + batch_size)
        f = model_fid(batch).detach().cpu().numpy().astype(np.float32)
        feats.append(f)
        del batch
        if DEVICE == "cuda": torch.cuda.empty_cache()
        gc.collect()
    return np.concatenate(feats, axis=0)

def frechet_distance(mu1, sigma1, mu2, sigma2):
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))

def fid_between_lists(gen_paths: List[str], real_paths: List[str], model_fid):
    if len(gen_paths) < 2 or len(real_paths) < 2:
        print("[FID] 입력 부족"); return None
    gen = get_inception_features(gen_paths,  model_fid)
    real = get_inception_features(real_paths, model_fid)
    mu1, sigma1 = real.mean(axis=0), np.cov(real, rowvar=False)
    mu2, sigma2 = gen.mean(axis=0),  np.cov(gen,  rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)

def fid_for_result_dir(result_dir: str, suffix_candidates: List[str], real_dir: str, model_fid):
    gen_paths, real_paths = collect_matched_pairs_flexible(result_dir, suffix_candidates, real_dir)
    if len(gen_paths) == 0: 
        print(f"[FID] {os.path.basename(result_dir)} -> 계산 불가 (매칭 0)")
        return None
    fid = fid_between_lists(gen_paths, real_paths, model_fid)
    if fid is not None:
        print(f"[FID] {os.path.basename(result_dir)} -> {fid:.4f}  (pairs={len(gen_paths)})")
    return fid

# =========================
# KID (unbiased polynomial MMD, degree 3)
# =========================
def _poly_mmd2_unbiased(x: np.ndarray, y: np.ndarray, degree=3, gamma=None, coef=1.0):
    # x,y: (N, D)
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    def k(a, b):
        return (gamma * a.dot(b.T) + coef) ** degree

    nx, ny = x.shape[0], y.shape[0]
    k_xx = k(x, x)
    k_yy = k(y, y)
    k_xy = k(x, y)

    # unbiased estimator
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    mmd = (k_xx.sum() / (nx * (nx - 1))
           + k_yy.sum() / (ny * (ny - 1))
           - 2 * k_xy.mean())
    return float(mmd)

def kid_between_lists(gen_paths: List[str], real_paths: List[str], model_fid) -> Optional[float]:
    if len(gen_paths) < 2 or len(real_paths) < 2:
        print("[KID] 입력 부족"); return None
    gen = get_inception_features(gen_paths, model_fid)
    real = get_inception_features(real_paths, model_fid)
    return _poly_mmd2_unbiased(gen, real, degree=3, gamma=1.0/gen.shape[1], coef=1.0)

def kid_for_result_dir(result_dir: str, suffix_candidates: List[str], real_dir: str, model_fid):
    gen_paths, real_paths = collect_matched_pairs_flexible(result_dir, suffix_candidates, real_dir)
    if len(gen_paths) == 0:
        print(f"[KID] {os.path.basename(result_dir)} -> 계산 불가 (매칭 0)")
        return None
    kid = kid_between_lists(gen_paths, real_paths, model_fid)
    if kid is not None:
        print(f"[KID] {os.path.basename(result_dir)} -> {kid:.6f}  (pairs={len(gen_paths)})")
    return kid

# =========================
# CLIP 이미지-이미지 유사도
# =========================
@torch.no_grad()
def clip_image_embeds(img_paths: List[str], preprocess, model) -> np.ndarray:
    if len(img_paths) == 0:
        return np.empty((0, 512), dtype=np.float32)
    feats = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        t = preprocess(img).unsqueeze(0).to(DEVICE)
        feat = model.encode_image(t)
        feat = F.normalize(feat, dim=-1)
        feats.append(feat.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)

def build_clip():
    if _HAS_OPEN_CLIP:
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=DEVICE)
        return model.eval(), preprocess
    if _HAS_ORIG_CLIP:
        model, preprocess = orig_clip.load("ViT-B/32", device=DEVICE, jit=False)
        return model.eval(), preprocess
    print("[CLIP] open_clip/clip 미설치 → CLIP 생략")
    return None, None

def clip_similarity_avg(gen_paths: List[str], real_paths: List[str]) -> Optional[float]:
    model, preprocess = build_clip()
    if model is None: 
        return None
    if len(gen_paths) == 0:
        return None
    g = clip_image_embeds(gen_paths, preprocess, model)
    r = clip_image_embeds(real_paths, preprocess, model)
    n = min(g.shape[0], r.shape[0])
    if n == 0:
        return None
    sim = (g[:n] * r[:n]).sum(axis=1)   # cosine (이미 정규화)
    return float(sim.mean())

# =========================
# (옵션) 배경 품질: PSNR / SSIM / LPIPS
# =========================
def _to_np(img_path: str, size: Tuple[int,int]=None) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)

def _psnr_np(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()
    if mse <= 1e-12: 
        return 100.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))

def _ssim_np(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if not HAS_SSIM:
        return None
    return float(ssim_sk(a, b, channel_axis=2, data_range=255))

def _lpips_torch(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if not HAS_LPIPS:
        return None
    # lpips는 [-1,1] 텐서 (NCHW)
    t_a = torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 127.5 - 1.0
    t_b = torch.from_numpy(b).permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 127.5 - 1.0
    loss_fn = lpips.LPIPS(net='alex').to(DEVICE).eval()
    with torch.no_grad():
        val = loss_fn(t_a, t_b).squeeze().item()
    return float(val)

def _apply_bg_mask(img_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """mask==1: 편집영역 → 배경만 남기기 위해 편집영역을 0으로."""
    m = (mask_np > 127).astype(np.uint8)
    bg = img_np.copy()
    bg[m.astype(bool)] = 0
    return bg

def bg_quality_metrics(gen_paths: List[str], real_paths: List[str], mask_dir: str,
                       vis_size=(256,256)) -> Dict[str, Optional[float]]:
    if (mask_dir is None) or (not os.path.isdir(mask_dir)):
        print("[BG] mask_dir 미지정 혹은 존재하지 않음 → BG metrics 생략")
        return {"PSNR": None, "SSIM": None, "LPIPS": None}

    psnr_list, ssim_list, lpips_list = [], [], []
    for g, r in zip(gen_paths, real_paths):
        base = os.path.splitext(os.path.basename(r))[0]
        # 가능한 마스크 파일 찾아보기 (여러 확장자 허용)
        mpath = None
        for ext in ALLOWED_EXTS:
            cand = os.path.join(mask_dir, base + ext)
            if os.path.exists(cand):
                mpath = cand; break
        if mpath is None:
            continue

        gen = _to_np(g, vis_size)
        real = _to_np(r, vis_size)
        mask = _to_np(mpath, vis_size)
        if mask.ndim == 3:
            mask = mask[:,:,0]

        gen_bg  = _apply_bg_mask(gen,  mask)
        real_bg = _apply_bg_mask(real, mask)

        psnr_list.append(_psnr_np(gen_bg, real_bg))
        ssim_val = _ssim_np(gen_bg, real_bg)
        if ssim_val is not None:
            ssim_list.append(ssim_val)
        lpips_val = _lpips_torch(gen_bg, real_bg)
        if lpips_val is not None:
            lpips_list.append(lpips_val)

    out = {
        "PSNR": float(np.mean(psnr_list)) if len(psnr_list) else None,
        "SSIM": float(np.mean(ssim_list)) if len(ssim_list) else None,
        "LPIPS": float(np.mean(lpips_list)) if len(lpips_list) else None,
    }
    return out

# =========================
# Run all tasks
# =========================
if __name__ == "__main__":
    print(f"[DEVICE] {DEVICE}")
    model_is   = get_inception_for_is()
    model_fidk = get_inception_for_fid_kid()

    results = {}
    for name, spec in TASKS.items():
        out_dir = spec["dir"]
        suf_cands = spec["suffix_candidates"]
        print(f"\n=== {name} ===")
        # IS: 폴더 자체 평가
        is_score = inception_score(out_dir, model_is)

        # FID/KID: 결과-원본 매칭 기반
        gen_paths, real_paths = collect_matched_pairs_flexible(out_dir, suf_cands, REAL_DIR)
        fid = fid_between_lists(gen_paths, real_paths, model_fidk) if len(gen_paths) else None
        if fid is not None:
            print(f"[FID] {os.path.basename(out_dir)} -> {fid:.4f} (pairs={len(gen_paths)})")

        kid = kid_between_lists(gen_paths, real_paths, model_fidk) if len(gen_paths) else None
        if kid is not None:
            print(f"[KID] {os.path.basename(out_dir)} -> {kid:.6f} (pairs={len(gen_paths)})")

        # CLIP: 결과-원본 유사도
        clip_avg = clip_similarity_avg(gen_paths, real_paths) if len(gen_paths) else None
        if clip_avg is not None:
            print(f"[CLIP] {os.path.basename(out_dir)} -> {clip_avg:.4f}")

        # (옵션) 배경 품질
        bg = bg_quality_metrics(gen_paths, real_paths, spec.get("mask_dir", None))

        results[name] = {
            "IS": is_score, "FID": fid, "KID": kid, "CLIP": clip_avg,
            "BG_PSNR": bg.get("PSNR"), "BG_SSIM": bg.get("SSIM"), "BG_LPIPS": bg.get("LPIPS")
        }

    print("\n===== Summary =====")
    header = f"{'task':>18} | {'IS':>8} | {'FID':>10} | {'KID':>12} | {'CLIP':>8} | {'BG-PSNR':>8} | {'BG-SSIM':>8} | {'BG-LPIPS':>8}"
    print(header)
    print("-"*len(header))
    for name, r in results.items():
        def fmt(x, nd=4):
            if x is None: return "  None  "
            return f"{x:.{nd}f}".rjust(8)
        line = f"{name:>18} | {fmt(r['IS'])} | {fmt(r['FID'])} | {fmt(r['KID'],6)} | {fmt(r['CLIP'])} | {fmt(r['BG_PSNR'])} | {fmt(r['BG_SSIM'])} | {fmt(r['BG_LPIPS'])}"
        print(line)
