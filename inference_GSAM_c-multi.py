# eval_all_plus_metrics.py
import os, glob, gc
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from scipy.linalg import sqrtm

# --------------------------
# Optional metrics deps
# --------------------------
# pip install lpips scikit-image transformers
import lpips
from skimage.metrics import structural_similarity as ssim
from transformers import CLIPProcessor, CLIPModel

# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# 원본/결과 루트 (필요시 수정)
REAL_DIR = r"E:\KSH\DGIST\DIP\real"
OUT_ROOT = r"E:\KSH\DGIST\DIP\GS_results_251101_regions_3_helmetRemoved"

# 배치 코드에서 저장한 폴더/접미사와 맞추기
TASKS = {
    "upper_to_formal": {
        "dir": os.path.join(OUT_ROOT, "out_upper_formal"),
        "suffix": "_upper_to_formal",
        # (선택) 배치 시 저장한 마스크 폴더가 있으면 넣기
        "mask_dir": None,
        # CLIP 정합용 프롬프트
        "clip_prompt": "upper body wearing a white dress shirt and a navy blazer"
    },
    "lower_to_casual": {
        "dir": os.path.join(OUT_ROOT, "out_lower_casual"),
        "suffix": "_lower_to_casual",
        "mask_dir": None,
        "clip_prompt": "lower body wearing blue denim jeans and white canvas sneakers"
    },
    "remove_helmet": {
        "dir": os.path.join(OUT_ROOT, "out_remove_helmet"),
        "suffix": "_remove_helmet",
        "mask_dir": None,
        "clip_prompt": "a construction worker without a hard hat"
    },
    "uniform_navy": {
        "dir": os.path.join(OUT_ROOT, "out_uniform_navy"),
        "suffix": "_uniform_navy",
        "mask_dir": None,
        "clip_prompt": "wearing a navy work uniform set"
    },
}

# 배치/전처리
BATCH_SIZE_IS  = 50
BATCH_SIZE_FID = 32
IMAGE_SIZE_299 = (299, 299)

# =========================
# Utils
# =========================
def list_images(folder):
    paths = glob.glob(os.path.join(folder, "*"))
    return [p for p in paths if os.path.splitext(p)[1] in ALLOWED_EXTS]

def parse_base_name(result_path, suffix):
    # 결과파일명 -> 원본 베이스명으로 역추적
    name = os.path.basename(result_path)
    stem, _ = os.path.splitext(name)
    if not stem.endswith(suffix):
        return None
    return stem[:-len(suffix)]

def find_real_by_base(real_dir, base):
    for ext in ALLOWED_EXTS:
        p = os.path.join(real_dir, base + ext)
        if os.path.exists(p):
            return p
    # 다른 확장자인 경우
    cands = glob.glob(os.path.join(real_dir, base + ".*"))
    cands = [c for c in cands if os.path.splitext(c)[1] in ALLOWED_EXTS]
    return cands[0] if cands else None

def collect_matched_pairs(result_dir, suffix, real_dir):
    gen_paths_all = list_images(result_dir)
    gen_paths, real_paths = [], []
    skipped = 0
    for g in gen_paths_all:
        base = parse_base_name(g, suffix)
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

def load_img(path):
    return Image.open(path).convert("RGB")

# (선택) 마스크 파일을 0/1 bool로 로드
def load_mask_bool(mask_path, size_hw):
    # mask는 0/255 회색 이미지라고 가정
    m = Image.open(mask_path).convert("L").resize((size_hw[1], size_hw[0]), resample=Image.NEAREST)
    m = np.array(m) > 127
    return m

# =========================
# Models & Transforms (load once)
# =========================
def get_inception_for_is():
    m = models.inception_v3(pretrained=True, transform_input=False).eval().to(DEVICE)
    return m

def get_inception_for_fid():
    m = models.inception_v3(pretrained=True, transform_input=False).eval().to(DEVICE)
    m.fc = torch.nn.Identity()  # 2048-d
    return m

transform_299 = transforms.Compose([
    transforms.Resize(IMAGE_SIZE_299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def load_batch_tensors(paths, start, end):
    imgs = []
    for p in paths[start:end]:
        img = Image.open(p).convert("RGB")
        imgs.append(transform_299(img).unsqueeze(0))
    return torch.cat(imgs, dim=0).to(DEVICE)

# =========================
# IS / FID / KID
# =========================
@torch.no_grad()
def inception_score(folder, model_is, batch_size=BATCH_SIZE_IS):
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

@torch.no_grad()
def get_inception_features(paths, model_fid, batch_size=BATCH_SIZE_FID):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = load_batch_tensors(paths, i, i + batch_size)
        f = model_fid(batch).detach().cpu().numpy()
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

def fid_between_lists(gen_paths, real_paths, model_fid):
    if len(gen_paths) == 0 or len(real_paths) == 0:
        print("[FID] 입력 부족"); return None
    gen = get_inception_features(gen_paths,  model_fid)
    real = get_inception_features(real_paths, model_fid)
    mu1, sigma1 = real.mean(axis=0), np.cov(real, rowvar=False)
    mu2, sigma2 = gen.mean(axis=0),  np.cov(gen,  rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)

def fid_for_result_dir(result_dir, suffix, real_dir, model_fid):
    gen_paths, real_paths = collect_matched_pairs(result_dir, suffix, real_dir)
    if len(gen_paths) == 0: return None, [], []
    fid = fid_between_lists(gen_paths, real_paths, model_fid)
    print(f"[FID] {os.path.basename(result_dir)} -> {fid:.4f} (pairs={len(gen_paths)})")
    return fid, gen_paths, real_paths

# KID (폴리 커널 MMD)
def polynomial_mmd_averages(codes_g, codes_r, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / codes_g.shape[1]
    K_gg = (gamma * codes_g @ codes_g.T + coef0) ** degree
    K_rr = (gamma * codes_r @ codes_r.T + coef0) ** degree
    K_gr = (gamma * codes_g @ codes_r.T + coef0) ** degree
    m = K_gg.shape[0]
    n = K_rr.shape[0]
    # unbiased
    np.fill_diagonal(K_gg, 0)
    np.fill_diagonal(K_rr, 0)
    mmd = K_gg.sum()/(m*(m-1)) + K_rr.sum()/(n*(n-1)) - 2*K_gr.mean()
    return float(mmd)

def kid_between_lists(gen_paths, real_paths, model_fid):
    if len(gen_paths)==0: return None
    gen = get_inception_features(gen_paths, model_fid)
    real = get_inception_features(real_paths, model_fid)
    return polynomial_mmd_averages(gen, real)

# =========================
# Background quality (Unmasked)
# =========================
lpips_loss = lpips.LPIPS(net='alex').to(DEVICE).eval()
_clip_model = None
_clip_proc  = None

def _build_clip():
    global _clip_model, _clip_proc
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
        _clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def clip_score(image_pil, text_prompt):
    _build_clip()
    inputs = _clip_proc(text=[text_prompt], images=image_pil, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        out = _clip_model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
        score = (img_emb @ txt_emb.t()).squeeze().item()
    return score

def compute_ssim_psnr_unmasked(gen_img, real_img, mask_bool):
    # mask_bool=True가 편집영역. 우리는 ~mask에서 SSIM/PSNR 측정.
    W, H = real_img.size
    g = np.array(gen_img.resize((W,H)))
    r = np.array(real_img.resize((W,H)))

    m = mask_bool
    if m.shape != (H, W):
        m_img = Image.fromarray((m.astype(np.uint8)*255))
        m_img = m_img.resize((W, H), resample=Image.NEAREST)
        m = (np.array(m_img) > 127)

    bg = ~m
    if bg.sum() == 0:
        return None, None

    # SSIM (RGB)
    ssim_val = ssim(r, g, channel_axis=2, data_range=255, gaussian_weights=True, use_sample_covariance=False)

    # PSNR (배경만)
    mse = np.mean((r[bg] - g[bg])**2)
    psnr_val = 10*np.log10((255.0**2)/(mse+1e-8)) if mse>0 else 99.0
    return ssim_val, psnr_val

def compute_lpips_unmasked(gen_img, real_img, mask_bool):
    W, H = real_img.size
    g = np.array(gen_img.resize((W,H))).astype(np.float32)/255.0
    r = np.array(real_img.resize((W,H))).astype(np.float32)/255.0

    m = mask_bool
    if m.shape != (H, W):
        m_img = Image.fromarray((m.astype(np.uint8)*255))
        m_img = m_img.resize((W, H), resample=Image.NEAREST)
        m = (np.array(m_img) > 127)

    # 편집영역은 동일(real)로 맞춰 배경 차이만 반영
    g_bg = g.copy()
    g_bg[m] = r[m]

    tg = torch.from_numpy(g_bg).permute(2,0,1).unsqueeze(0).to(DEVICE)*2-1
    tr = torch.from_numpy(r).permute(2,0,1).unsqueeze(0).to(DEVICE)*2-1
    with torch.no_grad():
        d = lpips_loss(tg, tr).item()
    return d

def background_quality_metrics(result_dir, suffix, real_dir, mask_dir=None):
    """
    BG-LPIPS/SSIM/PSNR 계산. mask_dir에 (base.png) 형태로 편집 마스크가 저장돼 있다고 가정.
    """
    if mask_dir is None or (mask_dir and not os.path.isdir(mask_dir)):
        print(f"[BG] mask_dir 미지정 혹은 존재하지 않음 → BG metrics 생략")
        return None

    gen_paths, real_paths = collect_matched_pairs(result_dir, suffix, real_dir)
    if len(gen_paths) == 0:
        return None

    lpips_list, ssim_list, psnr_list = [], [], []
    for g, r in tqdm(list(zip(gen_paths, real_paths)), total=len(gen_paths), desc=f"BG({os.path.basename(result_dir)})"):
        base = parse_base_name(g, suffix)
        mask_path = os.path.join(mask_dir, f"{base}.png")  # 파일 규칙에 맞게 수정
        if not os.path.exists(mask_path):
            continue

        gen_img  = load_img(g)
        real_img = load_img(r)
        mask_bool = load_mask_bool(mask_path, size_hw=(real_img.size[1], real_img.size[0]))

        ssim_val, psnr_val = compute_ssim_psnr_unmasked(gen_img, real_img, mask_bool)
        if ssim_val is None:
            continue
        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)
        lpips_val = compute_lpips_unmasked(gen_img, real_img, mask_bool)
        lpips_list.append(lpips_val)

    if len(lpips_list)==0:
        return None
    return {
        "LPIPS_bg": float(np.mean(lpips_list)),
        "SSIM_bg":  float(np.mean(ssim_list)),
        "PSNR_bg":  float(np.mean(psnr_list))
    }

def clip_alignment_for_task(result_dir, prompt_text):
    paths = list_images(result_dir)
    if len(paths)==0: return None
    scores = []
    for p in tqdm(paths, desc=f"CLIP({os.path.basename(result_dir)})"):
        img = load_img(p)
        scores.append(clip_score(img, prompt_text))
    return float(np.mean(scores))

# =========================
# Run all tasks
# =========================
if __name__ == "__main__":
    print(f"[DEVICE] {DEVICE}")
    model_is  = get_inception_for_is()
    model_fid = get_inception_for_fid()

    results = {}
    for name, spec in TASKS.items():
        out_dir = spec["dir"]; suffix = spec["suffix"]
        mask_dir = spec.get("mask_dir", None)
        clip_prompt = spec.get("clip_prompt", None)

        print(f"\n=== {name} ===")
        # IS
        is_score = inception_score(out_dir, model_is)

        # FID & 쌍 목록 (KID도 같은 쌍 재사용)
        fid, gen_paths, real_paths = fid_for_result_dir(out_dir, suffix, REAL_DIR, model_fid)

        # KID
        kid = kid_between_lists(gen_paths, real_paths, model_fid) if len(gen_paths)>0 else None
        if kid is not None:
            print(f"[KID] {os.path.basename(out_dir)} -> {kid:.6f} (pairs={len(gen_paths)})")

        # 배경 품질 (마스크가 있을 때만)
        bg = background_quality_metrics(out_dir, suffix, REAL_DIR, mask_dir=mask_dir)

        # CLIP 정합
        clip_avg = None
        if clip_prompt:
            clip_avg = clip_alignment_for_task(out_dir, clip_prompt)
            print(f"[CLIP] {os.path.basename(out_dir)} -> {clip_avg:.4f}")

        results[name] = {
            "IS": is_score,
            "FID": fid,
            "KID": kid,
            "BG": bg,
            "CLIP": clip_avg
        }

    print("\n===== Summary =====")
    for name, r in results.items():
        is_s  = None if r['IS']  is None else round(r['IS'],4)
        fid_s = None if r['FID'] is None else round(r['FID'],4)
        kid_s = None if r['KID'] is None else float(f"{r['KID']:.6f}") if r['KID'] is not None else None
        clip_s = None if r['CLIP'] is None else round(r['CLIP'],4)
        if r['BG'] is not None:
            bg_lp = round(r['BG']['LPIPS_bg'],4)
            bg_ss = round(r['BG']['SSIM_bg'],4)
            bg_ps = round(r['BG']['PSNR_bg'],2)
            bg_str = f"BG: LPIPS {bg_lp} | SSIM {bg_ss} | PSNR {bg_ps} dB"
        else:
            bg_str = "BG: -"
        print(f"{name:>16} | IS {is_s} | FID {fid_s} | KID {kid_s} | CLIP {clip_s} | {bg_str}")
