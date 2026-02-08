# sd_baseline_batch.py
# 베이스라인(Only SD): 추가 모델 없이 Stable Diffusion만 사용

# 부위 제한이 필요한 작업(상반신/하반신/헬멧)은 단순 기하 마스크(상·하 절반, 상단 띠 등)를 써서 Inpainting.

# 전신 톤 변경은 전체 프레임 Img2Img.

# 즉, “세그먼트 고도화(GD+SAM)” 없이도 돌아가는 순수 SD 파이프라인.
import os, glob, numpy as np, torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLOWED = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# --------- Masks: 단순 기하학적 마스크(세그/디텍션 없이) ---------
def mask_upper(img_pil, margin_ratio=0.05):
    W, H = img_pil.size
    m = np.zeros((H, W), np.uint8)
    y1, y2 = int(margin_ratio*H), int(0.5*H)
    m[y1:y2, :] = 255
    return Image.fromarray(m)

def mask_lower(img_pil, margin_ratio=0.05):
    W, H = img_pil.size
    m = np.zeros((H, W), np.uint8)
    y1, y2 = int(0.5*H), int((1.0 - margin_ratio)*H)
    m[y1:y2, :] = 255
    return Image.fromarray(m)

def mask_top_band(img_pil, band_ratio=0.22):
    # 헬멧 제거를 위한 상단 띠(머리 영역을 대략 커버)
    W, H = img_pil.size
    m = np.zeros((H, W), np.uint8)
    y2 = int(band_ratio*H)
    m[:y2, :] = 255
    return Image.fromarray(m)

# --------- Pipelines (Stable Diffusion Only) ---------
def build_inpaint(repo="stabilityai/stable-diffusion-2-inpainting", dtype=torch.float16):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        repo, torch_dtype=dtype, safety_checker=None, feature_extractor=None
    ).to(DEVICE)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()
    return pipe

def build_img2img(repo="runwayml/stable-diffusion-v1-5", dtype=torch.float16):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        repo, torch_dtype=dtype, safety_checker=None, feature_extractor=None
    ).to(DEVICE)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()
    return pipe

# --------- 베이스라인 프롬프트(우리 GS 실험과 의미 매칭) ---------
PROMPTS = {
    "upper_to_formal": {
        "pos": "a person wearing a formal white dress shirt and a dark blazer, natural fabric wrinkles, photorealistic",
        "neg": "t-shirt, jersey, safety vest, high-visibility, logo patches, artifacts",
        "mask_fn": mask_upper,
        "mode": "inpaint",
        "suffix": "_upper_to_formal",
        # "strength": 0.90
        "strength": 0.55
    },
    "lower_to_casual": {
        "pos": "a person wearing casual chinos and canvas sneakers, natural folds and shading, photorealistic",
        "neg": "work boots, steel-toe, suit pants, artifacts",
        "mask_fn": mask_lower,
        "mode": "inpaint",
        "suffix": "_lower_to_casual",
        # "strength": 0.90
        "strength": 0.55
    },
    "remove_helmet": {
        "pos": "a person without a helmet, natural hair continuity and lighting, photorealistic",
        "neg": "helmet, hard hat, strap, artifacts",
        "mask_fn": mask_top_band,
        "mode": "inpaint",
        "suffix": "_remove_helmet",
        # "strength": 0.90
        "strength": 0.50
    },
    "uniform_navy": {
        "pos": "navy workwear color tone, consistent with existing clothing structure, photorealistic",
        "neg": "fluorescent colors, reflective tape, color bleeding, artifacts",
        "mask_fn": None,
        "mode": "img2img",  # 전신 톤 변경은 전체 프레임 변환
        "suffix": "_uniform_navy",
        # "strength": 0.90
        "strength": 0.50
    },
}

# --------- 실행 ---------
def list_images(folder):
    files = []
    for ext in ALLOWED:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True); return d

def run_baseline(input_dir, out_root,
                 size=512, steps=30, guidance=7.5, seed=42):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    # 파이프라인 생성(1회)
    pipe_inpaint = build_inpaint()
    pipe_img2img = build_img2img()

    # 출력 폴더
    out_dirs = {k: ensure_dir(os.path.join(out_root, f"baseline_{k}"))
                for k in PROMPTS.keys()}

    imgs = list_images(input_dir)
    print(f"[INFO] found {len(imgs)} images.")

    for ipath in imgs:
        fname = os.path.splitext(os.path.basename(ipath))[0]
        img = Image.open(ipath).convert("RGB").resize((size, size), Image.LANCZOS)

        for name, spec in PROMPTS.items():
            try:
                if spec["mode"] == "inpaint":
                    mask = spec["mask_fn"](img)
                    out = pipe_inpaint(
                        prompt=spec["pos"],
                        negative_prompt=spec["neg"],
                        image=img,
                        mask_image=mask,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        strength=spec["strength"],
                        generator=gen
                    ).images[0]
                else:  # img2img
                    out = pipe_img2img(
                        prompt=spec["pos"],
                        negative_prompt=spec["neg"],
                        image=img,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        strength=spec["strength"],
                        generator=gen
                    ).images[0]

                out.save(os.path.join(out_dirs[name], f"{fname}{PROMPTS[name]['suffix']}.png"))
                print(f"[OK] {name}: {fname}")

            except Exception as e:
                print(f"[ERR] {name}: {fname} :: {e}")

    print("\n[Done] Baseline outputs:")
    for k, d in out_dirs.items():
        print(f" - {k}: {d}")

if __name__ == "__main__":
    INPUT_DIR = r"E:\KSH\DGIST\DIP\real"  # 또는 ImageNet 폴더
    OUT_ROOT  = r"E:\KSH\DGIST\DIP\GS_results_baseline_SDonly_regions_4_250917"
    run_baseline(INPUT_DIR, OUT_ROOT)
