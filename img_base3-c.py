# gs_batch_inpaint_regions.py
import os, glob, torch, numpy as np, cv2
from pathlib import Path
from PIL import Image
from datetime import datetime

from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler
from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- Utils --------------------
def log_append(log_path, msg):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}] {msg}\n")

def boxes_cxcywh_norm_to_xyxy_pix(box, W, H):
    # GroundingDINO: cx, cy, w, h in [0,1]
    cx, cy, bw, bh = box
    x1 = (cx - bw / 2) * W
    y1 = (cy - bh / 2) * H
    x2 = (cx + bw / 2) * W
    y2 = (cy + bh / 2) * H
    return np.array([x1, y1, x2, y2])

def debug_overlay_save(img_rgb, boxes_xyxy, mask_uint8, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = img_rgb.copy()
    for (x1,y1,x2,y2) in boxes_xyxy.astype(int):
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    overlay = img.copy()
    overlay[mask_uint8.astype(bool)] = (255, 0, 0)
    vis = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    Image.fromarray(vis).save(save_path)

# -------------------- Model Bundle --------------------
class ModelBundle:
    def __init__(
        self,
        groundingdino_cfg,
        groundingdino_ckpt,
        sam_ckpt,
        sd_inpaint_repo="stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
        vae_tiling=True,
        attention_slicing=True
    ):
        # GroundingDINO (once)
        self.gd_model = load_model(groundingdino_cfg, groundingdino_ckpt)

        # SAM (once) - predictor.set_image()만 매 이미지마다 호출
        sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
        self.sam_predictor = SamPredictor(sam.to(DEVICE))

        # Stable Diffusion Inpaint (once)
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            sd_inpaint_repo,
            torch_dtype=torch_dtype,
            safety_checker=None,
            feature_extractor=None
        ).to(DEVICE)

        # 스케줄러 안정화
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        if vae_tiling:
            self.pipe.enable_vae_tiling()
        if attention_slicing:
            self.pipe.enable_attention_slicing()

# -------------------- Mask (텍스트 질의) --------------------
def grounded_sam_mask(
    img_path,
    bundle: ModelBundle,
    text_queries=("shoes",),
    box_threshold=0.25,
    text_threshold=0.25,
    dilate_ksize=7,
    pick="union",   # "best" or "union"
    debug_save=None
):
    """
    텍스트 질의로 박스 -> SAM 정밀마스크
    Returns: PIL mask (uint8 0/255) or None
    """
    image_source, image = load_image(img_path)  # RGB np.uint8
    H, W = image_source.shape[:2]

    # 1) GroundingDINO: find boxes
    boxes_all = []
    for query in text_queries:
        boxes, logits, phrases = predict(
            model=bundle.gd_model,
            image=image,
            caption=query,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        if boxes is None: continue
        boxes_all.extend([b for b in boxes])

    if len(boxes_all) == 0:
        return None

    # 2) SAM: refine to masks  (반드시 원본 이미지를 넣어야 함)
    bundle.sam_predictor.set_image(image_source)
    mask_total = np.zeros((H, W), dtype=np.uint8)
    boxes_xyxy = []

    for b in boxes_all:
        box_xyxy = boxes_cxcywh_norm_to_xyxy_pix(b, W, H)
        boxes_xyxy.append(box_xyxy)
        masks, scores, _ = bundle.sam_predictor.predict(
            box=box_xyxy, point_coords=None, point_labels=None, multimask_output=True
        )
        if pick == "best":
            m = masks[np.argmax(scores)].astype(np.uint8)
        else:  # "union"
            m = (masks.sum(axis=0) > 0).astype(np.uint8)
        mask_total = np.maximum(mask_total, m)

    # 경계 보정(팽창)
    if dilate_ksize and dilate_ksize > 1:
        kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
        mask_total = cv2.dilate(mask_total, kernel, iterations=1)

    if debug_save is not None:
        debug_overlay_save(image_source, np.array(boxes_xyxy), mask_total, debug_save)

    return Image.fromarray((mask_total * 255).astype(np.uint8))

# -------------------- Mask (상/하/전신 영역 프리셋) --------------------
def person_region_mask(
    img_path,
    bundle: ModelBundle,
    region="upper",      # "upper" | "lower" | "full"
    person_query=("person",),
    box_threshold=0.25,
    text_threshold=0.25,
    dilate_ksize=7,
    pick="union",
    debug_save=None
):
    """
    사람 박스를 검출 -> 상/하/전신 서브박스로 SAM 정밀마스크
    Returns: PIL mask (uint8 0/255) or None
    """
    image_source, image = load_image(img_path)   # RGB np.uint8
    H, W = image_source.shape[:2]

    # 1) 사람 박스 검출 (가장 큰 박스 선택)
    person_boxes = []
    for q in person_query:
        boxes, logits, phrases = predict(
            model=bundle.gd_model, image=image, caption=q,
            box_threshold=box_threshold, text_threshold=text_threshold
        )
        if boxes is None: continue
        person_boxes.extend([b for b in boxes])
    if len(person_boxes) == 0:
        return None

    areas, xyxy_list = [], []
    for b in person_boxes:
        x1,y1,x2,y2 = boxes_cxcywh_norm_to_xyxy_pix(b, W, H)
        xyxy_list.append((x1,y1,x2,y2))
        areas.append(max(0,(x2-x1))*max(0,(y2-y1)))
    main_idx = int(np.argmax(areas))
    x1,y1,x2,y2 = xyxy_list[main_idx]

    # 2) 영역 분할 (상/하/전신) — 여유 패딩 (필요 시 비율 조정, 0.55, 0.45값 조정하면 됨)
    pad = 0.04
    ww, hh = (x2-x1), (y2-y1)
    if region == "upper":
        ry1, ry2 = y1, y1 + hh*0.55
        rx1, rx2 = x1, x2
    elif region == "lower":
        ry1, ry2 = y1 + hh*0.45, y2
        rx1, rx2 = x1, x2
    else:  # full
        rx1, ry1, rx2, ry2 = x1, y1, x2, y2

    rx1 = max(0, rx1 - ww*pad); ry1 = max(0, ry1 - hh*pad)
    rx2 = min(W, rx2 + ww*pad); ry2 = min(H, ry2 + hh*pad)
    region_box = np.array([rx1, ry1, rx2, ry2])

    # 3) SAM로 정밀 마스크
    bundle.sam_predictor.set_image(image_source)
    masks, scores, _ = bundle.sam_predictor.predict(
        box=region_box, point_coords=None, point_labels=None, multimask_output=True
    )
    if pick == "best":
        m = masks[np.argmax(scores)].astype(np.uint8)
    else:
        m = (masks.sum(axis=0) > 0).astype(np.uint8)

    # 4) 경계 보정
    if dilate_ksize and dilate_ksize > 1:
        kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
        m = cv2.dilate(m, kernel, 1)

    if debug_save is not None:
        debug_overlay_save(image_source, np.array([region_box]), m, debug_save)

    return Image.fromarray((m*255).astype(np.uint8))

# -------------------- Inpaint --------------------
def inpaint_once(pipe: StableDiffusionInpaintPipeline, img_path, mask_pil,
                 positive_prompt, negative_prompt="",
                 size=512, steps=30, guidance=7.5, strength=0.50, seed=42):
    image = Image.open(img_path).convert("RGB").resize((size, size))
    mask  = mask_pil.resize((size, size), resample=Image.NEAREST)

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    result = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        generator=generator
    )
    return result.images[0]

# -------------------- Prompts (Task Map) --------------------
TASKS = {
    # 1) 상반신을 정장 상의로 (셔츠+블레이저)
    "upper_to_formal": {
        "use_region": True,
        "region": "upper",
        "pos": "upper body wearing a white dress shirt and a navy blazer, keep original pose and lighting, fabric continuity, photorealistic, detailed textures",
        "neg": "safety vest, reflective tape, logos, blur, artifacts, extra tie, double collar",
        # "strength": 0.55
        # "strength": 0.85
        "strength": 0.90
    },

    # 2) 하반신을 캐주얼(청바지 + 캔버스화)
    "lower_to_casual": {
        "use_region": True,
        "region": "lower",
        "pos": "lower body wearing blue denim jeans and white canvas sneakers, correct perspective, fabric creases continuity, photorealistic",
        "neg": "work boots, steel-toe, leather dress shoes, safety pants, reflective stripes, blur, artifacts",
        # "strength": 0.55
        # "strength": 0.85
        "strength": 0.90
    },

    # 3) 헬멧 제거(텍스트 질의형)
    "remove_helmet": {
        "use_region": False,
        "queries": ("hard hat","safety helmet"),
        "pos": "construction worker without a hard hat, natural hair continuity and lighting, face intact, hair intact, photorealistic",
        # "neg": "hard hat, safety helmet, helmet strap, plastic shell, artifacts",
        "neg": (
            "hard hat, safety helmet, helmet strap, plastic shell, "
            "bald, shaved head, missing head, cut-off head, cropped head, "
            "deformed face, distorted face, melted face, asymmetrical face, bad face, "
            "extra head, extra face, fused face, "
            "hair loss, missing hair, blurred hair, "
            "facial blur, over-smoothing, waxy skin, artifacts, glitches, deformed anatomy, "
            "color bleeding, overspray, over-inpainting"
        ),
        # "strength": 0.50
        # "strength": 0.80
        "strength": 0.90
    },

    # 4) 전신을 네이비 유니폼 톤으로 통일
    "uniform_navy": {
        "use_region": True,
        "region": "full",
        "pos": "wearing a navy work uniform set (jacket and pants) with consistent fabric and shading, keep original pose and lighting, photorealistic",
        "neg": "logos, reflective tape, bright hi-vis colors, color bleeding, blur, artifacts",
        # "strength": 0.50
        # "strength": 0.80
        "strength": 0.90
    },
}

# -------------------- Main Batch --------------------
if __name__ == "__main__":
    # 경로 설정
    input_dir  = r"E:\KSH\DGIST\DIP\real"
    out_root   = r"E:\KSH\DGIST\DIP\GS_results_251203_tests_regions_3_helmetRemoved"
    log_path   = os.path.join(out_root, "skipped.txt")

    # 출력 폴더 맵
    out_dir_map = {
        "upper_to_formal": os.path.join(out_root, "out_upper_formal"),
        "lower_to_casual": os.path.join(out_root, "out_lower_casual"),
        "remove_helmet":   os.path.join(out_root, "out_remove_helmet"),
        "uniform_navy":    os.path.join(out_root, "out_uniform_navy"),
    }
    for d in out_dir_map.values():
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # 모델 체크포인트/설정
    groundingdino_cfg  = r"E:\KSH\DGIST\DIP\weights\GroundingDINO_SwinT_OGC.py"
    groundingdino_ckpt = r"E:\KSH\DGIST\DIP\weights\groundingdino_swint_ogc.pth"
    sam_ckpt           = r"E:\KSH\DGIST\DIP\weights\sam_vit_h_4b8939.pth"

    # 번들 1회 초기화
    bundle = ModelBundle(
        groundingdino_cfg=groundingdino_cfg,
        groundingdino_ckpt=groundingdino_ckpt,
        sam_ckpt=sam_ckpt,
        sd_inpaint_repo="stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16
    )

    # 배치 대상 수집
    exts = ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG")
    img_files = []
    for e in exts:
        img_files.extend(glob.glob(os.path.join(input_dir, e)))
    img_files = sorted(img_files)

    # 공통 하이퍼 (필요 시 조정)
    BOX_TH = 0.20
    TEXT_TH = 0.20
    SIZE = 512
    STEPS = 30
    GUIDANCE = 7.5
    DILATE_K = 7
    PICK = "union"   # "best" or "union"
    SEED = 42

    # (옵션) 디버그 오버레이 저장
    DEBUG = False
    debug_dir = os.path.join(out_root, "_debug")
    if DEBUG:
        os.makedirs(debug_dir, exist_ok=True)

    for img_path in img_files:
        fname = os.path.splitext(os.path.basename(img_path))[0]

        for task_name, spec in TASKS.items():
            try:
                # 마스크 생성 분기
                if spec.get("use_region", False):
                    mask = person_region_mask(
                        img_path,
                        bundle,
                        region=spec.get("region", "upper"),
                        person_query=("person",),
                        box_threshold=BOX_TH,
                        text_threshold=TEXT_TH,
                        dilate_ksize=DILATE_K,
                        pick=PICK,
                        debug_save=(os.path.join(debug_dir, f"{fname}_{task_name}_overlay.jpg") if DEBUG else None)
                    )
                else:
                    mask = grounded_sam_mask(
                        img_path,
                        bundle,
                        text_queries=spec.get("queries", ("person",)),
                        box_threshold=BOX_TH,
                        text_threshold=TEXT_TH,
                        dilate_ksize=DILATE_K,
                        pick=PICK,
                        debug_save=(os.path.join(debug_dir, f"{fname}_{task_name}_overlay.jpg") if DEBUG else None)
                    )

                if mask is None:
                    log_append(log_path, f"NO_MASK (task={task_name}) :: {img_path}")
                    continue

                out_img = inpaint_once(
                    bundle.pipe, img_path, mask,
                    positive_prompt=spec["pos"],
                    negative_prompt=spec["neg"],
                    size=SIZE, steps=STEPS, guidance=GUIDANCE,
                    strength=spec["strength"], seed=SEED
                )

                out_path = os.path.join(out_dir_map[task_name], f"{fname}_{task_name}.png")
                out_img.save(out_path)
                print(f"[OK] {task_name} :: {fname}")

            except Exception as e:
                log_append(log_path, f"ERROR (task={task_name}) :: {img_path} :: {type(e).__name__}: {e}")
                print(f"[ERR] {task_name} :: {fname} :: {e}")

    print("완료!")
    print("출력 폴더:")
    for k, v in out_dir_map.items():
        print(f" - {k}: {v}")
    print(f"스킵/에러 로그: {log_path}")
