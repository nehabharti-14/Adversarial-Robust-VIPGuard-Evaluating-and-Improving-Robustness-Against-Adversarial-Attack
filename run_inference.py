"""
VIPGuard End-to-End Inference Script
Run any image through the full pipeline:
  TransFace similarity scoring -> Qwen2.5-VL verdict -> printed result

Usage examples:
  # Real VIP image (should say Yes)
  python run_inference.py --image ./FaceDATA/Training_Img/id0/r_r_i/id0_1/2.png --id 0

  # Deepfake (should say No)
  python run_inference.py --image ./FaceDATA/Training_Img/id0/r_efs_i/id0_1/fake.png --id 0

  # FGSM adversarial attack (should say Yes with curriculum-trained token)
  python run_inference.py --image ./data/adversarial/fgsm/adv_real/id0/id0_1/adv.png --id 0

  # PGD adversarial attack (should say Yes with curriculum-trained token)
  python run_inference.py --image ./data/adversarial/pgd/adv_real/id0/id0_1/adv.png --id 0

  # Use a custom image file
  python run_inference.py --image /path/to/your/photo.jpg --id 0
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import cv2
import torch.nn.functional as F
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE            = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_PATH   = os.path.join(BASE, "checkpoints", "checkpoints_attr_Stage2_merge")
VIP_TOKEN_DIR   = os.path.join(BASE, "checkpoints", "Stage3_scratch")   # curriculum-trained
FACE_CENTER_DIR = os.path.join(BASE, "FaceDATA", "FaceEmb_Center")
TOKEN_NUM       = 32
MAX_IMG_SIZE    = 224   # resize for memory safety


def parse_args():
    p = argparse.ArgumentParser(description="VIPGuard end-to-end inference")
    p.add_argument("--image", required=True, help="Path to probe image (any format)")
    p.add_argument("--id",    required=True, type=int, help="VIP identity index (0-21)")
    p.add_argument("--token_dir", default=None,
                   help="Override VIP token directory (default: checkpoints/Stage3_scratch)")
    return p.parse_args()


def load_models(device_map="auto"):
    print("\n[1/4] Loading Qwen2.5-VL (32B) with 4-bit quantization...")
    from Models.VIPGuard import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLProcessor,
    )
    from Models.Wrapper import Wrapper
    from Models.Face_Model.FaceModel import FG_Face

    torch.cuda.empty_cache()

    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            PRETRAIN_PATH,
            device_map=device_map,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
        )
        use_4bit = True
    except Exception as e:
        print(f"  [WARN] 4-bit failed ({e}), falling back to float16.")
        vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            PRETRAIN_PATH,
            device_map=device_map,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        use_4bit = False

    processor = Qwen2_5_VLProcessor.from_pretrained(PRETRAIN_PATH)
    model = Wrapper(
        vl_model=vl_model,
        processor=processor,
        optim_facechecker=False,
        vip_token_num=TOKEN_NUM,
    )

    # Cast only FaceChecker to float16 — quantized weights must stay in bnb format
    if use_4bit:
        model.vl_model.facechecker.half()
    else:
        model = model.half()

    face_model = FG_Face(attributes="", token_dim=3584, model_name="transface")
    face_model.to("cpu").eval()

    print("    VLM loaded.")
    return model, processor, face_model


def load_vip_token(model, vip_id, token_dir):
    path = os.path.join(token_dir, f"id{vip_id}", "vip_token.pt")
    if not os.path.exists(path):
        # fallback: pretrained token
        path = os.path.join(BASE, "FaceDATA", "Pretrained_VIPToken", f"id{vip_id}.pt")
    print(f"[2/4] Loading VIP token: {path}")
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "vip_token" in state:
        model.vl_model.facechecker.face_checker.vip_prompt.data = state["vip_token"]
    elif isinstance(state, (torch.Tensor, torch.nn.Parameter)):
        model.vl_model.facechecker.face_checker.vip_prompt.data = state
    else:
        raise ValueError(f"Unrecognised VIP token format: {type(state)}")
    print(f"    Token shape: {model.vl_model.facechecker.face_checker.vip_prompt.data.shape}")


def compute_similarity(img_path, face_model, center):
    print("[3/4] Computing TransFace similarity score...")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
    x = (x - 0.5) / 0.5
    with torch.no_grad():
        emb = face_model.face_emb_forward(x).squeeze()
    emb    = F.normalize(emb, dim=0)
    center = F.normalize(center.float(), dim=0)
    sim = torch.dot(emb, center).item()
    score = int((0.5 + 0.5 * sim) * 100)
    print(f"    Similarity score: {score}/100")
    return score


def run_vlm_inference(img_path, sim_score, model, processor):
    print("[4/4] Running VIPGuard VLM inference...")
    from qwen_vl_utils import process_vision_info

    # Prompt must match Stage3 training format exactly (two <|face_pad|> tokens)
    prompt = (
        f"<|face_pad|>Please determine whether the person in the input image is VIP user. "
        f"The face similarity between the input face and VIP user is {sim_score}/100. "
        f"The face tokens are shown as follows, <|face_pad|> . "
        f"You should first give your answer by 'yes' or 'no'. "
        f"Then, you should explain your reasoning step by step based on different facial attributes."
    )
    message = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text",  "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(message)

    # Resize for memory safety
    for i, img in enumerate(image_inputs):
        h, w = img.size
        if max(h, w) > MAX_IMG_SIZE:
            scale = MAX_IMG_SIZE / max(h, w)
            image_inputs[i] = img.resize((int(w * scale), int(h * scale)))

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
        our_token=True,
        our_token_length=TOKEN_NUM,
        face_pad=True,
        face_length=model.vl_model.facechecker.face_checker.vip_prompt.data.shape[0],
    )
    inputs = {
        k: (v.to("cuda").to(torch.float16) if v.dtype == torch.float32 else v.to("cuda"))
        for k, v in inputs.items()
    }

    print("    Generating response (this takes ~1-3 minutes on first run, be patient)...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,       # greedy — faster and deterministic
            temperature=None,
            top_p=None,
        )

    return processor.batch_decode(output, skip_special_tokens=True)[0]


def print_result(img_path, vip_id, sim_score, raw_output):
    # Extract verdict
    verdict = "UNKNOWN"
    if "<Conclusion>[Yes]" in raw_output or "[Yes]" in raw_output:
        verdict = "YES — Authenticated as VIP"
    elif "<Conclusion>[No]" in raw_output or "[No]" in raw_output:
        verdict = "NO  — Not authenticated"

    # Extract conclusion text
    conclusion = raw_output
    if "<Conclusion>" in raw_output:
        start = raw_output.find("<Conclusion>")
        end   = raw_output.find("</Conclusion>")
        if end != -1:
            conclusion = raw_output[start:end + len("</Conclusion>")]

    print("\n" + "=" * 60)
    print("  VIPGUARD INFERENCE RESULT")
    print("=" * 60)
    print(f"  Image   : {img_path}")
    print(f"  VIP ID  : id{vip_id}")
    print(f"  Sim     : {sim_score}/100")
    print(f"  Verdict : {verdict}")
    print("-" * 60)
    print("  Model output:")
    print(f"  {conclusion}")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    token_dir = args.token_dir or VIP_TOKEN_DIR

    # Verify paths
    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)
    center_path = os.path.join(FACE_CENTER_DIR, f"id{args.id}.ckpt")
    if not os.path.exists(center_path):
        print(f"ERROR: Face centre not found: {center_path}")
        sys.exit(1)

    print("=" * 60)
    print("  VIPGUARD END-TO-END INFERENCE")
    print(f"  Image : {args.image}")
    print(f"  VIP   : id{args.id}")
    print("=" * 60)

    # Load
    model, processor, face_model = load_models()
    load_vip_token(model, args.id, token_dir)
    center = torch.load(center_path, map_location="cpu")

    # Run
    sim   = compute_similarity(args.image, face_model, center)
    out   = run_vlm_inference(args.image, sim, model, processor)
    print_result(args.image, args.id, sim, out)


if __name__ == "__main__":
    main()
