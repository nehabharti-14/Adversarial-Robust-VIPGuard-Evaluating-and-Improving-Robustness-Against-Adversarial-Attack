"""
Shared utilities for VIPGuard adversarial robustness scripts.

Centralises functions that would otherwise be duplicated between
generate_adversarial_dataset.py and evaluate_robustness.py.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ─── Face Model ──────────────────────────────────────────────────────────────

def load_face_fg(device: str = 'cpu'):
    """
    Load FG_Face wrapper (for @no_grad similarity computation).
    Returns the full FG_Face object so callers can use face_emb_forward.
    """
    from Models.Face_Model.FaceModel import FG_Face
    fg_face = FG_Face(attributes=[], token_dim=3584, model_name='transface')
    fg_face.face_model.eval()
    fg_face.face_model.to(device)
    return fg_face


def load_face_backbone(device: str = 'cuda') -> torch.nn.Module:
    """
    Load the raw TransFace VisionTransformer for gradient-based attacks.

    Returns fg_face.face_model directly — bypasses the @no_grad decorator
    on FG_Face.face_emb_forward so gradients flow during attack generation.
    """
    fg_face = load_face_fg(device)
    return fg_face.face_model


# ─── VIP Center ──────────────────────────────────────────────────────────────

def load_center(id_name: str, center_dir: str = './FaceDATA/FaceEmb_Center') -> torch.Tensor:
    """
    Load the VIP embedding center for a given identity.

    Returns: shape (512,) float32 tensor on CPU.
    The .ckpt file is a raw tensor (confirmed from inference.ipynb), but
    a dict fallback is included for safety.
    """
    path = os.path.join(center_dir, f'{id_name}.ckpt')
    center = torch.load(path, map_location='cpu')
    if isinstance(center, dict):
        center = center.get('face_emb', next(iter(center.values())))
    return center.float()


# ─── Similarity ──────────────────────────────────────────────────────────────

def cosine_sim_score(emb: torch.Tensor, center: torch.Tensor) -> int:
    """
    Compute the 0–100 similarity score used throughout VIPGuard.

    Matches inference.ipynb formula:
      sim = int((0.5 + 0.5 * cosine_similarity(emb, center)) * 100)
    Both inputs are normalised internally so raw embeddings can be passed.
    """
    emb_n = F.normalize(emb.float().cpu(), dim=0)
    ctr_n = F.normalize(center.float().cpu(), dim=0)
    cos = torch.dot(emb_n, ctr_n).item()
    return int((0.5 + 0.5 * cos) * 100)


def sim_from_path(img_path: str, fg_face, center: torch.Tensor) -> int:
    """Compute similarity score for an image file using FG_Face."""
    img = cv2.imread(img_path)
    if img is None:
        return 50
    return sim_from_array(img, fg_face, center)


def sim_from_array(img_bgr: np.ndarray, fg_face, center: torch.Tensor) -> int:
    """Compute similarity score for a BGR uint8 numpy image using FG_Face."""
    img = cv2.resize(img_bgr, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
    x = (x - 0.5) / 0.5
    # Move input to same device as face model
    device = next(fg_face.face_model.parameters()).device
    x = x.to(device)
    emb = fg_face.face_emb_forward(x).squeeze()
    return cosine_sim_score(emb, center)


# ─── VIP Token Loading ────────────────────────────────────────────────────────

def load_vip_token_into(model, vip_token_path: str) -> None:
    """
    Swap VIP tokens into an already-loaded Wrapper model.

    Handles both checkpoint formats:
      - dict: {'vip_token': tensor}
      - raw tensor / Parameter
    This allows the VLM to be loaded once and re-used across identities
    by calling this function to switch the active VIP token.
    """
    state = torch.load(vip_token_path, map_location='cpu')
    target = model.vl_model.facechecker.face_checker.vip_prompt
    if isinstance(state, dict) and 'vip_token' in state:
        token = state['vip_token']
    elif isinstance(state, (torch.Tensor, torch.nn.Parameter)):
        token = state
    else:
        model.load_vip(state)
        return
    # If token count differs (e.g. checkpoint has 32 tokens, model uses 8),
    # take the first N tokens to match the model's allocated size.
    if token.shape[0] != target.shape[0]:
        token = token[:target.shape[0]]
    # Use copy_ to preserve the existing parameter's dtype and device
    target.data.copy_(token.to(dtype=target.dtype, device=target.device))
