"""
FGSM (Fast Gradient Sign Method) adversarial attack against TransFace embeddings.

Attack type: Untargeted dodge attack.
Goal: Push the adversarial image's face embedding away from the VIP center embedding,
      causing the face similarity score to drop and (ideally) fool VIPGuard.

Reference: Goodfellow et al. "Explaining and Harnessing Adversarial Examples" (2015)

CRITICAL IMPLEMENTATION NOTE:
  FG_Face.face_emb_forward() is decorated with @torch.no_grad(), which blocks gradient
  flow. Attacks must call fg_face.face_model(x) directly (the raw VisionTransformer).
  The return is a 3-tuple: (feat, weight, patch_entropy). Use feat (shape: [B, 512]).

Epsilon convention:
  epsilon=8/255 is specified in pixel space [0,255].
  Internally, images are normalized via (x/255 - 0.5) / 0.5 -> [-1, 1].
  Mapping: eps_normalized = epsilon_pixel / 0.5 = 8/127.5 ≈ 0.0627
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def preprocess_for_attack(img_bgr: np.ndarray, device: str) -> torch.Tensor:
    """
    Convert a BGR uint8 image to a normalized float32 tensor suitable for TransFace.

    Args:
        img_bgr: BGR uint8 numpy array of shape (H, W, 3).
        device: Target device string, e.g. 'cuda' or 'cpu'.

    Returns:
        Tensor of shape (1, 3, 112, 112), dtype float32, range [-1, 1].
        Does NOT have requires_grad set — caller must set it if needed.
    """
    img = cv2.resize(img_bgr, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # (3, 112, 112)
    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
    x = (x - 0.5) / 0.5  # -> [-1, 1]
    return x.to(device)


def postprocess_from_attack(adv_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized float32 adversarial tensor back to a BGR uint8 image.

    Args:
        adv_tensor: Tensor of shape (1, 3, 112, 112), range [-1, 1].

    Returns:
        BGR uint8 numpy array of shape (112, 112, 3).
    """
    adv = adv_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    adv = ((adv * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(adv, cv2.COLOR_RGB2BGR)


def fgsm_attack(
    face_backbone: torch.nn.Module,
    img_bgr: np.ndarray,
    center: torch.Tensor,
    epsilon: float = 8 / 255,
    device: str = "cuda",
) -> np.ndarray:
    """
    Single-step FGSM dodge attack on TransFace.

    Computes the gradient of cosine similarity (between the test embedding and the
    VIP center) with respect to the input image, then adds epsilon * sign(gradient)
    to push the embedding away from the VIP center.

    Args:
        face_backbone: The raw VisionTransformer (fg_face.face_model), NOT FG_Face.
                       Must be the unwrapped model so gradients flow through it.
        img_bgr:       Original image as BGR uint8 numpy array (H, W, 3).
        center:        VIP embedding center, shape (512,), on any device.
        epsilon:       Perturbation budget in pixel space [0, 255]. Default 8/255.
        device:        Device for computation. Default 'cuda'.

    Returns:
        Adversarial image as BGR uint8 numpy array (112, 112, 3).
    """
    face_backbone.eval()  # Disable BatchNorm running-stats update and random masking
    center_norm = F.normalize(center.float().to(device), dim=0).detach()

    # Convert epsilon from pixel-space to normalized [-1,1] space
    eps_norm = epsilon / 0.5

    # Preprocess and enable gradient tracking on input
    x = preprocess_for_attack(img_bgr, device).requires_grad_(True)

    # Forward pass — disable autocast to prevent FP16 underflow in gradients
    with torch.amp.autocast('cuda', enabled=False):
        feat, _, _ = face_backbone(x.float())

    emb_norm = F.normalize(feat.squeeze(0), dim=0)

    # Dodge attack: MINIMIZE cosine similarity → gradient DESCENT on sim.
    # The gradient ∇_x sim points toward increasing sim, so we subtract it.
    sim = torch.dot(emb_norm, center_norm)
    sim.backward()

    with torch.no_grad():
        grad_sign = x.grad.sign()
        x_adv = (x - eps_norm * grad_sign).clamp(-1.0, 1.0)

    return postprocess_from_attack(x_adv)
