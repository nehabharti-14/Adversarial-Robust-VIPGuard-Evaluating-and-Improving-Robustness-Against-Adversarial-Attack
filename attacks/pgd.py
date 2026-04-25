"""
PGD (Projected Gradient Descent) adversarial attack against TransFace embeddings.

Attack type: Untargeted dodge attack (iterative, stronger than FGSM).
Goal: Push the adversarial image's face embedding away from the VIP center embedding
      over multiple steps, projecting back into the epsilon-ball after each step.

Reference: Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018)

CRITICAL IMPLEMENTATION NOTE:
  In the PGD loop, x_adv MUST be detached at the start of each iteration before
  calling requires_grad_(True). Without detach(), the computation graph accumulates
  across steps and causes an OOM error on GPU.

  Also: FG_Face.face_emb_forward() is @torch.no_grad() — attacks must call
  fg_face.face_model(x) (the raw VisionTransformer) directly.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from attacks.fgsm import preprocess_for_attack, postprocess_from_attack


def pgd_attack(
    face_backbone: torch.nn.Module,
    img_bgr: np.ndarray,
    center: torch.Tensor,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    num_steps: int = 10,
    device: str = "cuda",
    random_start: bool = True,
) -> np.ndarray:
    """
    Multi-step PGD dodge attack on TransFace.

    At each step, computes gradient of cosine similarity w.r.t. the current
    adversarial image, takes an alpha-sized step in the gradient direction,
    then projects back into the epsilon-ball around the original image.

    Args:
        face_backbone: The raw VisionTransformer (fg_face.face_model), NOT FG_Face.
        img_bgr:       Original image as BGR uint8 numpy array (H, W, 3).
        center:        VIP embedding center, shape (512,), on any device.
        epsilon:       Total perturbation budget in pixel space [0,255]. Default 8/255.
        alpha:         Per-step perturbation size in pixel space. Default 2/255.
        num_steps:     Number of PGD iterations. Default 10.
        device:        Device for computation. Default 'cuda'.
        random_start:  If True, initialize with a random perturbation within epsilon-ball.
                       Random start avoids saddle points and produces stronger attacks.

    Returns:
        Adversarial image as BGR uint8 numpy array (112, 112, 3).
    """
    face_backbone.eval()  # Disable BatchNorm running-stats update and random masking
    center_norm = F.normalize(center.float().to(device), dim=0).detach()

    # Convert epsilon and alpha from pixel-space to normalized [-1,1] space
    eps_n = epsilon / 0.5
    alpha_n = alpha / 0.5

    # Preprocess original image (no gradient needed on x_orig)
    x_orig = preprocess_for_attack(img_bgr, device).detach()

    # Initialize adversarial image
    if random_start:
        noise = torch.empty_like(x_orig).uniform_(-eps_n, eps_n)
        x_adv = (x_orig + noise).clamp(-1.0, 1.0).detach()
    else:
        x_adv = x_orig.clone().detach()

    for _ in range(num_steps):
        # CRITICAL: detach before setting requires_grad to prevent graph accumulation
        x_adv = x_adv.detach().requires_grad_(True)

        # Forward pass — disable autocast to prevent FP16 underflow in gradients
        with torch.amp.autocast('cuda', enabled=False):
            feat, _, _ = face_backbone(x_adv.float())

        emb_norm = F.normalize(feat.squeeze(0), dim=0)

        # Dodge attack: MINIMIZE cosine similarity → gradient DESCENT on sim.
        # Subtract the sign of the gradient so the embedding moves away from center.
        sim = torch.dot(emb_norm, center_norm)
        sim.backward()

        with torch.no_grad():
            # Step in NEGATIVE gradient direction (descent to minimize sim)
            x_adv = x_adv - alpha_n * x_adv.grad.sign()

            # Project back into epsilon-ball around original image
            delta = (x_adv - x_orig).clamp(-eps_n, eps_n)
            x_adv = (x_orig + delta).clamp(-1.0, 1.0)

    return postprocess_from_attack(x_adv)
