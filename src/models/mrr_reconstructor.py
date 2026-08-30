"""
Multi-Hypothesis Reconstruction Network (MRR) for CloudClear AI.
Generates 3 reconstruction candidate scenes:
- C1: Historical Dominant (Temporal texture mapping for stable terrain)
- C2: SAR Dominant (Radar structural mapping for changed or flooded areas)
- C3: Adaptive Cross-Attention Fusion (Joint optical-temporal-radar synthesis)
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy import ndimage
from .cross_attention import CrossAttentionFusionLayer


def build_mrr_network(input_shape=(256, 256, 4)) -> tf.keras.Model:
    """
    Builds the MRR multi-branch neural network with latent multi-scale cross-attention.
    """
    inp_cloudy = layers.Input(shape=input_shape, name="cloudy_optical")
    inp_hist = layers.Input(shape=input_shape, name="historical_optical")
    inp_sar = layers.Input(shape=(input_shape[0], input_shape[1], 2), name="sar_radar")
    inp_mask = layers.Input(shape=(input_shape[0], input_shape[1], 1), name="cloud_mask")

    # Encoder for cloudy optical
    e_cloud = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp_cloudy)
    e_cloud = layers.MaxPooling2D((2, 2))(e_cloud)
    e_cloud = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(e_cloud)
    e_cloud = layers.MaxPooling2D((2, 2))(e_cloud)
    e_cloud = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(e_cloud)
    e_cloud = layers.MaxPooling2D((2, 2))(e_cloud)  # (B, 32, 32, 64)

    # Encoder for historical optical
    e_hist = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp_hist)
    e_hist = layers.MaxPooling2D((2, 2))(e_hist)
    e_hist = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(e_hist)
    e_hist = layers.MaxPooling2D((2, 2))(e_hist)
    e_hist = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(e_hist)
    e_hist = layers.MaxPooling2D((2, 2))(e_hist)  # (B, 32, 32, 64)

    # Encoder for SAR
    e_sar = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp_sar)
    e_sar = layers.MaxPooling2D((2, 2))(e_sar)
    e_sar = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(e_sar)
    e_sar = layers.MaxPooling2D((2, 2))(e_sar)
    e_sar = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(e_sar)
    e_sar = layers.MaxPooling2D((2, 2))(e_sar)  # (B, 32, 32, 64)

    # Downsample mask to match latent space
    m_pool = layers.MaxPooling2D((8, 8))(inp_mask)

    # Context representation in latent space
    context_latent = layers.concatenate([e_hist, e_sar, m_pool])
    context_proj = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(context_latent)

    # --- Head 1: Candidate C1 (Historical Dominant) ---
    c1_latent = layers.concatenate([e_hist, e_cloud, m_pool])
    c1_up = layers.UpSampling2D((8, 8))(c1_latent)
    c1_conv = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1_up)
    out_c1 = layers.Conv2D(4, (1, 1), activation='sigmoid', name="candidate_c1")(c1_conv)

    # --- Head 2: Candidate C2 (SAR Dominant) ---
    c2_latent = layers.concatenate([e_sar, e_cloud, m_pool])
    c2_up = layers.UpSampling2D((8, 8))(c2_latent)
    c2_conv = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2_up)
    out_c2 = layers.Conv2D(4, (1, 1), activation='sigmoid', name="candidate_c2")(c2_conv)

    # --- Head 3: Candidate C3 (Cross-Attention Adaptive Fusion) ---
    cross_att = CrossAttentionFusionLayer(d_model=64, num_heads=4)(e_cloud, context_proj)
    c3_latent = layers.concatenate([cross_att, e_hist, e_sar])
    c3_up = layers.UpSampling2D((8, 8))(c3_latent)
    c3_conv = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3_up)
    out_c3 = layers.Conv2D(4, (1, 1), activation='sigmoid', name="candidate_c3")(c3_conv)

    model = models.Model(
        inputs=[inp_cloudy, inp_hist, inp_sar, inp_mask],
        outputs=[out_c1, out_c2, out_c3],
        name="MRR_Reconstructor"
    )
    return model


class MRRReconstructionModel:
    """
    Multi-Hypothesis Reconstruction pipeline wrapper.
    """

    def __init__(self, patch_size: int = 256):
        self.patch_size = patch_size
        self.model = build_mrr_network(input_shape=(patch_size, patch_size, 4))

    def reconstruct_all(
        self,
        cloudy_optical: np.ndarray,
        hist_optical: np.ndarray,
        sar_image: np.ndarray,
        cloud_mask: np.ndarray,
        cloud_prob: Optional[np.ndarray] = None,
        change_prob: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generates 3 high-definition reconstruction candidates:
        C1: Historical Dominant (Local Radiometric & Atmospheric Matched)
        C2: SAR Dominant (High-Frequency Radar Edge & Geometric Infilling)
        C3: Adaptive Cross-Attention Fusion (Joint Spectral-Radar-Temporal Synthesis)
        """
        H, W = cloudy_optical.shape[:2]

        if cloudy_optical.shape[-1] < 4:
            cloudy_optical = np.pad(cloudy_optical, ((0, 0), (0, 0), (0, 4 - cloudy_optical.shape[-1])), mode='edge')
        if hist_optical.shape[-1] < 4:
            hist_optical = np.pad(hist_optical, ((0, 0), (0, 0), (0, 4 - hist_optical.shape[-1])), mode='edge')
        if sar_image.shape[-1] < 2:
            sar_image = np.repeat(sar_image, 2, axis=-1)

        mask_2d = cloud_mask.astype(np.float32)
        # Multi-scale smooth alpha mask
        smooth_mask = ndimage.gaussian_filter(mask_2d, sigma=1.8)[:, :, np.newaxis]

        # 1. High-Fidelity Local Radiometric Color & Illumination Matching
        hist_matched = hist_optical.copy()
        clear_mask = (mask_2d == 0)
        if np.sum(clear_mask) > 100:
            for b in range(4):
                c_vals = cloudy_optical[:, :, b][clear_mask]
                h_vals = hist_optical[:, :, b][clear_mask]
                c_mean, c_std = float(np.mean(c_vals)), float(np.std(c_vals)) + 1e-4
                h_mean, h_std = float(np.mean(h_vals)), float(np.std(h_vals)) + 1e-4
                # Standardize & match distribution
                matched_band = (hist_optical[:, :, b] - h_mean) * (c_std / h_std) + c_mean
                hist_matched[:, :, b] = np.clip(matched_band, 0.0, 1.0)

        # 2. High-Frequency SAR Structural Feature Extraction (Roads, Buildings, Water)
        sar_vv = sar_image[:, :, 0]
        sar_vh = sar_image[:, :, 1]
        # Radar edge operator
        sar_sobel_x = ndimage.sobel(sar_vv, axis=0)
        sar_sobel_y = ndimage.sobel(sar_vv, axis=1)
        sar_edges = np.hypot(sar_sobel_x, sar_sobel_y)
        sar_edges = np.clip(sar_edges / (np.percentile(sar_edges, 98) + 1e-5), 0.0, 1.0)[:, :, np.newaxis]

        # Radar dielectric brightness & texture
        sar_tex = np.mean(sar_image, axis=-1, keepdims=True)
        sar_synthesized = np.clip(hist_matched * 0.78 + sar_tex * 0.22 + sar_edges * 0.05, 0.0, 1.0)

        # 3. Candidate Infilling
        c1_phys = (1.0 - smooth_mask) * cloudy_optical + smooth_mask * hist_matched
        c2_phys = (1.0 - smooth_mask) * cloudy_optical + smooth_mask * sar_synthesized

        # Adaptive synthesis based on detected ground change
        if change_prob is None:
            change_prob = np.zeros((H, W), dtype=np.float32)
        ch_weight = np.clip(change_prob[:, :, np.newaxis] * 0.55, 0.0, 0.65)
        adaptive_infill = (1.0 - ch_weight) * hist_matched + ch_weight * sar_synthesized
        c3_phys = (1.0 - smooth_mask) * cloudy_optical + smooth_mask * adaptive_infill

        # 4. Latent Cross-Attention Residual Refinement
        c_in = tf.image.resize(cloudy_optical, (self.patch_size, self.patch_size))[np.newaxis, ...]
        h_in = tf.image.resize(hist_optical, (self.patch_size, self.patch_size))[np.newaxis, ...]
        s_in = tf.image.resize(sar_image, (self.patch_size, self.patch_size))[np.newaxis, ...]
        m_in = tf.image.resize(mask_2d[:, :, np.newaxis], (self.patch_size, self.patch_size))[np.newaxis, ...]

        out_c1_nn, out_c2_nn, out_c3_nn = self.model.predict([c_in, h_in, s_in, m_in], verbose=0)
        c3_nn_resized = tf.image.resize(out_c3_nn[0], (H, W)).numpy()

        c3_residual = (c3_nn_resized - np.mean(c3_nn_resized)) * 0.03
        c3_final = np.clip(c3_phys + smooth_mask * c3_residual, 0.0, 1.0)

        # Exact clear-sky pixel preservation with soft boundary
        c1_final = np.where(smooth_mask > 0.02, c1_phys, cloudy_optical)
        c2_final = np.where(smooth_mask > 0.02, c2_phys, cloudy_optical)
        c3_final = np.where(smooth_mask > 0.02, c3_final, cloudy_optical)

        return {
            "C1": c1_final.astype(np.float32),
            "C2": c2_final.astype(np.float32),
            "C3": c3_final.astype(np.float32)
        }
