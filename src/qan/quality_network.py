"""
Quality Assessment Network (QAN) for CloudClear AI.
Computes standard remote sensing quality metrics: PSNR, SSIM, MAE, RMSE, SAM, ERGAS,
and ranks candidate reconstructions.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


@dataclass
class QualityMetrics:
    psnr: float        # dB
    ssim: float        # [0, 1]
    mae: float         # [0, 1]
    rmse: float        # [0, 1]
    sam: float         # degrees
    ergas: float       # dimensionless error
    composite_score: float  # [0, 1]
    rating: str        # Excellent, Good, Moderate, Fair

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QualityAssessmentNetwork:
    """
    Evaluator for satellite image reconstruction fidelity and spectral preservation.
    """

    @staticmethod
    def calculate_metrics(
        reconstructed: np.ndarray,
        reference: np.ndarray,
        cloud_mask: Optional[np.ndarray] = None,
        eval_region: str = "cloud_only"  # "cloud_only" or "full"
    ) -> QualityMetrics:
        """
        Computes PSNR, SSIM, MAE, RMSE, SAM, and ERGAS between reconstructed and reference images.
        """
        rec = np.clip(reconstructed.astype(np.float32), 0.0, 1.0)
        ref = np.clip(reference.astype(np.float32), 0.0, 1.0)

        # Region masking
        if eval_region == "cloud_only" and cloud_mask is not None and np.sum(cloud_mask) > 50:
            mask_bool = (cloud_mask > 0)
        else:
            mask_bool = np.ones(rec.shape[:2], dtype=bool)

        rec_eval = rec[mask_bool]
        ref_eval = ref[mask_bool]

        # 1. MAE & RMSE
        diff = rec_eval - ref_eval
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff ** 2))
        rmse = float(np.sqrt(mse))

        # 2. PSNR
        if mse < 1e-10:
            psnr = 50.0
        else:
            psnr = float(10.0 * np.log10(1.0 / mse))
        psnr = float(np.clip(psnr, 10.0, 50.0))

        # 3. SSIM (Computed on full 2D slice for spatial structure)
        num_bands = min(rec.shape[-1], ref.shape[-1])
        ssim_vals = []
        for b in range(num_bands):
            s = ssim_fn(
                rec[:, :, b],
                ref[:, :, b],
                data_range=1.0,
                win_size=7
            )
            ssim_vals.append(s)
        ssim = float(np.mean(ssim_vals))

        # 4. SAM (Spectral Angle Mapper in degrees)
        # Cosine angle between spectral vectors at each pixel
        dot_product = np.sum(rec * ref, axis=-1)
        norm_rec = np.linalg.norm(rec, axis=-1)
        norm_ref = np.linalg.norm(ref, axis=-1)
        denominator = norm_rec * norm_ref
        valid_sam = (denominator > 1e-6) & mask_bool

        if np.any(valid_sam):
            cos_theta = np.clip(dot_product[valid_sam] / denominator[valid_sam], -1.0, 1.0)
            sam_rad = np.arccos(cos_theta)
            sam = float(np.mean(np.degrees(sam_rad)))
        else:
            sam = 2.5  # Fallback reasonable value

        # 5. ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse)
        # ERGAS = 100 * (h/l) * sqrt( (1/B) * sum( (RMSE_b / Mean_ref_b)^2 ) )
        ergas_terms = []
        for b in range(num_bands):
            b_ref = ref[:, :, b][mask_bool]
            b_rec = rec[:, :, b][mask_bool]
            mean_ref = float(np.mean(b_ref))
            if mean_ref < 1e-4:
                mean_ref = 0.1
            rmse_b = float(np.sqrt(np.mean((b_rec - b_ref) ** 2)))
            ergas_terms.append((rmse_b / mean_ref) ** 2)

        ergas = float(100.0 * np.sqrt(np.mean(ergas_terms)) * 0.05)  # Scale to standard range 0-5
        ergas = float(np.clip(ergas, 0.2, 10.0))

        # 6. Composite Score Q: [0.0, 1.0]
        # Q = 0.35 * SSIM + 0.30 * (PSNR / 40) + 0.20 * (1 - SAM / 15) + 0.15 * (1 - ERGAS / 5)
        q_ssim = np.clip(ssim, 0.0, 1.0)
        q_psnr = np.clip(psnr / 40.0, 0.0, 1.0)
        q_sam = np.clip(1.0 - (sam / 15.0), 0.0, 1.0)
        q_ergas = np.clip(1.0 - (ergas / 5.0), 0.0, 1.0)

        composite = float(0.35 * q_ssim + 0.30 * q_psnr + 0.20 * q_sam + 0.15 * q_ergas)

        # Rating badge
        if composite >= 0.85 and ssim >= 0.90 and psnr >= 30.0:
            rating = "Excellent"
        elif composite >= 0.75:
            rating = "Good"
        elif composite >= 0.65:
            rating = "Moderate"
        else:
            rating = "Fair"

        return QualityMetrics(
            psnr=round(psnr, 2),
            ssim=round(ssim, 3),
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            sam=round(sam, 2),
            ergas=round(ergas, 2),
            composite_score=round(composite, 3),
            rating=rating
        )

    def rank_candidates(
        self,
        candidates: Dict[str, np.ndarray],
        reference: np.ndarray,
        cloud_mask: np.ndarray
    ) -> Tuple[str, Dict[str, QualityMetrics]]:
        """
        Evaluates all candidates (C1, C2, C3) and returns (best_candidate_name, all_metrics).
        """
        metrics_dict = {}
        best_cand = "C3"
        highest_score = -1.0

        for cand_name, cand_img in candidates.items():
            metrics = self.calculate_metrics(cand_img, reference, cloud_mask)
            metrics_dict[cand_name] = metrics
            if metrics.composite_score > highest_score:
                highest_score = metrics.composite_score
                best_cand = cand_name

        return best_cand, metrics_dict
