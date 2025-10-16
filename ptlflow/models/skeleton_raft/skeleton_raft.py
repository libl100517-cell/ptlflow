"""Skeleton-aware RAFT variant tailored for crack correspondence estimation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.models.raft.raft import RAFT
from ptlflow.models.raft.corr import get_corr_block
from ptlflow.models.raft.utils import upflow8
from ptlflow.utils.registry import register_model, trainable
from ptlflow.utils.utils import forward_interpolate_batch


def _coords_grid(batch: int, height: int, width: int, *, device, dtype) -> torch.Tensor:
    """Create a grid of absolute pixel coordinates."""

    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    coords = torch.stack((xx, yy), dim=0)
    return coords.unsqueeze(0).repeat(batch, 1, 1, 1)


def _normalize_grid(grid: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert absolute pixel coordinates to the normalized grid used by grid_sample."""

    x = grid[:, 0]
    y = grid[:, 1]
    x_norm = 2.0 * (x / max(width - 1, 1) - 0.5)
    y_norm = 2.0 * (y / max(height - 1, 1) - 0.5)
    return torch.stack((x_norm, y_norm), dim=-1)


def _charbonnier(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def _masked_average(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.sum()
    if weight <= 0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return (values * mask).sum() / weight


class SkeletonSequenceLoss(nn.Module):
    """Loss tailored to skeletonised crack correspondence."""

    def __init__(
        self,
        gamma: float = 0.8,
        charbonnier_eps: float = 1e-3,
        truncation_threshold: float = 2.0,
        tangent_weight: float = 0.5,
        smooth_weight_parallel: float = 0.1,
        smooth_weight_perpendicular: float = 0.3,
        cycle_weight: float = 0.5,
        use_skeleton_loss: bool = True,
        use_distance_loss: bool = True,
        use_tangent_loss: bool = True,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.charbonnier_eps = charbonnier_eps
        self.truncation_threshold = truncation_threshold
        self.tangent_weight = tangent_weight
        self.smooth_weight_parallel = smooth_weight_parallel
        self.smooth_weight_perpendicular = smooth_weight_perpendicular
        self.cycle_weight = cycle_weight
        self.use_skeleton_loss = use_skeleton_loss
        self.use_distance_loss = use_distance_loss
        self.use_tangent_loss = use_tangent_loss

    def forward(self, outputs: Dict[str, List[torch.Tensor]], inputs: Dict[str, torch.Tensor]):
        flow_preds_fwd: List[torch.Tensor] = outputs["flow_preds"]
        flow_preds_bwd: List[torch.Tensor] = outputs["flow_preds_bwd"]

        skeletons = inputs.get("skeletons")
        distances = inputs.get("distances")
        tangents = inputs.get("tangents")
        bands = inputs.get("narrow_bands")

        assert skeletons is not None, "Dataset must provide 'skeletons' for SkeletonRAFT loss."
        assert distances is not None, "Dataset must provide 'distances' for SkeletonRAFT loss."
        assert bands is not None, "Dataset must provide 'narrow_bands' for SkeletonRAFT loss."

        s1, s2 = skeletons[:, 0], skeletons[:, 1]
        d1, d2 = distances[:, 0], distances[:, 1]
        b1, b2 = bands[:, 0], bands[:, 1]

        if tangents is not None:
            t1, t2 = tangents[:, 0], tangents[:, 1]
        else:
            t1 = t2 = None

        n_predictions = len(flow_preds_fwd)
        assert len(flow_preds_bwd) == n_predictions

        loss = torch.zeros((), device=flow_preds_fwd[0].device)
        total_weight = 0.0
        log_components = {
            "primary": torch.zeros((), device=loss.device),
            "tangent": torch.zeros((), device=loss.device),
            "cycle": torch.zeros((), device=loss.device),
            "smooth_parallel": torch.zeros((), device=loss.device),
            "smooth_perpendicular": torch.zeros((), device=loss.device),
        }

        for i in range(n_predictions):
            weight = self.gamma ** (n_predictions - i - 1)
            lf = self._direction_terms(
                flow_preds_fwd[i],
                s1,
                s2,
                d1,
                d2,
                b1,
                b2,
                t1,
                t2,
            )
            lb = self._direction_terms(
                flow_preds_bwd[i],
                s2,
                s1,
                d2,
                d1,
                b2,
                b1,
                t2,
                t1,
            )
            cycle = self._cycle_terms(
                flow_preds_fwd[i],
                flow_preds_bwd[i],
                b2,
                b1,
            )

            primary = 0.5 * (lf["primary"] + lb["primary"])
            tangent = 0.5 * (lf["tangent"] + lb["tangent"])
            smooth_parallel = 0.5 * (lf["smooth_parallel"] + lb["smooth_parallel"])
            smooth_perpendicular = 0.5 * (lf["smooth_perpendicular"] + lb["smooth_perpendicular"])

            iter_loss = primary
            iter_loss = iter_loss + self.tangent_weight * tangent
            iter_loss = iter_loss + self.cycle_weight * cycle
            iter_loss = (
                iter_loss
                + self.smooth_weight_parallel * smooth_parallel
                + self.smooth_weight_perpendicular * smooth_perpendicular
            )

            loss = loss + weight * iter_loss
            total_weight += weight

            log_components["primary"] = log_components["primary"] + weight * primary
            log_components["tangent"] = log_components["tangent"] + weight * tangent
            log_components["cycle"] = log_components["cycle"] + weight * cycle
            log_components["smooth_parallel"] = (
                log_components["smooth_parallel"] + weight * smooth_parallel
            )
            log_components["smooth_perpendicular"] = (
                log_components["smooth_perpendicular"] + weight * smooth_perpendicular
            )

        loss = loss / max(total_weight, 1.0)
        for k in log_components:
            log_components[k] = log_components[k] / max(total_weight, 1.0)

        log_components["loss"] = loss
        return log_components

    def _direction_terms(
        self,
        flow: torch.Tensor,
        src_skel: torch.Tensor,
        tgt_skel: torch.Tensor,
        src_dist: torch.Tensor,
        tgt_dist: torch.Tensor,
        src_band: torch.Tensor,
        tgt_band: torch.Tensor,
        src_tangent: Optional[torch.Tensor],
        tgt_tangent: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = flow.device
        batch, _, height, width = flow.shape
        coords0 = _coords_grid(batch, height, width, device=device, dtype=flow.dtype)
        coords1 = coords0 + flow
        grid = _normalize_grid(coords1, height, width)

        warped_skel = F.grid_sample(
            src_skel,
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        )
        warped_dist = F.grid_sample(
            src_dist,
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        )
        warped_band = F.grid_sample(
            src_band,
            grid,
            mode="nearest",
            align_corners=True,
            padding_mode="zeros",
        )
        if src_tangent is not None and tgt_tangent is not None:
            warped_tangent = F.grid_sample(
                src_tangent,
                grid,
                mode="bilinear",
                align_corners=True,
                padding_mode="zeros",
            )
        else:
            warped_tangent = None

        valid_mask = (
            (coords1[:, 0:1] >= 0)
            & (coords1[:, 0:1] <= width - 1)
            & (coords1[:, 1:2] >= 0)
            & (coords1[:, 1:2] <= height - 1)
        ).float()
        valid_mask = valid_mask * tgt_band * warped_band

        losses = {
            "primary": torch.zeros((), device=device),
            "tangent": torch.zeros((), device=device),
            "smooth_parallel": torch.zeros((), device=device),
            "smooth_perpendicular": torch.zeros((), device=device),
        }

        if self.use_skeleton_loss:
            diff = torch.abs(warped_skel - tgt_skel)
            if self.truncation_threshold > 0:
                mask = (diff <= self.truncation_threshold).float() * valid_mask
            else:
                mask = valid_mask
            charb = _charbonnier(diff, self.charbonnier_eps)
            losses["primary"] = losses["primary"] + _masked_average(charb, mask)

        if self.use_distance_loss:
            diff = warped_dist - tgt_dist
            if self.truncation_threshold > 0:
                mask = (diff.abs() <= self.truncation_threshold).float() * valid_mask
            else:
                mask = valid_mask
            charb = _charbonnier(diff.abs(), self.charbonnier_eps)
            losses["primary"] = losses["primary"] + _masked_average(charb, mask)

        if self.use_skeleton_loss and self.use_distance_loss:
            losses["primary"] = losses["primary"] * 0.5

        if self.use_tangent_loss and warped_tangent is not None and tgt_tangent is not None:
            warped_tangent = F.normalize(warped_tangent, dim=1)
            tgt_norm = F.normalize(tgt_tangent, dim=1)
            dot = (warped_tangent * tgt_norm).sum(dim=1, keepdim=True)
            tangent_cost = (1.0 - dot).clamp(min=0.0)
            losses["tangent"] = _masked_average(tangent_cost, valid_mask * (tgt_skel > 0.5))

        grad_dx = F.pad(flow[:, :, :, 1:] - flow[:, :, :, :-1], (0, 1, 0, 0))
        grad_dy = F.pad(flow[:, :, 1:, :] - flow[:, :, :-1, :], (0, 0, 0, 1))

        dist_dx = F.pad(tgt_dist[:, :, :, 1:] - tgt_dist[:, :, :, :-1], (0, 1, 0, 0))
        dist_dy = F.pad(tgt_dist[:, :, 1:, :] - tgt_dist[:, :, :-1, :], (0, 0, 0, 1))
        normal = torch.stack((dist_dx, dist_dy), dim=1)
        normal = F.normalize(normal, dim=1)
        tangent_vec = torch.stack((-normal[:, 1], normal[:, 0]), dim=1)

        tangent_grad = torch.abs(grad_dx * tangent_vec[:, 0:1] + grad_dy * tangent_vec[:, 1:2]).sum(
            dim=1, keepdim=True
        )
        normal_grad = torch.abs(grad_dx * normal[:, 0:1] + grad_dy * normal[:, 1:2]).sum(
            dim=1, keepdim=True
        )

        losses["smooth_parallel"] = _masked_average(tangent_grad, tgt_band)
        losses["smooth_perpendicular"] = _masked_average(normal_grad, tgt_band)

        return losses

    def _cycle_terms(
        self,
        flow_fwd: torch.Tensor,
        flow_bwd: torch.Tensor,
        tgt_band: torch.Tensor,
        src_band: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = flow_fwd.shape
        coords0 = _coords_grid(batch, height, width, device=flow_fwd.device, dtype=flow_fwd.dtype)
        coords1 = coords0 + flow_fwd
        grid = _normalize_grid(coords1, height, width)

        warped_bwd = F.grid_sample(
            flow_bwd,
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        )
        warped_src_band = F.grid_sample(
            src_band,
            grid,
            mode="nearest",
            align_corners=True,
            padding_mode="zeros",
        )
        valid_mask = (
            (coords1[:, 0:1] >= 0)
            & (coords1[:, 0:1] <= width - 1)
            & (coords1[:, 1:2] >= 0)
            & (coords1[:, 1:2] <= height - 1)
        ).float()
        mask = valid_mask * tgt_band * warped_src_band

        cycle = torch.abs(flow_fwd + warped_bwd).sum(dim=1, keepdim=True)
        return _masked_average(cycle, mask)


@register_model
@trainable
class skeleton_raft(RAFT):
    """RAFT variant that operates on skeleton encodings instead of RGB images."""

    def __init__(
        self,
        input_channels: int = 2,
        corr_levels: int = 4,
        corr_radius: int = 4,
        dropout: float = 0.0,
        gamma: float = 0.8,
        max_flow: float = 400.0,
        iters: int = 24,
        alternate_corr: bool = False,
        charbonnier_eps: float = 1e-3,
        truncation_threshold: float = 2.0,
        tangent_weight: float = 0.5,
        smooth_weight_parallel: float = 0.1,
        smooth_weight_perpendicular: float = 0.3,
        cycle_weight: float = 0.5,
        use_skeleton_loss: bool = True,
        use_distance_loss: bool = True,
        use_tangent_loss: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            dropout=dropout,
            gamma=gamma,
            max_flow=max_flow,
            iters=iters,
            alternate_corr=alternate_corr,
            image_channels=input_channels,
            **kwargs,
        )
        self.loss_fn = SkeletonSequenceLoss(
            gamma=gamma,
            charbonnier_eps=charbonnier_eps,
            truncation_threshold=truncation_threshold,
            tangent_weight=tangent_weight,
            smooth_weight_parallel=smooth_weight_parallel,
            smooth_weight_perpendicular=smooth_weight_perpendicular,
            cycle_weight=cycle_weight,
            use_skeleton_loss=use_skeleton_loss,
            use_distance_loss=use_distance_loss,
            use_tangent_loss=use_tangent_loss,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=0.0,
            bgr_mult=1.0,
            bgr_to_rgb=False,
            resize_mode="pad",
            pad_mode="constant",
            pad_value=0.0,
            pad_two_side=True,
        )

        image1 = images[:, 0]
        image2 = images[:, 1]

        fmap1, fmap2 = self.fnet([image1, image2])
        corr_fn = get_corr_block(
            fmap1=fmap1,
            fmap2=fmap2,
            radius=self.corr_radius,
            num_levels=self.corr_levels,
            alternate_corr=self.alternate_corr,
        )
        flow_preds_fwd, flow_small_fwd, flow_up_fwd = self._run_direction(
            image1,
            image2,
            fmap1,
            fmap2,
            corr_fn,
            image_resizer,
            inputs.get("prev_preds"),
            forward_key="flow_small",
        )

        # Backward direction reuses swapped features
        corr_fn_bwd = get_corr_block(
            fmap1=fmap2,
            fmap2=fmap1,
            radius=self.corr_radius,
            num_levels=self.corr_levels,
            alternate_corr=self.alternate_corr,
        )
        flow_preds_bwd, flow_small_bwd, flow_up_bwd = self._run_direction(
            image2,
            image1,
            fmap2,
            fmap1,
            corr_fn_bwd,
            image_resizer,
            inputs.get("prev_preds"),
            forward_key="flow_small_bwd",
        )

        outputs: Dict[str, torch.Tensor | List[torch.Tensor]] = {
            "flows": flow_up_fwd[:, None],
            "flow_preds": flow_preds_fwd,
            "flow_small": flow_small_fwd,
            "flow_bwd": flow_up_bwd[:, None],
            "flow_preds_bwd": flow_preds_bwd,
            "flow_small_bwd": flow_small_bwd,
        }
        return outputs

    def _run_direction(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        corr_fn,
        image_resizer,
        prev_preds: Optional[Dict[str, torch.Tensor]],
        forward_key: str,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        hdim = self.hidden_dim
        cdim = self.context_dim

        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if prev_preds is not None and prev_preds.get(forward_key) is not None:
            forward_flow = forward_interpolate_batch(prev_preds[forward_key])
            coords1 = coords1 + forward_flow

        flow_predictions: List[torch.Tensor] = []

        for _ in range(self.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow

            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = self.postprocess_predictions(
                flow_up,
                image_resizer,
                is_flow=True,
            )
            flow_predictions.append(flow_up)

        flow_small = coords1 - coords0
        flow_small = self.postprocess_predictions(flow_small, image_resizer, is_flow=True)
        flow_up_final = flow_predictions[-1]

        return flow_predictions, flow_small, flow_up_final
