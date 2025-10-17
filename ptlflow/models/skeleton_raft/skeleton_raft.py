"""Skeleton-aware RAFT variant tailored for crack correspondence estimation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from ptlflow.models.raft.raft import RAFT
from ptlflow.models.raft.corr import get_corr_block
from ptlflow.models.raft.utils import upflow8
from ptlflow.utils.registry import register_model, trainable
from ptlflow.utils.utils import forward_interpolate_batch


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
        _ = (
            charbonnier_eps,
            truncation_threshold,
            tangent_weight,
            smooth_weight_parallel,
            smooth_weight_perpendicular,
            cycle_weight,
            use_skeleton_loss,
            use_distance_loss,
            use_tangent_loss,
        )  # Kept for backwards-compatible init signature.

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
