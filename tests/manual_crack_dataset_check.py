"""Utility to visualize and sanity-check crack skeleton dataset samples.

This script inspects the :class:`~ptlflow.data.crack_skeleton_dataset.CrackSkeletonDataset`
outputs, saves the first few generated image pairs, and reports simple heuristics
that can highlight potential data generation issues.

Example usage::

    python tests/manual_crack_dataset_check.py \
        --root /path/to/crack/masks \
        --output ./debug_pairs \
        --count 4
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imports needed only for type hints
    import cv2  # type: ignore


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale an array to ``uint8`` for visualization."""
    finite = np.isfinite(image)
    if not np.all(finite):
        image = np.where(finite, image, 0.0)
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - minimum) / (maximum - minimum)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def _build_visualization(channels: np.ndarray, cv_module: "cv2") -> np.ndarray:
    """Create a side-by-side visualization of the encoded crack channels."""
    if channels.ndim != 3:
        raise ValueError("Expected (H, W, C) array for visualization")

    tiles: List[np.ndarray] = []

    skeleton = _normalize_to_uint8(channels[..., 0])
    tiles.append(cv_module.cvtColor(skeleton, cv_module.COLOR_GRAY2BGR))

    if channels.shape[-1] >= 2:
        distance = _normalize_to_uint8(channels[..., 1])
        tiles.append(cv_module.cvtColor(distance, cv_module.COLOR_GRAY2BGR))

    if channels.shape[-1] >= 4:
        tangent_sin = channels[..., 2]
        tangent_cos = channels[..., 3]
        angle = np.arctan2(tangent_sin, tangent_cos)
        angle_norm = (angle + np.pi) / (2 * np.pi)
        tangent = np.clip(np.round(angle_norm * 255.0), 0, 255).astype(np.uint8)
        tiles.append(cv_module.applyColorMap(tangent, cv_module.COLORMAP_TURBO))

    if channels.shape[-1] >= 6:
        branch = np.maximum(channels[..., 4], channels[..., 5])
        branch_vis = _normalize_to_uint8(branch)
        tiles.append(cv_module.cvtColor(branch_vis, cv_module.COLOR_GRAY2BGR))

    return np.concatenate(tiles, axis=1)


def _mask_to_bgr(mask: Optional[np.ndarray], cv_module: "cv2") -> Optional[np.ndarray]:
    """Convert a binary mask to a 3-channel visualization tile."""
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.size == 0:
        return None
    while arr.ndim > 2:
        arr = arr[..., 0]
    arr = np.squeeze(arr)
    arr_uint8 = np.clip(np.round(arr.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    return cv_module.cvtColor(arr_uint8, cv_module.COLOR_GRAY2BGR)


def _ensure_tile_size(
    image: np.ndarray, target_shape: Tuple[int, int], cv_module: "cv2"
) -> np.ndarray:
    """Resize a visualization tile so it matches the given ``(H, W)`` shape."""
    height, width = target_shape
    if image.shape[0] == height and image.shape[1] == width:
        return image
    resized = cv_module.resize(image, (width, height), interpolation=cv_module.INTER_NEAREST)
    if resized.ndim == 2:
        resized = cv_module.cvtColor(resized, cv_module.COLOR_GRAY2BGR)
    return resized


def _assemble_quadrant_image(
    mask_tiles: List[np.ndarray],
    viz_tiles: List[np.ndarray],
    cv_module: "cv2",
) -> np.ndarray:
    """Build a 2x2 composite image from mask and visualization tiles."""

    if len(mask_tiles) != 2 or len(viz_tiles) != 2:
        raise ValueError("Expected two mask tiles and two visualization tiles")

    target_shape = mask_tiles[0].shape[:2]
    top_left = _ensure_tile_size(mask_tiles[0], target_shape, cv_module)
    top_right = _ensure_tile_size(viz_tiles[0], target_shape, cv_module)
    bottom_left = _ensure_tile_size(mask_tiles[1], target_shape, cv_module)
    bottom_right = _ensure_tile_size(viz_tiles[1], target_shape, cv_module)

    top_row = cv_module.hconcat([top_left, top_right])
    bottom_row = cv_module.hconcat([bottom_left, bottom_right])
    return cv_module.vconcat([top_row, bottom_row])


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect CrackSkeletonDataset samples and save visualization pairs.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing crack mask images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("crack_dataset_pairs"),
        help="Directory where the visualization PNGs will be stored.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=4,
        help="Number of dataset items (pairs) to inspect.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load (train/val).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible sampling.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size passed to the dataset (set to 0 to disable).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        import cv2 as cv
    except ImportError as exc:  # pragma: no cover - environment specific
        raise SystemExit(
            "OpenCV (cv2) is required to run this script. Please install ``opencv-python`` "
            "or ``opencv-python-headless`` and ensure system dependencies like libGL are available."
        ) from exc

    try:
        from ptlflow.data.crack_skeleton_dataset import CrackSkeletonDataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Could not import CrackSkeletonDataset: " + str(exc)) from exc

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_size = args.patch_size if args.patch_size > 0 else None

    supports_patch_size = False

    dataset_kwargs = {
        "root_dir": str(args.root),
        "split": args.split,
        "rng_seed": args.seed,
    }

    init_signature = inspect.signature(CrackSkeletonDataset.__init__)
    if "skeleton_patch_size" in init_signature.parameters:
        dataset_kwargs["skeleton_patch_size"] = patch_size
        supports_patch_size = True
    elif patch_size is not None:
        print(
            "Warning: installed CrackSkeletonDataset does not support skeleton_patch_size;"
            " falling back to the implementation's default patch sampling behaviour.",
            file=sys.stderr,
        )

    dataset = CrackSkeletonDataset(**dataset_kwargs)

    sample_count = min(args.count, len(dataset))
    problems: List[str] = []
    expected_size = args.patch_size if supports_patch_size and args.patch_size > 0 else None

    for idx in range(sample_count):
        sample = dataset[idx]
        images = sample["images"]
        masks = sample.get("masks") if isinstance(sample, dict) else None
        if len(images) < 2:
            problems.append(f"Sample {idx} did not yield an image pair.")
            continue

        skeleton_counts = []
        mask_tiles: List[np.ndarray] = []
        viz_tiles: List[np.ndarray] = []

        for view_idx, view in enumerate(images[:2]):
            if view.ndim != 3:
                problems.append(
                    f"Sample {idx} view {view_idx} has unexpected shape {view.shape}."
                )
                fallback_size = (
                    (expected_size or 256),
                    (expected_size or 256),
                )
                blank_tile = np.zeros((*fallback_size, 3), dtype=np.uint8)
                mask_tiles.append(blank_tile)
                viz_tiles.append(blank_tile.copy())
                continue

            height, width = view.shape[:2]

            if (
                expected_size is not None
                and (view.shape[0] != expected_size or view.shape[1] != expected_size)
            ):
                problems.append(
                    f"Sample {idx} view {view_idx} shape {view.shape[:2]} does not match"
                    f" requested patch size {expected_size}."
                )

            mask_plane: Optional[np.ndarray] = None
            if masks is not None and view_idx < len(masks):
                mask_candidate = masks[view_idx]
                if mask_candidate is not None:
                    mask_plane = np.asarray(mask_candidate)
                    while mask_plane.ndim > 2:
                        mask_plane = mask_plane[..., 0]
                    mask_plane = np.squeeze(mask_plane)
                    if mask_plane.shape != (height, width):
                        mask_plane = cv.resize(
                            mask_plane.astype(np.float32),
                            (width, height),
                            interpolation=cv.INTER_NEAREST,
                        )
                    if (
                        expected_size is not None
                        and mask_plane.shape != (expected_size, expected_size)
                    ):
                        problems.append(
                            f"Sample {idx} mask {view_idx} shape {mask_plane.shape} does not match"
                            f" requested patch size {expected_size}."
                        )

            mask_tile = _mask_to_bgr(mask_plane, cv)
            if mask_tile is None:
                mask_tile = np.zeros((height, width, 3), dtype=np.uint8)
            mask_tile = _ensure_tile_size(mask_tile, (height, width), cv)
            mask_tiles.append(mask_tile)

            visualization = _build_visualization(view, cv)
            visualization = _ensure_tile_size(visualization, (height, width), cv)
            viz_tiles.append(visualization)

            skeleton_channel = view[..., 0]
            skeleton_nonzero = int(np.count_nonzero(skeleton_channel))
            skeleton_counts.append(skeleton_nonzero)

            if skeleton_nonzero == 0:
                problems.append(
                    f"Sample {idx} view {view_idx} produced an empty skeleton channel."
                )
            if not np.isfinite(view).all():
                problems.append(
                    f"Sample {idx} view {view_idx} contains NaN or Inf values."
                )

        try:
            composite = _assemble_quadrant_image(mask_tiles, viz_tiles, cv)
        except ValueError as exc:
            problems.append(f"Sample {idx} composite generation failed: {exc}")
        else:
            out_path = output_dir / f"pair_{idx:02d}.png"
            if not cv.imwrite(str(out_path), composite):
                problems.append(f"Failed to write visualization to {out_path}.")

        if (
            len(skeleton_counts) == 2
            and skeleton_counts[0] > 0
            and skeleton_counts[1] > 0
            and np.allclose(images[0][..., 0], images[1][..., 0])
        ):
            problems.append(
                f"Sample {idx} skeleton pair is identical; random deformation may be disabled."
            )

    print(f"Saved {sample_count} composite images to {output_dir}")
    if problems:
        print("Potential issues detected:")
        for msg in problems:
            print(f" - {msg}")
        return 1

    print("No obvious issues detected in the inspected pairs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
