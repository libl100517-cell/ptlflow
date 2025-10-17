"""Dataset utilities for crack skeleton self-supervision."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2 as cv
from loguru import logger
import numpy as np
import torch
from torch.utils.data import Dataset


class CrackSkeletonDataset(Dataset):
    """Generate skeleton-based inputs for self-supervised crack correspondence."""

    def __init__(
        self,
        root_dir: str,
        pairs_list: Optional[Sequence[Tuple[str, str]]] = None,
        pairs_file: str = "pairs.txt",
        transform=None,
        split: str = "train",
        mask_threshold: int = 127,
        include_distance_channel: bool = True,
        include_tangent_channel: bool = True,
        include_branch_channel: bool = False,
        tangent_window: int = 5,
        narrow_band_radius: float = 6.0,
        distance_clip: Optional[float] = None,
        distance_normalizer: Optional[float] = None,
        mask_suffixes: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
        recursive: bool = True,
        rng_seed: Optional[int] = None,
        affine_max_rotation: float = 10.0,
        affine_scale_jitter: float = 0.1,
        affine_translation: float = 0.05,
        elastic_alpha: float = 6.0,
        elastic_sigma: float = 4.0,
        width_jitter_radius: int = 1,
        noise_flip_prob: float = 0.0,
        skeleton_patch_size: Optional[int] = 256,
        skeleton_transform_rotation: float = 2.0,
        skeleton_transform_scale: float = 0.02,
        skeleton_transform_translation: float = 0.05,
        skeleton_transform_translation_limit: Optional[float] = 3.0,
        skeleton_occlusion_prob: float = 0.5,
        skeleton_occlusion_count: Tuple[int, int] = (0, 3),
        skeleton_occlusion_radius: Tuple[float, float] = (12.0, 48.0),
        skeleton_noise_std: float = 1.0,
        skeleton_noise_limit: Optional[float] = 3.0,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.mask_threshold = mask_threshold
        self.include_distance_channel = include_distance_channel
        self.include_tangent_channel = include_tangent_channel
        self.include_branch_channel = include_branch_channel
        self.tangent_window = max(3, tangent_window | 1)
        self.narrow_band_radius = narrow_band_radius
        self.distance_clip = distance_clip
        self.distance_normalizer = distance_normalizer

        self.mask_suffixes = tuple(suffix.lower() for suffix in mask_suffixes)
        self.recursive = recursive

        self.affine_max_rotation = float(abs(affine_max_rotation))
        self.affine_scale_jitter = max(0.0, float(affine_scale_jitter))
        self.affine_translation = max(0.0, float(affine_translation))
        self.elastic_alpha = max(0.0, float(elastic_alpha))
        self.elastic_sigma = max(0.0, float(elastic_sigma))
        self.width_jitter_radius = max(0, int(width_jitter_radius))
        self.noise_flip_prob = min(1.0, max(0.0, noise_flip_prob))

        self.use_patch_sampling = skeleton_patch_size is not None and int(skeleton_patch_size) > 0
        self.skeleton_patch_size = int(skeleton_patch_size) if self.use_patch_sampling else None
        self.skeleton_transform_rotation = float(abs(skeleton_transform_rotation))
        self.skeleton_transform_scale = max(0.0, float(skeleton_transform_scale))
        self.skeleton_transform_translation = max(0.0, float(skeleton_transform_translation))
        self.skeleton_transform_translation_limit = (
            None
            if skeleton_transform_translation_limit is None
            else max(0.0, float(skeleton_transform_translation_limit))
        )
        self.skeleton_occlusion_prob = min(1.0, max(0.0, float(skeleton_occlusion_prob)))
        occ_lo, occ_hi = skeleton_occlusion_count
        occ_lo = max(0, int(occ_lo))
        occ_hi = max(occ_lo, int(occ_hi))
        self.skeleton_occlusion_count: Tuple[int, int] = (occ_lo, occ_hi)
        rad_lo, rad_hi = skeleton_occlusion_radius
        rad_lo = max(1.0, float(rad_lo))
        rad_hi = max(rad_lo, float(rad_hi))
        self.skeleton_occlusion_radius: Tuple[float, float] = (rad_lo, rad_hi)
        self.skeleton_noise_std = max(0.0, float(skeleton_noise_std))
        self.skeleton_noise_limit = (
            None if skeleton_noise_limit is None else max(0.0, float(skeleton_noise_limit))
        )

        self._rng = np.random.default_rng(rng_seed)
        self._patch_center_sample_attempts = 8

        self._paired_mode = False
        self.samples: List[Tuple[Path, Path]] = []
        self.mask_paths: List[Path] = []

        if pairs_list is not None:
            self._paired_mode = True
            self.samples = [
                (self.root_dir / Path(p1), self.root_dir / Path(p2)) for p1, p2 in pairs_list
            ]
        else:
            candidate = self.root_dir / pairs_file if pairs_file else None
            if candidate is not None and candidate.exists():
                self._paired_mode = True
                with candidate.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if len(line) == 0 or line.startswith("#"):
                            continue
                        tokens = line.split()
                        if len(tokens) < 2:
                            continue
                        self.samples.append(
                            (self.root_dir / Path(tokens[0]), self.root_dir / Path(tokens[1]))
                        )

        if self._paired_mode:
            if len(self.samples) == 0:
                logger.warning(
                    "No crack mask pairs were found in {}. The dataset will be empty.",
                    self.root_dir,
                )
            else:
                logger.info(
                    "Loading {} crack skeleton pairs from {}.", len(self.samples), self.root_dir
                )
        else:
            self.mask_paths = self._gather_mask_paths()
            if len(self.mask_paths) == 0:
                raise FileNotFoundError(
                    f"No mask files were found in {self.root_dir}. Expected suffixes: {self.mask_suffixes}."
                )
            logger.info(
                "Generating crack skeleton training pairs on-the-fly from {} masks in {}.",
                len(self.mask_paths),
                self.root_dir,
            )

    def __len__(self) -> int:
        return len(self.samples) if self._paired_mode else len(self.mask_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:  # noqa: D401
        if self._paired_mode:
            mask_path_1, mask_path_2 = self.samples[index]
            base_mask_1 = self._read_mask(mask_path_1)
            base_mask_2 = self._read_mask(mask_path_2)
        else:
            mask_path_1 = mask_path_2 = self.mask_paths[index]
            base_mask = self._read_mask(mask_path_1)
            if self.split.lower() == "train":
                rng1 = self._spawn_rng()
                rng2 = self._spawn_rng()
                base_mask_1 = self._augment_mask(base_mask, rng1)
                base_mask_2 = self._augment_mask(base_mask, rng2)
            else:
                base_mask_1 = base_mask.copy()
                base_mask_2 = base_mask.copy()

        if self.use_patch_sampling:
            pair_rng = self._spawn_rng()
            mask1, mask2, skeleton1, skeleton2 = self._generate_skeleton_patch_pair(
                base_mask_1,
                base_mask_2,
                pair_rng,
                apply_random=self.split.lower() == "train",
            )
        else:
            mask1 = base_mask_1
            mask2 = base_mask_2
            skeleton1 = self._skeletonize(mask1)
            skeleton2 = self._skeletonize(mask2)

        sample1 = self._encode_mask(mask1, skeleton_override=skeleton1)
        sample2 = self._encode_mask(mask2, skeleton_override=skeleton2)

        encoded1 = self._compose_channels(sample1)
        encoded2 = self._compose_channels(sample2)

        zero_flow = np.zeros((*encoded1.shape[:2], 2), dtype=np.float32)
        zero_valid = np.zeros((*encoded1.shape[:2], 1), dtype=np.float32)

        sample = {
            "images": [encoded1, encoded2],
            "masks": [
                mask1[..., None].astype(np.float32),
                mask2[..., None].astype(np.float32),
            ],
            "skeletons": [sample1["skeleton"][..., None], sample2["skeleton"][..., None]],
            "distances": [sample1["distance"][..., None], sample2["distance"][..., None]],
            "narrow_bands": [sample1["band"], sample2["band"]],
            "flows": [zero_flow],
            "valids": [zero_valid],
            "meta": {
                "dataset_name": "crack_skeleton",
                "split_name": self.split,
                "paths": [str(mask_path_1), str(mask_path_2)],
            },
        }

        if sample1.get("tangent") is not None and sample2.get("tangent") is not None:
            sample["tangents"] = [sample1["tangent"], sample2["tangent"]]
        if sample1.get("branch") is not None and sample2.get("branch") is not None:
            sample["branches"] = [sample1["branch"], sample2["branch"]]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _read_mask(self, path: Path) -> np.ndarray:
        mask = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read crack mask at {path}")
        mask = (mask > self.mask_threshold).astype(np.uint8)
        return mask

    def _gather_mask_paths(self) -> List[Path]:
        def _match_suffixes(files: Iterable[Path]) -> List[Path]:
            return [
                f
                for f in files
                if f.is_file() and f.suffix.lower() in self.mask_suffixes
            ]

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory {self.root_dir} does not exist")

        files: List[Path]
        if self.recursive:
            files = _match_suffixes(self.root_dir.rglob("*"))
        else:
            files = _match_suffixes(self.root_dir.glob("*"))

        files.sort()
        return files

    def _encode_mask(
        self,
        mask: np.ndarray,
        *,
        skeleton_override: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        skeleton = (
            skeleton_override.astype(np.uint8)
            if skeleton_override is not None
            else self._skeletonize(mask)
        )
        dist = cv.distanceTransform((1 - skeleton).astype(np.uint8), cv.DIST_L2, 3)
        if self.distance_clip is not None:
            dist = np.minimum(dist, self.distance_clip)
        norm = self.distance_normalizer if self.distance_normalizer else max(mask.shape)
        norm = max(float(norm), 1.0)
        dist_norm = np.clip(dist / norm, 0.0, 1.0)
        band = (dist <= self.narrow_band_radius).astype(np.float32)[..., None]

        tangent = self._estimate_tangent_map(skeleton) if self.include_tangent_channel else None
        branch = self._estimate_branch_map(skeleton) if self.include_branch_channel else None

        return {
            "skeleton": skeleton.astype(np.float32),
            "distance": dist.astype(np.float32),
            "distance_norm": dist_norm.astype(np.float32),
            "band": band.astype(np.float32),
            "tangent": tangent.astype(np.float32) if tangent is not None else None,
            "branch": branch.astype(np.float32) if branch is not None else None,
        }

    def _compose_channels(self, components: Dict[str, np.ndarray]) -> np.ndarray:
        channels: List[np.ndarray] = [components["skeleton"]]
        if self.include_distance_channel:
            channels.append(components["distance_norm"])
        if self.include_tangent_channel and components["tangent"] is not None:
            channels.extend(
                [
                    components["tangent"][:, :, 0],
                    components["tangent"][:, :, 1],
                ]
            )
        if self.include_branch_channel and components["branch"] is not None:
            channels.extend(
                [
                    components["branch"][:, :, 0],
                    components["branch"][:, :, 1],
                ]
            )
        stacked = np.stack(channels, axis=-1).astype(np.float32)
        return stacked

    def _augment_mask(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        augmented = mask.astype(np.float32)
        augmented = self._apply_random_affine(augmented, rng)
        if self.elastic_alpha > 0:
            augmented = self._apply_random_elastic(augmented, rng)
        augmented = (augmented > 0.5).astype(np.uint8)
        augmented = self._apply_width_jitter(augmented, rng)
        augmented = self._apply_random_noise(augmented, rng)
        return augmented

    def _apply_random_affine(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if mask.ndim != 2:
            raise ValueError("Expected 2D mask for affine augmentation")
        height, width = mask.shape
        angle = rng.uniform(-self.affine_max_rotation, self.affine_max_rotation)
        scale = rng.uniform(1.0 - self.affine_scale_jitter, 1.0 + self.affine_scale_jitter)
        tx = rng.uniform(-self.affine_translation, self.affine_translation) * width
        ty = rng.uniform(-self.affine_translation, self.affine_translation) * height
        center = (width / 2.0, height / 2.0)
        matrix = cv.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty
        warped = cv.warpAffine(
            mask.astype(np.float32),
            matrix,
            (width, height),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_REFLECT_101,
        )
        return warped

    def _apply_random_elastic(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if mask.ndim != 2:
            raise ValueError("Expected 2D mask for elastic augmentation")
        height, width = mask.shape
        if height == 0 or width == 0:
            return mask

        dx = rng.uniform(-1.0, 1.0, size=mask.shape).astype(np.float32)
        dy = rng.uniform(-1.0, 1.0, size=mask.shape).astype(np.float32)

        if self.elastic_sigma > 0:
            ksize = int(ceil(self.elastic_sigma * 4))
            if ksize % 2 == 0:
                ksize += 1
            ksize = max(3, ksize)
            dx = cv.GaussianBlur(dx, (ksize, ksize), self.elastic_sigma)
            dy = cv.GaussianBlur(dy, (ksize, ksize), self.elastic_sigma)

        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        map_x = grid_x + dx * self.elastic_alpha
        map_y = grid_y + dy * self.elastic_alpha

        warped = cv.remap(
            mask.astype(np.float32),
            map_x,
            map_y,
            interpolation=cv.INTER_LINEAR,
            borderMode=cv.BORDER_REFLECT_101,
        )
        return warped

    def _apply_width_jitter(
        self, mask: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        if self.width_jitter_radius <= 0:
            return mask
        radius = rng.integers(0, self.width_jitter_radius + 1)
        if radius <= 0:
            return mask
        kernel_size = 2 * radius + 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if rng.random() < 0.5:
            jittered = cv.dilate(mask, kernel)
        else:
            jittered = cv.erode(mask, kernel)
        return (jittered > 0).astype(np.uint8)

    def _apply_random_noise(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.noise_flip_prob <= 0:
            return mask
        noise = rng.random(mask.shape)
        flipped = np.where(noise < self.noise_flip_prob, 1 - mask, mask)
        return flipped.astype(np.uint8)

    def _spawn_rng(self) -> np.random.Generator:
        seed = int(self._rng.integers(0, 2**31 - 1))
        return np.random.default_rng(seed)

    def _generate_skeleton_patch_pair(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray,
        rng: np.random.Generator,
        apply_random: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.skeleton_patch_size is None:
            return mask1, mask2

        skeleton_full = self._skeletonize(mask1)
        center_y = None
        center_x = None
        patch1: Optional[np.ndarray] = None
        skel1: Optional[np.ndarray] = None

        for _ in range(self._patch_center_sample_attempts):
            candidate_y, candidate_x = self._sample_skeleton_center(skeleton_full, rng)
            candidate_patch = self._crop_patch_with_padding(
                mask1, candidate_y, candidate_x, self.skeleton_patch_size
            )
            candidate_skeleton = self._skeletonize(candidate_patch)

            if patch1 is None:
                center_y, center_x = candidate_y, candidate_x
                patch1 = candidate_patch
                skel1 = candidate_skeleton

            if np.count_nonzero(candidate_skeleton) > 0:
                center_y, center_x = candidate_y, candidate_x
                patch1 = candidate_patch
                skel1 = candidate_skeleton
                break

        if patch1 is None or skel1 is None:
            center_y = mask1.shape[0] // 2
            center_x = mask1.shape[1] // 2
            patch1 = self._crop_patch_with_padding(mask1, center_y, center_x, self.skeleton_patch_size)
            skel1 = self._skeletonize(patch1)

        patch2 = self._crop_patch_with_padding(mask2, center_y, center_x, self.skeleton_patch_size)
        skel2 = self._skeletonize(patch2)

        if apply_random and (self.skeleton_transform_rotation > 0 or self.skeleton_transform_scale > 0 or self.skeleton_transform_translation > 0):
            skel2 = self._apply_affine_to_skeleton(skel2, rng)

        if apply_random and self.skeleton_occlusion_prob > 0 and rng.random() < self.skeleton_occlusion_prob:
            target = rng.integers(0, 2)
            if target == 0:
                skel1 = self._apply_random_occlusions(skel1, rng)
            else:
                skel2 = self._apply_random_occlusions(skel2, rng)

        if self.skeleton_noise_std > 0:
            skel1 = self._jitter_skeleton_points(skel1, rng, apply_random)
            skel2 = self._jitter_skeleton_points(skel2, rng, apply_random)

        return (
            patch1.astype(np.uint8),
            patch2.astype(np.uint8),
            skel1.astype(np.uint8),
            skel2.astype(np.uint8),
        )

    def _sample_skeleton_center(
        self, skeleton: np.ndarray, rng: np.random.Generator
    ) -> Tuple[int, int]:
        coords = np.argwhere(skeleton > 0)
        if coords.size == 0:
            h, w = skeleton.shape
            return int(h // 2), int(w // 2)
        idx = int(rng.integers(0, coords.shape[0]))
        y, x = coords[idx]
        return int(y), int(x)

    def _crop_patch_with_padding(
        self, mask: np.ndarray, center_y: int, center_x: int, size: int
    ) -> np.ndarray:
        if size <= 0:
            return mask.copy()
        half = size // 2
        h, w = mask.shape[:2]
        y0 = center_y - half
        x0 = center_x - half
        y1 = y0 + size
        x1 = x0 + size

        pad_top = max(0, -y0)
        pad_left = max(0, -x0)
        pad_bottom = max(0, y1 - h)
        pad_right = max(0, x1 - w)

        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
        mask_padded = np.pad(mask, pad_width, mode="constant", constant_values=0)

        y0 += pad_top
        y1 += pad_top
        x0 += pad_left
        x1 += pad_left

        cropped = mask_padded[y0:y1, x0:x1]
        if cropped.shape[0] != size or cropped.shape[1] != size:
            cropped = cv.resize(cropped, (size, size), interpolation=cv.INTER_NEAREST)
        return (cropped > 0).astype(np.uint8)

    def _apply_affine_to_skeleton(
        self, skeleton: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        if skeleton.size == 0 or np.count_nonzero(skeleton) == 0:
            return skeleton

        h, w = skeleton.shape
        angle = rng.uniform(-self.skeleton_transform_rotation, self.skeleton_transform_rotation)
        scale = rng.uniform(1.0 - self.skeleton_transform_scale, 1.0 + self.skeleton_transform_scale)
        if self.skeleton_transform_translation <= 1.0:
            max_translation = self.skeleton_transform_translation * float(min(h, w))
        else:
            max_translation = self.skeleton_transform_translation
        if self.skeleton_transform_translation_limit is not None:
            max_translation = min(max_translation, self.skeleton_transform_translation_limit)

        tx = rng.uniform(-max_translation, max_translation)
        ty = rng.uniform(-max_translation, max_translation)

        center = (w / 2.0, h / 2.0)
        matrix = cv.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty

        warped = cv.warpAffine(
            skeleton.astype(np.float32),
            matrix,
            (w, h),
            flags=cv.INTER_NEAREST,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0,
        )
        return (warped > 0.5).astype(np.uint8)

    def _apply_random_occlusions(
        self, skeleton: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        low, high = self.skeleton_occlusion_count
        if high <= 0:
            return skeleton
        count = int(rng.integers(low, high + 1)) if high > low else int(high)
        if count <= 0:
            return skeleton

        occluded = skeleton.copy()
        h, w = skeleton.shape
        r_min, r_max = self.skeleton_occlusion_radius
        for _ in range(count):
            cy = float(rng.uniform(0, h))
            cx = float(rng.uniform(0, w))
            radius = float(rng.uniform(r_min, r_max))
            radius = float(np.clip(radius, 1.0, max(h, w)))
            cv.circle(occluded, (int(round(cx)), int(round(cy))), int(round(radius)), 0, thickness=-1)

        if np.count_nonzero(occluded) == 0:
            return skeleton
        return occluded.astype(np.uint8)

    def _jitter_skeleton_points(
        self, skeleton: np.ndarray, rng: np.random.Generator, apply_random: bool
    ) -> np.ndarray:
        if not apply_random or self.skeleton_noise_std <= 0:
            return skeleton.astype(np.uint8)
        coords = np.argwhere(skeleton > 0)
        if coords.shape[0] == 0:
            return skeleton.astype(np.uint8)
        noise = rng.normal(0.0, self.skeleton_noise_std, size=coords.shape)
        if self.skeleton_noise_limit is not None:
            noise = np.clip(noise, -self.skeleton_noise_limit, self.skeleton_noise_limit)
        jittered = coords.astype(np.float32) + noise.astype(np.float32)
        jittered = np.round(jittered).astype(int)
        h, w = skeleton.shape
        jittered[:, 0] = np.clip(jittered[:, 0], 0, h - 1)
        jittered[:, 1] = np.clip(jittered[:, 1], 0, w - 1)
        jitter_mask = np.zeros_like(skeleton, dtype=np.uint8)
        jitter_mask[jittered[:, 0], jittered[:, 1]] = 1
        if np.count_nonzero(jitter_mask) == 0:
            return skeleton.astype(np.uint8)
        return jitter_mask

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        skeleton = np.zeros_like(mask)
        element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        done = False
        current = mask.copy()
        while not done:
            eroded = cv.erode(current, element)
            temp = cv.dilate(eroded, element)
            temp = cv.subtract(current, temp)
            skeleton = cv.bitwise_or(skeleton, temp)
            current = eroded.copy()
            done = cv.countNonZero(current) == 0
        return (skeleton > 0).astype(np.uint8)

    def _estimate_tangent_map(self, skeleton: np.ndarray) -> np.ndarray:
        coords = np.argwhere(skeleton > 0)
        tangent_map = np.zeros((*skeleton.shape, 2), dtype=np.float32)
        half = self.tangent_window // 2
        for y, x in coords:
            y0 = max(0, y - half)
            y1 = min(skeleton.shape[0], y + half + 1)
            x0 = max(0, x - half)
            x1 = min(skeleton.shape[1], x + half + 1)
            patch = skeleton[y0:y1, x0:x1]
            patch_coords = np.argwhere(patch > 0)
            if patch_coords.shape[0] < 2:
                continue
            patch_coords = patch_coords.astype(np.float32)
            patch_coords[:, 0] += y0
            patch_coords[:, 1] += x0
            patch_coords -= patch_coords.mean(axis=0, keepdims=True)
            cov = patch_coords.T @ patch_coords
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, np.argmax(eigvals)]
            dx = principal[1]
            dy = principal[0]
            norm = np.hypot(dx, dy)
            if norm < 1e-6:
                continue
            tangent_map[y, x, 0] = dx / norm
            tangent_map[y, x, 1] = dy / norm
        return tangent_map

    def _estimate_branch_map(self, skeleton: np.ndarray) -> np.ndarray:
        kernel = np.array(
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
            dtype=np.uint8,
        )
        neighbor_count = cv.filter2D(skeleton.astype(np.uint8), -1, kernel, borderType=cv.BORDER_CONSTANT)
        endpoints = ((neighbor_count == 1) & (skeleton > 0)).astype(np.float32)
        junctions = ((neighbor_count >= 3) & (skeleton > 0)).astype(np.float32)
        return np.stack([endpoints, junctions], axis=-1)
