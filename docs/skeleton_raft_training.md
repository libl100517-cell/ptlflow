# Training `skeleton_raft` on crack skeleton data

This guide describes the end-to-end steps required to train the `skeleton_raft` model on self-supervised crack correspondence data. It builds on the dataset and loss definitions implemented under `ptlflow/data/crack_skeleton_dataset.py` and `ptlflow/models/skeleton_raft/`.

## 1. Organize crack masks

1. Place all binary crack masks inside a root directory. The dataset loader accepts Windows and POSIX paths, so you can point the entry directly to locations such as `D:/GitHub/RPMNet-master/change_dataset/train` without copying the data.【F:ptlflow/data/crack_skeleton_dataset.py†L32-L144】
2. Files are discovered recursively (configurable) and must use one of the supported suffixes (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`).【F:ptlflow/data/crack_skeleton_dataset.py†L74-L138】
3. During training the loader samples a single mask and synthesizes a forward/backward pair by applying random affine jitter, elastic (TPS/FFD-style) warps, width changes, and sparse pixel flips before skeletonizing, building distance transforms, tangents, and narrow-band masks on the fly.【F:ptlflow/data/crack_skeleton_dataset.py†L140-L275】

> **Tip:** If you already curated explicit pairs, you can still drop a `pairs.txt` file next to the masks. When present, it overrides the synthetic pairing pipeline and uses the listed paths verbatim.【F:ptlflow/data/crack_skeleton_dataset.py†L87-L132】

> **Channels:** Enable tangent or branch cues through `include_tangent_channel` / `include_branch_channel` arguments when selecting the dataset variant.【F:ptlflow/data/flow_datamodule.py†L676-L742】

## 2. Point the datamodule to your dataset

Update `datasets.yaml` so that the `crack` entry points to the absolute paths of your training and validation directories (quotes are required for Windows drive letters):
```yaml
autoflow: /path/to/autoflow
...
crack:
  train: "D:/GitHub/RPMNet-master/change_dataset/train"
  val: "D:/GitHub/RPMNet-master/change_dataset/val"
```
【F:datasets.yaml†L1-L13】

At runtime you can still override these with `--data.crack_root_dir`, `--data.crack_train_root_dir`, `--data.crack_val_root_dir`, or by adding `root=` arguments directly to the dataset specifier.【F:ptlflow/data/flow_datamodule.py†L57-L90】【F:ptlflow/data/flow_datamodule.py†L706-L752】

## 3. Choose training and validation splits

The crack datamodule wrapper exposes a single dataset ID `crack`. When it is selected for training (`--data.train_dataset crack`) the loader uses the `train` split, applies tensor conversion plus optional flips, and enables the synthetic deformation pipeline. During validation (`--data.val_dataset crack`) it switches to the `val` split, disables the random deformations, and simply encodes each mask once for deterministic evaluation.【F:ptlflow/data/flow_datamodule.py†L668-L752】【F:ptlflow/data/crack_skeleton_dataset.py†L118-L168】 Use the new `crack.train` / `crack.val` entries or the CLI overrides above to map each split to a different directory.

## 4. Launch training

`skeleton_raft` is registered in the PTLFlow CLI, so you can instantiate it directly from `train.py`. The example below trains with four skeleton channels (binary skeleton, normalized distance, and sine/cosine tangents), AdamW optimizer, 12 update iterations, and a batch size of 2:

```bash
python train.py \
  --model skeleton_raft \
  --model.input_channels 4 \
  --model.iters 12 \
  --model.use_tangent_loss true \
  --optimizer.class_path torch.optim.AdamW \
  --optimizer.init_args.lr 1e-3 \
  --optimizer.init_args.weight_decay 0.01 \
  --data.train_dataset crack \
  --data.val_dataset crack \
  --data.crack_root_dir /data/crack_skeleton \
  --data.train_batch_size 2 \
  --data.train_num_workers 4 \
  --trainer.max_epochs 20 \
  --trainer.accelerator gpu \
  --trainer.devices 1
```

When you mirror these flags inside a YAML config, list the datamodule arguments directly under `data:`—the CLI already binds
`FlowDataModule`, so adding nested `class_path`/`init_args` keys (as you might for other Lightning projects) will trigger a
`Validation failed: Group 'data' does not accept nested key ...` error.【F:configs/skeleton_raft_debug.yaml†L1-L35】

Key flags to tune:
- `--model.input_channels`: match it to the number of channels emitted by `CrackSkeletonDataset` (2 for `[S, D]`, 4 for `[S, D, Tsin, Tcos]`, 6 if branch cues are appended).【F:ptlflow/data/crack_skeleton_dataset.py†L102-L156】
- `--model.iters`: number of RAFT refinements per forward pass.【F:ptlflow/models/skeleton_raft/skeleton_raft.py†L333-L451】
- `--optimizer.*`: optimizer configuration; the self-supervised recipe in the proposal suggests AdamW with `lr=1e-3` and a small batch size.
- `--trainer.*`: Lightning trainer options such as epochs, precision, gradient clipping, etc.【F:train.py†L15-L124】

## 5. Monitoring losses

The custom loss function logs per-component scalars (`loss`, `primary`, `tangent`, `cycle`, `smooth_parallel`, `smooth_perpendicular`). These metrics are available in the Lightning logs and checkpoints via the callbacks configured in `train.py`.【F:ptlflow/models/skeleton_raft/skeleton_raft.py†L36-L329】【F:train.py†L38-L124】

## 6. Running inference or validation only

Once a checkpoint is available, you can evaluate it with the same CLI:

```bash
python train.py \
  --model skeleton_raft \
  --model.input_channels 4 \
  --ckpt_path path/to/checkpoint.ckpt \
  --data.val_dataset crack \
  --data.crack_root_dir /data/crack_skeleton \
  --trainer.max_epochs 1 \
  --trainer.limit_train_batches 0 \
  --trainer.limit_val_batches 1.0
```

Set `--trainer.limit_train_batches 0` to skip further optimization when only validating. Adjust `--data.predict_dataset` or `--data.test_dataset` analogously for other stages.

### Debugging from VS Code

If you prefer launching training from VS Code's debugger, load the ready-made configuration file at `configs/skeleton_raft_debug.yaml`. It mirrors the command-line example above and already references the Windows dataset paths for both splits. Point your VS Code debug configuration at `train.py` and pass `--config configs/skeleton_raft_debug.yaml` as the program arguments to reproduce the same run without manually retyping flags.【F:configs/skeleton_raft_debug.yaml†L1-L35】【F:ptlflow/data/flow_datamodule.py†L668-L752】

With these steps you can fine-tune the skeleton-aware RAFT on your crack imagery and iterate on loss or augmentation choices without editing the core code.
