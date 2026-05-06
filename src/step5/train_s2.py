"""S2 supervised conditional training for Step 5."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.reporting import append_log_message

from .config import ResolvedStep5Config
from .dataset import (
    Step5ConditionalDataset,
    build_condition_scaler,
    build_source_batch_counts,
    build_step5_supervised_frames,
    step5_collate_fn,
    summarize_chi_augmentation_eligibility,
)
from .supervised import (
    Step5AuxHeads,
    build_optimizer_and_scheduler,
    build_s2_components_from_step1,
    compute_s2_mt_losses,
    load_step5_checkpoint_into_modules,
)


@dataclass
class S2TrainingArtifacts:
    """Outputs from supervised S2-family training."""

    tokenizer: object
    diffusion_model: torch.nn.Module
    aux_heads: Optional[Step5AuxHeads]
    checkpoint_path: Path
    last_checkpoint_path: Path
    step1_checkpoint_path: Path
    scaler: object
    history_df: pd.DataFrame
    augmentation_diag_df: pd.DataFrame
    batch_mix_counts: Dict[str, int]
    backbone_finetune_info: Dict[str, object]


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _concat_batches(parts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not parts:
        raise ValueError("No batch parts to concatenate.")
    keys = parts[0].keys()
    merged = {key: torch.cat([part[key] for part in parts], dim=0) for key in keys}
    if merged["input_ids"].shape[0] > 1:
        perm = torch.randperm(merged["input_ids"].shape[0])
        merged = {key: value[perm] for key, value in merged.items()}
    return merged


def _sample_source_batch(
    dataset: Step5ConditionalDataset,
    *,
    count: int,
    rng: np.random.Generator,
) -> Dict[str, torch.Tensor]:
    if count <= 0:
        raise ValueError(f"Requested non-positive source batch count: {count}")
    if len(dataset) == 0:
        raise ValueError("Attempted to sample from an empty dataset.")
    indices = rng.integers(0, len(dataset), size=int(count))
    samples = [dataset[int(idx)] for idx in indices.tolist()]
    return step5_collate_fn(samples)


def _build_val_loader(
    dataset: Step5ConditionalDataset,
    *,
    batch_size: int,
) -> Optional[DataLoader]:
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=step5_collate_fn,
    )


def _compute_batch_losses(
    *,
    diffusion_model,
    aux_heads: Optional[Step5AuxHeads],
    batch: Dict[str, torch.Tensor],
    mt_aux_cfg: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    outputs = diffusion_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        condition_bundle=batch["condition_bundle"],
    )
    diffusion_loss = outputs["loss"]
    total_loss = diffusion_loss
    aux_metrics = {
        "aux_total_loss": diffusion_loss.new_tensor(0.0),
        "aux_soluble_loss": diffusion_loss.new_tensor(0.0),
        "aux_chi_loss": diffusion_loss.new_tensor(0.0),
    }
    if aux_heads is not None:
        pooled = diffusion_model.backbone.get_pooled_output(
            outputs["noisy_ids"],
            outputs["timesteps"],
            batch["attention_mask"],
            condition_bundle=batch["condition_bundle"],
            condition_drop_mask=outputs["condition_drop_mask"],
            pooling="mean",
        )
        aux_metrics = compute_s2_mt_losses(
            pooled_output=pooled,
            aux_heads=aux_heads,
            soluble_target=batch["soluble_target"],
            chi_target_aux=batch["chi_goal"],
            soluble_loss_weight=float(mt_aux_cfg.get("soluble_loss_weight", 0.1)),
            chi_loss_weight=float(mt_aux_cfg.get("chi_loss_weight", 0.1)),
        )
        total_loss = diffusion_loss + aux_metrics["aux_total_loss"]
    return {
        "total_loss": total_loss,
        "diffusion_loss": diffusion_loss,
        **aux_metrics,
    }


def _evaluate_validation(
    *,
    diffusion_model,
    aux_heads: Optional[Step5AuxHeads],
    val_loaders: Dict[str, Optional[DataLoader]],
    device: str,
    mt_aux_cfg: Dict[str, float],
) -> Dict[str, float]:
    diffusion_model.eval()
    if aux_heads is not None:
        aux_heads.eval()

    totals = {
        "n_samples": 0,
        "diffusion_loss": 0.0,
        "total_loss": 0.0,
        "aux_soluble_loss": 0.0,
        "aux_chi_loss": 0.0,
    }
    with torch.no_grad():
        for loader in val_loaders.values():
            if loader is None:
                continue
            for batch in loader:
                batch = _move_batch_to_device(batch, device)
                losses = _compute_batch_losses(
                    diffusion_model=diffusion_model,
                    aux_heads=aux_heads,
                    batch=batch,
                    mt_aux_cfg=mt_aux_cfg,
                )
                batch_size = int(batch["input_ids"].shape[0])
                totals["n_samples"] += batch_size
                totals["diffusion_loss"] += float(losses["diffusion_loss"].item()) * batch_size
                totals["total_loss"] += float(losses["total_loss"].item()) * batch_size
                totals["aux_soluble_loss"] += float(losses["aux_soluble_loss"].item()) * batch_size
                totals["aux_chi_loss"] += float(losses["aux_chi_loss"].item()) * batch_size

    n_samples = max(1, int(totals["n_samples"]))
    return {
        "val_diffusion_loss": totals["diffusion_loss"] / n_samples,
        "val_total_loss": totals["total_loss"] / n_samples,
        "val_aux_soluble_loss": totals["aux_soluble_loss"] / n_samples,
        "val_aux_chi_loss": totals["aux_chi_loss"] / n_samples,
        "val_num_samples": float(totals["n_samples"]),
    }


def _resolve_checkpoint_selection_metric(s2_cfg: Dict[str, object], *, aux_heads: Optional[Step5AuxHeads]) -> str:
    metric = str(s2_cfg.get("checkpoint_selection_metric", "auto")).strip().lower()
    if metric == "auto":
        return "val_total_loss" if aux_heads is not None else "val_diffusion_loss"
    allowed = {
        "val_diffusion_loss",
        "val_total_loss",
        "val_aux_soluble_loss",
        "val_aux_chi_loss",
    }
    if metric not in allowed:
        raise ValueError(
            "Unsupported Step 5 s2.checkpoint_selection_metric="
            f"{metric!r}. Allowed values are {sorted(allowed | {'auto'})}."
        )
    if aux_heads is None and metric in {"val_total_loss", "val_aux_soluble_loss", "val_aux_chi_loss"}:
        if metric == "val_total_loss":
            return "val_diffusion_loss"
        raise ValueError(
            f"Step 5 checkpoint_selection_metric={metric!r} requires MT aux heads, "
            "but this run uses the pure S2 variant."
        )
    return metric


def _save_supervised_checkpoint(
    *,
    checkpoint_path: Path,
    diffusion_model,
    aux_heads: Optional[Step5AuxHeads],
    optimizer,
    scheduler,
    global_step: int,
    best_val_diffusion_loss: float,
    checkpoint_selection_metric: str,
    best_checkpoint_metric_value: float,
    resolved: ResolvedStep5Config,
    run_cfg: Dict[str, object],
    scaler,
    step1_checkpoint_path: Path,
    backbone_finetune_info: Dict[str, object],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": diffusion_model.state_dict(),
        "aux_state_dict": aux_heads.state_dict() if aux_heads is not None else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": int(global_step),
        "best_val_diffusion_loss": float(best_val_diffusion_loss),
        "checkpoint_selection_metric": str(checkpoint_selection_metric),
        "best_checkpoint_metric_value": float(best_checkpoint_metric_value),
        "model_size": resolved.model_size,
        "split_mode": resolved.split_mode,
        "run_name": str(run_cfg["run_name"]),
        "variant": str(run_cfg["s2"]["variant"]),
        "cfg_scale": float(run_cfg["s2"]["cfg_scale"]),
        "finetune_last_layers": backbone_finetune_info.get("finetune_last_layers"),
        "backbone_num_layers": int(backbone_finetune_info["backbone_num_layers"]),
        "backbone_finetune_mode": str(backbone_finetune_info["backbone_finetune_mode"]),
        "backbone_finetune_enabled": bool(backbone_finetune_info["backbone_finetune_enabled"]),
        "step1_checkpoint_path": str(step1_checkpoint_path),
        "condition_scaler": {
            "temperature_min": float(scaler.temperature_min),
            "temperature_max": float(scaler.temperature_max),
            "phi_min": float(scaler.phi_min),
            "phi_max": float(scaler.phi_max),
            "chi_goal_min": float(scaler.chi_goal_min),
            "chi_goal_max": float(scaler.chi_goal_max),
        },
    }
    torch.save(payload, checkpoint_path)


def _clone_module_state_dict(module: Optional[torch.nn.Module]) -> Optional[Dict[str, torch.Tensor]]:
    if module is None:
        return None
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def train_s2_supervised_run(
    *,
    resolved: ResolvedStep5Config,
    run_cfg: Dict[str, object],
    run_dirs: Dict[str, Path],
    device: str,
    pruning_callback: Optional[Callable[..., None]] = None,
    pruning_stage: str = "s2",
    skip_disk_checkpoints: bool = False,
) -> S2TrainingArtifacts:
    """Train the Step 5 conditional diffusion model for one S2-family run."""

    s2_cfg = run_cfg["s2"]
    scaler = build_condition_scaler(resolved)
    supervised_frames = build_step5_supervised_frames(resolved)
    augmentation_diag_df = summarize_chi_augmentation_eligibility(
        supervised_frames["d_chi"],
        scaler=scaler,
        chi_lookup=resolved.chi_lookup,
    )
    augmentation_diag_path = run_dirs["metrics_dir"] / "chi_target_augmentation_eligibility.csv"
    augmentation_diag_df.to_csv(augmentation_diag_path, index=False)

    tokenizer, diffusion_model, aux_heads, step1_checkpoint_path, backbone_finetune_info = build_s2_components_from_step1(
        resolved,
        device=device,
        run_cfg=run_cfg,
    )

    train_datasets = {
        "d_chi": Step5ConditionalDataset(
            supervised_frames["train_d_chi"],
            tokenizer=tokenizer,
            scaler=scaler,
            chi_lookup=resolved.chi_lookup,
            split="train",
            chi_target_augmentation_rate=float(s2_cfg["chi_target_augmentation_rate"]),
            train=True,
            random_seed=int(resolved.step5["random_seed"]),
        ),
        "d_water": Step5ConditionalDataset(
            supervised_frames["train_d_water"],
            tokenizer=tokenizer,
            scaler=scaler,
            chi_lookup=resolved.chi_lookup,
            split="train",
            chi_target_augmentation_rate=0.0,
            train=True,
            random_seed=int(resolved.step5["random_seed"]) + 17,
        ),
    }
    val_loaders = {
        "d_chi": _build_val_loader(
            Step5ConditionalDataset(
                supervised_frames["val_d_chi"],
                tokenizer=tokenizer,
                scaler=scaler,
                chi_lookup=resolved.chi_lookup,
                split="val",
                chi_target_augmentation_rate=0.0,
                train=False,
                random_seed=int(resolved.step5["random_seed"]),
            ),
            batch_size=int(s2_cfg["batch_size"]),
        ),
        "d_water": _build_val_loader(
            Step5ConditionalDataset(
                supervised_frames["val_d_water"],
                tokenizer=tokenizer,
                scaler=scaler,
                chi_lookup=resolved.chi_lookup,
                split="val",
                chi_target_augmentation_rate=0.0,
                train=False,
                random_seed=int(resolved.step5["random_seed"]) + 17,
            ),
            batch_size=int(s2_cfg["batch_size"]),
        ),
    }

    batch_mix_counts = build_source_batch_counts(
        int(s2_cfg["batch_size"]),
        dict(s2_cfg["train_batch_mix"]),
    )
    with open(run_dirs["metrics_dir"] / "train_batch_mix_resolved.json", "w", encoding="utf-8") as handle:
        json.dump(batch_mix_counts, handle, indent=2)
    with open(run_dirs["metrics_dir"] / "step1_checkpoint.json", "w", encoding="utf-8") as handle:
        json.dump({"checkpoint_path": str(step1_checkpoint_path)}, handle, indent=2)

    modules = {"diffusion_model": diffusion_model}
    if aux_heads is not None:
        modules["aux_heads"] = aux_heads
    optimizer, scheduler = build_optimizer_and_scheduler(
        modules=modules,
        learning_rate=float(s2_cfg["learning_rate"]),
        weight_decay=float(s2_cfg["weight_decay"]),
        warmup_steps=int(s2_cfg["warmup_steps"]),
        max_steps=int(s2_cfg["max_steps"]),
        warmup_schedule=str(s2_cfg["warmup_schedule"]),
        lr_schedule=str(s2_cfg["lr_schedule"]),
    )

    best_checkpoint_path = run_dirs["checkpoints_dir"] / "conditional_diffusion_best.pt"
    last_checkpoint_path = run_dirs["checkpoints_dir"] / "conditional_diffusion_last.pt"
    best_val_diffusion_loss = float("inf")
    checkpoint_selection_metric = _resolve_checkpoint_selection_metric(s2_cfg, aux_heads=aux_heads)
    best_checkpoint_metric_value = float("inf")
    patience_counter = 0
    rng = np.random.default_rng(int(resolved.step5["random_seed"]))
    history_rows: List[Dict[str, float]] = []
    running_train: List[float] = []
    best_diffusion_state: Optional[Dict[str, torch.Tensor]] = None
    best_aux_state: Optional[Dict[str, torch.Tensor]] = None
    append_log_message(
        run_dirs["run_dir"],
        (
            f"S2 train start | run={run_cfg['run_name']} variant={s2_cfg['variant']} "
            f"max_steps={int(s2_cfg['max_steps'])} val_interval={int(s2_cfg['val_check_interval_steps'])} "
            f"train_d_chi={len(supervised_frames['train_d_chi'])} train_d_water={len(supervised_frames['train_d_water'])} "
            f"val_d_chi={len(supervised_frames['val_d_chi'])} val_d_water={len(supervised_frames['val_d_water'])}"
        ),
        echo=True,
    )

    for global_step in range(1, int(s2_cfg["max_steps"]) + 1):
        diffusion_model.train()
        if aux_heads is not None:
            aux_heads.train()
        batch_parts = []
        for source_name, count in batch_mix_counts.items():
            if count <= 0:
                continue
            dataset = train_datasets[source_name]
            if len(dataset) == 0:
                continue
            batch_parts.append(_sample_source_batch(dataset, count=count, rng=rng))
        if not batch_parts:
            raise ValueError("No non-empty training sources available for S2 supervised training.")
        batch = _move_batch_to_device(_concat_batches(batch_parts), device)

        optimizer.zero_grad(set_to_none=True)
        losses = _compute_batch_losses(
            diffusion_model=diffusion_model,
            aux_heads=aux_heads,
            batch=batch,
            mt_aux_cfg=dict(s2_cfg.get("mt_aux", {})),
        )
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), float(s2_cfg["max_grad_norm"]))
        if aux_heads is not None:
            torch.nn.utils.clip_grad_norm_(aux_heads.parameters(), float(s2_cfg["max_grad_norm"]))
        optimizer.step()
        scheduler.step()

        running_train.append(float(losses["diffusion_loss"].item()))

        if global_step % int(s2_cfg["val_check_interval_steps"]) != 0 and global_step != int(s2_cfg["max_steps"]):
            continue

        val_metrics = _evaluate_validation(
            diffusion_model=diffusion_model,
            aux_heads=aux_heads,
            val_loaders=val_loaders,
            device=device,
            mt_aux_cfg=dict(s2_cfg.get("mt_aux", {})),
        )
        history_row = {
            "global_step": int(global_step),
            "train_diffusion_loss_window": float(np.mean(running_train)) if running_train else np.nan,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_eligible_augmented_draws": float(batch["augmented_to_step3_target"].sum().item()),
            "train_eligible_rows_in_window": float(batch["augmentation_eligible"].sum().item()),
            **val_metrics,
        }
        history_rows.append(history_row)
        running_train = []

        current_selection_value = float(val_metrics[checkpoint_selection_metric])
        if pruning_callback is not None:
            pruning_callback(
                stage=str(pruning_stage),
                step=int(global_step),
                value=-float(current_selection_value),
                metrics={**history_row, "pruning_metric": str(checkpoint_selection_metric)},
            )

        current_val = float(val_metrics["val_diffusion_loss"])
        improved = current_selection_value < (
            best_checkpoint_metric_value - float(s2_cfg["early_stopping_min_delta"])
        )
        if improved:
            best_val_diffusion_loss = current_val
            best_checkpoint_metric_value = current_selection_value
            patience_counter = 0
            if skip_disk_checkpoints:
                best_diffusion_state = _clone_module_state_dict(diffusion_model)
                best_aux_state = _clone_module_state_dict(aux_heads)
            else:
                _save_supervised_checkpoint(
                    checkpoint_path=best_checkpoint_path,
                    diffusion_model=diffusion_model,
                    aux_heads=aux_heads,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    best_val_diffusion_loss=best_val_diffusion_loss,
                    checkpoint_selection_metric=checkpoint_selection_metric,
                    best_checkpoint_metric_value=best_checkpoint_metric_value,
                    resolved=resolved,
                    run_cfg=run_cfg,
                    scaler=scaler,
                    step1_checkpoint_path=step1_checkpoint_path,
                    backbone_finetune_info=backbone_finetune_info,
                )
        else:
            patience_counter += 1

        append_log_message(
            run_dirs["run_dir"],
            (
                f"S2 val | run={run_cfg['run_name']} step={int(global_step)}/{int(s2_cfg['max_steps'])} "
                f"train_loss={float(history_row['train_diffusion_loss_window']):.4f} "
                f"val_diffusion={float(current_val):.4f} "
                f"{checkpoint_selection_metric}={float(current_selection_value):.4f} "
                f"best_{checkpoint_selection_metric}={float(best_checkpoint_metric_value):.4f} "
                f"improved={int(improved)} patience={int(patience_counter)}/{int(s2_cfg['early_stopping_patience_checks'])}"
            ),
            echo=True,
        )

        if patience_counter >= int(s2_cfg["early_stopping_patience_checks"]):
            append_log_message(
                run_dirs["run_dir"],
                (
                    f"S2 early stop | run={run_cfg['run_name']} "
                    f"step={int(global_step)} patience={int(patience_counter)}"
                ),
                echo=True,
            )
            break

    if not skip_disk_checkpoints:
        _save_supervised_checkpoint(
            checkpoint_path=last_checkpoint_path,
            diffusion_model=diffusion_model,
            aux_heads=aux_heads,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=(history_rows[-1]["global_step"] if history_rows else 0),
            best_val_diffusion_loss=best_val_diffusion_loss,
            checkpoint_selection_metric=checkpoint_selection_metric,
            best_checkpoint_metric_value=best_checkpoint_metric_value,
            resolved=resolved,
            run_cfg=run_cfg,
            scaler=scaler,
            step1_checkpoint_path=step1_checkpoint_path,
            backbone_finetune_info=backbone_finetune_info,
        )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(run_dirs["metrics_dir"] / "supervised_training_history.csv", index=False)
    summary = {
        "run_name": str(run_cfg["run_name"]),
        "variant": str(s2_cfg["variant"]),
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "best_checkpoint_metric_value": float(best_checkpoint_metric_value),
        "best_val_diffusion_loss": float(best_val_diffusion_loss),
        "num_validation_checks": int(len(history_df)),
        "batch_mix_counts": batch_mix_counts,
        "backbone_finetune_info": backbone_finetune_info,
        "best_checkpoint_path": (None if skip_disk_checkpoints else str(best_checkpoint_path)),
        "last_checkpoint_path": (None if skip_disk_checkpoints else str(last_checkpoint_path)),
        "disk_checkpoints_saved": bool(not skip_disk_checkpoints),
    }
    with open(run_dirs["metrics_dir"] / "supervised_training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    append_log_message(
        run_dirs["run_dir"],
        (
            f"S2 train complete | run={run_cfg['run_name']} "
            f"checkpoint_metric={checkpoint_selection_metric} "
            f"best_checkpoint_metric_value={float(best_checkpoint_metric_value):.4f} "
            f"best_val_diffusion_loss={float(best_val_diffusion_loss):.4f} "
            f"validation_checks={int(len(history_df))}"
        ),
        echo=True,
    )

    if skip_disk_checkpoints:
        if best_diffusion_state is not None:
            diffusion_model.load_state_dict(best_diffusion_state)
        if aux_heads is not None and best_aux_state is not None:
            aux_heads.load_state_dict(best_aux_state)
    elif best_checkpoint_path.exists():
        load_step5_checkpoint_into_modules(
            checkpoint_path=best_checkpoint_path,
            diffusion_model=diffusion_model,
            aux_heads=aux_heads,
            device=device,
        )

    return S2TrainingArtifacts(
        tokenizer=tokenizer,
        diffusion_model=diffusion_model,
        aux_heads=aux_heads,
        checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        step1_checkpoint_path=step1_checkpoint_path,
        scaler=scaler,
        history_df=history_df,
        augmentation_diag_df=augmentation_diag_df,
        batch_mix_counts=batch_mix_counts,
        backbone_finetune_info=backbone_finetune_info,
    )
