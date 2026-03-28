from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tarfile
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.io import read_video

try:
    from vh_mvp.data.synthetic import CONDITION_CARDINALITIES, CONDITION_KEYS
except Exception:
    CONDITION_CARDINALITIES = {
        "shape": 2,
        "color": 4,
        "dir_x": 2,
        "dir_y": 2,
        "size": 2,
        "speed": 3,
        "motion": 3,
        "background": 3,
    }
    CONDITION_KEYS = tuple(CONDITION_CARDINALITIES.keys())


VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}

SEMANTIC_VALUE_NAMES = {
    "shape": ("indoor", "outdoor"),
    "color": ("eye_detail", "lip_detail", "locomotion_apparatus", "competitive_sport"),
    "dir_x": ("no_prop", "uses_prop"),
    "dir_y": ("local_motion", "translational_motion"),
    "size": ("close_focus", "full_scene"),
    "speed": ("low", "medium", "high"),
    "motion": ("fine", "cyclic", "explosive"),
    "background": ("personal", "apparatus", "field_court"),
}

SEMANTIC_LABEL_CONDITIONS = {
    "ApplyEyeMakeup": {"shape": 0, "color": 0, "dir_x": 1, "dir_y": 0, "size": 0, "speed": 0, "motion": 0, "background": 0},
    "ApplyLipstick": {"shape": 0, "color": 1, "dir_x": 1, "dir_y": 0, "size": 0, "speed": 0, "motion": 0, "background": 0},
    "BabyCrawling": {"shape": 0, "color": 2, "dir_x": 0, "dir_y": 1, "size": 1, "speed": 1, "motion": 1, "background": 0},
    "BalanceBeam": {"shape": 0, "color": 2, "dir_x": 1, "dir_y": 1, "size": 1, "speed": 1, "motion": 1, "background": 1},
    "BandMarching": {"shape": 1, "color": 2, "dir_x": 1, "dir_y": 1, "size": 1, "speed": 1, "motion": 1, "background": 2},
    "BenchPress": {"shape": 0, "color": 2, "dir_x": 1, "dir_y": 0, "size": 1, "speed": 0, "motion": 1, "background": 1},
    "Archery": {"shape": 1, "color": 3, "dir_x": 1, "dir_y": 0, "size": 1, "speed": 1, "motion": 2, "background": 2},
    "BaseballPitch": {"shape": 1, "color": 3, "dir_x": 1, "dir_y": 0, "size": 1, "speed": 2, "motion": 2, "background": 2},
    "Basketball": {"shape": 0, "color": 3, "dir_x": 1, "dir_y": 1, "size": 1, "speed": 2, "motion": 1, "background": 2},
    "BasketballDunk": {"shape": 0, "color": 3, "dir_x": 1, "dir_y": 1, "size": 1, "speed": 2, "motion": 2, "background": 2},
}


@dataclass(frozen=True)
class VideoItem:
    video_path: Path
    label: str
    source_relpath: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UCF101 subset into folder+manifest format.")
    parser.add_argument(
        "--archive",
        type=str,
        default="data/raw/UCF101_subset.tar.gz",
        help="Path to UCF101 subset tar.gz archive.",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default="data/raw/UCF101_subset",
        help="Directory where archive should be extracted.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/real/ucf101_subset",
        help="Output root for frames and manifests.",
    )
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--condition-mode",
        type=str,
        default="hash",
        choices=("hash", "semantic"),
        help="How to construct conditions for real videos.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing frames/manifests.")
    return parser.parse_args()


def sanitize_text(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()


def ensure_extracted(archive_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists() and any(extract_dir.rglob("*")):
        return
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Archive not found: {archive_path}. Download it first from "
            "https://huggingface.co/datasets/sayakpaul/ucf101-subset"
        )
    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(extract_dir.parent)
    if not extract_dir.exists():
        raise FileNotFoundError(
            f"Expected extraction directory '{extract_dir}' was not found after extraction. "
            "Please check archive content layout."
        )


def infer_label(video_path: Path, root: Path) -> str:
    rel_parts = video_path.relative_to(root).parts
    ignore = {"train", "test", "val", "validation", "videos", "video", "ucf101_subset"}
    for part in rel_parts[:-1]:
        if part.lower() not in ignore:
            return part
    return video_path.parent.name


def collect_videos(root: Path) -> list[VideoItem]:
    videos = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    items: list[VideoItem] = []
    for path in videos:
        label = infer_label(path, root)
        items.append(VideoItem(video_path=path, label=label, source_relpath=str(path.relative_to(root))))
    return items


def map_label_to_condition(label: str) -> dict[str, int]:
    condition: dict[str, int] = {}
    for key in CONDITION_KEYS:
        cardinality = CONDITION_CARDINALITIES[key]
        digest = hashlib.sha1(f"{label}::{key}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:4], "big") % cardinality
        condition[key] = int(value)
    return condition


def semantic_condition_for_label(label: str) -> dict[str, int]:
    if label not in SEMANTIC_LABEL_CONDITIONS:
        raise KeyError(f"No semantic condition mapping defined for label '{label}'.")
    return dict(SEMANTIC_LABEL_CONDITIONS[label])


def format_semantic_condition(condition: dict[str, int]) -> str:
    lines = []
    for key in CONDITION_KEYS:
        value = int(condition[key])
        lines.append(f"{key}: {SEMANTIC_VALUE_NAMES[key][value]}")
    return "\n".join(lines)


def condition_for_label(label: str, mode: str) -> tuple[dict[str, int], str | None]:
    if mode == "semantic":
        condition = semantic_condition_for_label(label)
        return condition, format_semantic_condition(condition)
    if mode == "hash":
        return map_label_to_condition(label), None
    raise ValueError(f"Unsupported condition mode: {mode}")


def build_schema_record(mode: str) -> dict:
    if mode == "semantic":
        return {
            "mode": "semantic",
            "keys": list(CONDITION_KEYS),
            "values": {key: list(values) for key, values in SEMANTIC_VALUE_NAMES.items()},
            "label_conditions": SEMANTIC_LABEL_CONDITIONS,
        }
    return {
        "mode": "hash",
        "keys": list(CONDITION_KEYS),
        "cardinalities": CONDITION_CARDINALITIES,
    }


def sample_indices(total_frames: int, seq_len: int) -> torch.Tensor:
    if total_frames <= 0:
        return torch.zeros(seq_len, dtype=torch.long)
    if total_frames >= seq_len:
        return torch.linspace(0, total_frames - 1, steps=seq_len).round().long()
    base = torch.arange(total_frames, dtype=torch.long)
    pad = torch.full((seq_len - total_frames,), total_frames - 1, dtype=torch.long)
    return torch.cat([base, pad], dim=0)


def decode_and_resize(video_path: Path, seq_len: int, image_size: int) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities of torchvision.*")
        warnings.filterwarnings("ignore", message=".*pts_unit 'pts' gives wrong results.*")
        video, _, _ = read_video(str(video_path), pts_unit="sec", output_format="TCHW")
    if video.numel() == 0:
        raise RuntimeError(f"Decoded empty video: {video_path}")
    video = video.float() / 255.0
    frame_indices = sample_indices(video.size(0), seq_len)
    sampled = video[frame_indices]
    sampled = F.interpolate(sampled, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return sampled.clamp(0.0, 1.0)


def save_frames(frames: torch.Tensor, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(frames.size(0)):
        frame = frames[frame_idx].permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray((frame * 255.0).astype("uint8"))
        image.save(target_dir / f"{frame_idx:03d}.png")


def split_items(items: list[VideoItem], val_ratio: float, seed: int) -> tuple[list[VideoItem], list[VideoItem]]:
    rng = random.Random(seed)
    groups: dict[str, list[VideoItem]] = defaultdict(list)
    for item in items:
        groups[item.label].append(item)

    train_items: list[VideoItem] = []
    val_items: list[VideoItem] = []
    for label, group in groups.items():
        group_copy = list(group)
        rng.shuffle(group_copy)
        if len(group_copy) <= 1:
            train_items.extend(group_copy)
            continue
        val_count = max(1, int(round(len(group_copy) * val_ratio)))
        val_items.extend(group_copy[:val_count])
        train_items.extend(group_copy[val_count:])

    if not val_items and len(train_items) > 1:
        val_items.append(train_items.pop())
    return train_items, val_items


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    archive_path = Path(args.archive)
    extract_dir = Path(args.extract_dir)
    output_root = Path(args.output_root)

    ensure_extracted(archive_path, extract_dir)

    all_items = collect_videos(extract_dir)
    if not all_items:
        raise RuntimeError(f"No videos found under {extract_dir}.")

    rng = random.Random(args.seed)
    rng.shuffle(all_items)
    if args.max_samples > 0:
        all_items = all_items[: args.max_samples]

    train_items, val_items = split_items(all_items, val_ratio=args.val_ratio, seed=args.seed)
    if not train_items:
        raise RuntimeError("No train samples after split. Increase --max-samples.")
    if not val_items:
        raise RuntimeError("No val samples after split. Increase --max-samples.")

    if args.force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_to_items = {"train": train_items, "val": val_items}
    split_rows: dict[str, list[dict]] = {"train": [], "val": []}
    skipped = 0

    for split_name, items in split_to_items.items():
        for idx, item in enumerate(items):
            sample_id = f"{split_name}_{idx:05d}_{sanitize_text(item.label)}_{sanitize_text(item.video_path.stem)}"
            rel_frames_dir = Path("frames") / split_name / sample_id
            abs_frames_dir = output_root / rel_frames_dir
            try:
                frames = decode_and_resize(item.video_path, seq_len=args.seq_len, image_size=args.image_size)
                save_frames(frames, abs_frames_dir)
            except Exception as exc:
                skipped += 1
                print(f"[warn] skipped {item.video_path}: {exc}")
                continue

            condition, condition_text = condition_for_label(item.label, args.condition_mode)
            row = {
                "id": sample_id,
                "frames_dir": str(rel_frames_dir),
                "condition": condition,
                "meta": {
                    "source_video": item.source_relpath,
                    "label": item.label,
                },
            }
            if condition_text is not None:
                row["condition_text"] = condition_text
            split_rows[split_name].append(row)

    if not split_rows["train"] or not split_rows["val"]:
        raise RuntimeError(
            "Prepared dataset has empty train or val split. "
            "Try increasing --max-samples or check decode warnings."
        )

    write_jsonl(output_root / "train.jsonl", split_rows["train"])
    write_jsonl(output_root / "val.jsonl", split_rows["val"])

    label_map = {}
    for row in split_rows["train"] + split_rows["val"]:
        label = row["meta"]["label"]
        if label not in label_map:
            label_map[label] = {
                "condition": row["condition"],
                "condition_text": row.get("condition_text"),
            }
    with (output_root / "label_condition_map.json").open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, ensure_ascii=False, indent=2)
    with (output_root / "condition_schema.json").open("w", encoding="utf-8") as handle:
        json.dump(build_schema_record(args.condition_mode), handle, ensure_ascii=False, indent=2)

    print(
        f"prepared output_root={output_root} train={len(split_rows['train'])} "
        f"val={len(split_rows['val'])} skipped={skipped} labels={len(label_map)} mode={args.condition_mode}"
    )


if __name__ == "__main__":
    main()
