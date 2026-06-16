"""Add LeRobot image path columns to parquet episodes.

This is useful for datasets that saved image frames under ``images/`` but did not
include the corresponding Hugging Face ``Image`` columns in episode parquet files.
The source dataset is left untouched unless ``--in-place`` is passed.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import shutil
from typing import Any

import datasets
import pyarrow.parquet as pq
import tyro

DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_IMAGE_PATH = "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpg"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n")


def _is_dataset_root(path: Path) -> bool:
    return (path / "meta/info.json").exists() and (path / "data").is_dir()


def _dataset_roots(path: Path) -> list[Path]:
    if _is_dataset_root(path):
        return [path]

    roots = sorted(child for child in path.iterdir() if child.is_dir() and _is_dataset_root(child))
    if not roots:
        raise ValueError(f"No LeRobot dataset roots found under: {path}")
    return roots


def _default_output_root(input_root: Path) -> Path:
    return input_root.with_name(f"{input_root.name}_with_image_paths")


def _prepare_output_root(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {dst}. Pass --overwrite to replace it.")
        shutil.rmtree(dst)

    dst.mkdir(parents=True)
    shutil.copytree(src / "meta", dst / "meta")

    for child in src.iterdir():
        if child.name in {"data", "meta"}:
            continue
        target = dst / child.name
        if child.is_dir():
            target.symlink_to(child.resolve(), target_is_directory=True)
        else:
            shutil.copy2(child, target)

    (dst / "data").mkdir(exist_ok=True)


def _short_image_key(image_key: str) -> str:
    return image_key.rsplit(".", maxsplit=1)[-1]


def _format_template(template: str, *, image_key: str, episode_index: int, frame_index: int, episode_chunk: int) -> Path:
    return Path(
        template.format(
            image_key=image_key,
            episode_index=episode_index,
            frame_index=frame_index,
            episode_chunk=episode_chunk,
        )
    )


def _candidate_image_paths(
    dataset_root: Path,
    template: str,
    *,
    image_key: str,
    episode_index: int,
    frame_index: int,
    episode_chunk: int,
) -> Iterable[Path]:
    seen: set[Path] = set()
    image_key_variants = (image_key, _short_image_key(image_key))
    frame_names = (
        f"frame_{frame_index:06d}",
        f"{frame_index:06d}",
        str(frame_index),
    )

    for key_variant in image_key_variants:
        rel_path = _format_template(
            template,
            image_key=key_variant,
            episode_index=episode_index,
            frame_index=frame_index,
            episode_chunk=episode_chunk,
        )
        candidate = dataset_root / rel_path
        if candidate not in seen:
            seen.add(candidate)
            yield candidate

        # If the template extension or frame prefix differs from the actual files,
        # search the same image directory using common LeRobot-style names.
        image_dir = dataset_root / "images" / key_variant / f"episode_{episode_index:06d}"
        for frame_name in frame_names:
            for ext in IMAGE_EXTENSIONS:
                candidate = image_dir / f"{frame_name}{ext}"
                if candidate not in seen:
                    seen.add(candidate)
                    yield candidate


def _resolve_image_path(
    dataset_root: Path,
    template: str,
    *,
    image_key: str,
    episode_index: int,
    frame_index: int,
    episode_chunk: int,
) -> Path:
    for candidate in _candidate_image_paths(
        dataset_root,
        template,
        image_key=image_key,
        episode_index=episode_index,
        frame_index=frame_index,
        episode_chunk=episode_chunk,
    ):
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Missing image for {image_key=} {episode_index=} {frame_index=} under {dataset_root}"
    )


def _hf_feature_from_meta(feature: dict[str, Any]) -> Any:
    dtype = feature["dtype"]
    shape = tuple(feature.get("shape") or ())

    if dtype == "image":
        return datasets.Image()
    if dtype == "video":
        raise ValueError("Video features are not supported by this image-path patch script.")
    if shape in {(), (1,)}:
        return datasets.Value(dtype)
    if len(shape) == 1:
        return datasets.Sequence(feature=datasets.Value(dtype), length=shape[0])
    if len(shape) == 2:
        return datasets.Array2D(shape=shape, dtype=dtype)
    if len(shape) == 3:
        return datasets.Array3D(shape=shape, dtype=dtype)
    if len(shape) == 4:
        return datasets.Array4D(shape=shape, dtype=dtype)
    if len(shape) == 5:
        return datasets.Array5D(shape=shape, dtype=dtype)

    raise ValueError(f"Unsupported feature shape for Hugging Face datasets: {feature}")


def _features_from_info(info: dict[str, Any], parquet_columns: list[str]) -> datasets.Features:
    meta_features = dict(info.get("features") or {})
    features: dict[str, Any] = {}

    for key in parquet_columns:
        if key not in meta_features:
            raise ValueError(f"Parquet column {key!r} is missing from meta/info.json features.")
        features[key] = _hf_feature_from_meta(meta_features[key])

    for key, feature in meta_features.items():
        if feature.get("dtype") == "image" and key not in features:
            features[key] = datasets.Image()

    return datasets.Features(features)


def _episode_parquet_path(dataset_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get("data_path") or DEFAULT_DATA_PATH
    episode_chunk = episode_index // chunks_size
    return dataset_root / data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)


def _episode_indices(dataset_root: Path, info: dict[str, Any]) -> list[int]:
    episodes_path = dataset_root / "meta/episodes.jsonl"
    if episodes_path.exists():
        return [int(item["episode_index"]) for item in _read_jsonl(episodes_path)]

    return [
        int(path.stem.removeprefix("episode_"))
        for path in sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    ]


def _patch_dataset(
    dataset_root: Path,
    output_root: Path,
    *,
    in_place: bool,
    overwrite: bool,
    embed_images: bool,
    overwrite_image_columns: bool,
) -> tuple[int, int]:
    info = _read_json(dataset_root / "meta/info.json")
    image_keys = [key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "image"]
    if not image_keys:
        raise ValueError(f"No image features found in {dataset_root / 'meta/info.json'}")

    image_path_template = info.get("image_path") or DEFAULT_IMAGE_PATH
    episode_indices = _episode_indices(dataset_root, info)

    if not in_place:
        _prepare_output_root(dataset_root, output_root, overwrite=overwrite)

    total_frames = 0
    for episode_index in episode_indices:
        src_parquet = _episode_parquet_path(dataset_root, info, episode_index)
        if not src_parquet.exists():
            raise FileNotFoundError(src_parquet)

        table = pq.read_table(src_parquet)
        features = _features_from_info(info, table.column_names)
        rows = table.to_pylist()
        chunks_size = int(info.get("chunks_size", 1000))
        episode_chunk = episode_index // chunks_size

        for row_offset, row in enumerate(rows):
            frame_index = int(row.get("frame_index", row_offset))
            row_episode_index = int(row.get("episode_index", episode_index))
            for image_key in image_keys:
                if image_key in row and row[image_key] is not None and not overwrite_image_columns:
                    continue

                image_path = _resolve_image_path(
                    dataset_root,
                    image_path_template,
                    image_key=image_key,
                    episode_index=row_episode_index,
                    frame_index=frame_index,
                    episode_chunk=episode_chunk,
                )
                row[image_key] = str(image_path)

        episode_dataset = datasets.Dataset.from_list(rows, features=features)
        if embed_images:
            from lerobot.common.datasets.utils import embed_images as _embed_images

            episode_dataset = _embed_images(episode_dataset)

        dst_parquet = output_root / src_parquet.relative_to(dataset_root)
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        tmp_parquet = dst_parquet.with_suffix(".tmp.parquet")
        episode_dataset.to_parquet(tmp_parquet)
        tmp_parquet.replace(dst_parquet)
        total_frames += len(rows)

    if not in_place:
        # Preserve the source metadata as-is. The image features were already declared
        # there; this script only makes the parquet match that declaration.
        _write_json(output_root / "meta/info.json", info)

    return len(episode_indices), total_frames


def main(
    input_root: Path,
    output_root: Path | None = None,
    *,
    in_place: bool = False,
    overwrite: bool = False,
    embed_images: bool = False,
    overwrite_image_columns: bool = False,
) -> None:
    """Patch one dataset root, or every immediate dataset root under a parent directory.

    Args:
        input_root: A LeRobot dataset root, or a parent directory containing dataset roots.
        output_root: Destination root. Defaults to ``<input_root>_with_image_paths``.
        in_place: Rewrite parquet files in the input dataset. Use with care.
        overwrite: Replace an existing output root.
        embed_images: Store image bytes in parquet instead of only path references.
        overwrite_image_columns: Replace existing non-null image columns.
    """

    input_root = input_root.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    if in_place and output_root is not None:
        raise ValueError("Do not pass output_root together with --in-place.")

    dataset_roots = _dataset_roots(input_root)
    base_output_root = input_root if in_place else (output_root.expanduser().resolve() if output_root else _default_output_root(input_root))

    for dataset_root in dataset_roots:
        if in_place:
            dataset_output_root = dataset_root
        elif len(dataset_roots) == 1 and _is_dataset_root(input_root):
            dataset_output_root = base_output_root
        else:
            dataset_output_root = base_output_root / dataset_root.name

        episodes, frames = _patch_dataset(
            dataset_root,
            dataset_output_root,
            in_place=in_place,
            overwrite=overwrite,
            embed_images=embed_images,
            overwrite_image_columns=overwrite_image_columns,
        )
        print(f"OK: {dataset_root} -> {dataset_output_root}")
        print(f"episodes={episodes} frames={frames}")


if __name__ == "__main__":
    tyro.cli(main)
