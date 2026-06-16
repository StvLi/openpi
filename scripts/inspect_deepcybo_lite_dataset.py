"""Inspect a DeepCybo Lite LeRobot dataset before openpi training."""

import json
from pathlib import Path

import pandas as pd
import tyro

ACTION_DIM = 16
IMAGE_KEYS = ("image_head", "image_wrist_left", "image_wrist_right")


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _feature_shape(info: dict, key: str) -> list[int] | None:
    feature = info.get("features", {}).get(key)
    if not feature:
        return None
    shape = feature.get("shape")
    return list(shape) if shape is not None else None


def _expect(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_episode(dataset_root: Path, episode_index: int, expected_fps: int | None) -> int:
    parquet_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    _expect(parquet_path.exists(), f"Missing parquet: {parquet_path}")

    episode_df = pd.read_parquet(parquet_path)
    _expect(len(episode_df) > 0, f"Episode {episode_index} has no frames")
    _expect("action" in episode_df.columns, f"{parquet_path} missing action column")
    _expect("observation.state" in episode_df.columns, f"{parquet_path} missing observation.state column")
    _expect("timestamp" in episode_df.columns, f"{parquet_path} missing timestamp column")

    action_lengths = episode_df["action"].map(len)
    state_lengths = episode_df["observation.state"].map(len)
    _expect((action_lengths == ACTION_DIM).all(), f"Episode {episode_index} has non-16D action rows")
    _expect((state_lengths == ACTION_DIM).all(), f"Episode {episode_index} has non-16D state rows")
    _expect(episode_df["timestamp"].is_monotonic_increasing, f"Episode {episode_index} timestamps are not monotonic")

    if expected_fps is not None and "frame_index" in episode_df.columns:
        expected_timestamps = episode_df["frame_index"] / expected_fps
        max_error = (episode_df["timestamp"] - expected_timestamps).abs().max()
        _expect(max_error < 1.0 / expected_fps, f"Episode {episode_index} timestamp/frame_index mismatch: {max_error}")

    for image_key in IMAGE_KEYS:
        image_dir = dataset_root / "images" / image_key / f"episode_{episode_index:06d}"
        _expect(image_dir.exists(), f"Missing image directory: {image_dir}")
        frames = sorted(image_dir.glob("frame_*.jpg"))
        _expect(
            len(frames) == len(episode_df),
            f"{image_dir} frame count {len(frames)} != parquet rows {len(episode_df)}",
        )

    return len(episode_df)


def main(dataset_root: Path, *, expected_fps: int = 30) -> None:
    dataset_root = dataset_root.expanduser().resolve()
    _expect(dataset_root.exists(), f"Dataset root does not exist: {dataset_root}")

    meta_dir = dataset_root / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"

    _expect(info_path.exists(), f"Missing {info_path}")
    _expect(episodes_path.exists(), f"Missing {episodes_path}")
    _expect(tasks_path.exists(), f"Missing {tasks_path}")

    info = _read_json(info_path)
    episodes = _read_jsonl(episodes_path)
    tasks = _read_jsonl(tasks_path)

    _expect(info.get("fps") == expected_fps, f"Expected fps={expected_fps}, got {info.get('fps')}")
    _expect(_feature_shape(info, "action") == [ACTION_DIM], "meta/info.json action feature is not [16]")
    _expect(
        _feature_shape(info, "observation.state") == [ACTION_DIM],
        "meta/info.json observation.state feature is not [16]",
    )
    _expect(len(episodes) > 0, "No episodes found")
    _expect(len(tasks) > 0, "No tasks found")

    total_frames = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        frames = _check_episode(dataset_root, episode_index, expected_fps)
        if "length" in episode:
            _expect(int(episode["length"]) == frames, f"episodes.jsonl length mismatch for episode {episode_index}")
        total_frames += frames

    if "total_frames" in info:
        _expect(int(info["total_frames"]) == total_frames, "meta/info.json total_frames mismatch")
    if "total_episodes" in info:
        _expect(int(info["total_episodes"]) == len(episodes), "meta/info.json total_episodes mismatch")

    print(f"OK: {dataset_root}")
    print(f"episodes={len(episodes)} frames={total_frames} fps={expected_fps}")
    print(f"tasks={len(tasks)} image_keys={','.join(IMAGE_KEYS)} action_dim={ACTION_DIM}")


if __name__ == "__main__":
    tyro.cli(main)
