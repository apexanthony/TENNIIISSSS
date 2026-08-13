import argparse
import json
import re
from pathlib import Path

import pandas as pd


def natural_key(value):
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", str(value))]


def read_clip(clip_dir, game, clip):
    label_path = clip_dir / "Label.csv"
    if not label_path.exists():
        raise FileNotFoundError(f"missing label file: {label_path}")
    labels = pd.read_csv(label_path)
    expected = {"file name", "visibility", "x-coordinate", "y-coordinate", "status"}
    missing = expected.difference(labels.columns)
    if missing:
        raise ValueError(f"{label_path} missing columns: {sorted(missing)}")

    labels = labels.reset_index(drop=True)
    rows = []
    for index in range(2, len(labels)):
        current = labels.iloc[index]
        previous = labels.iloc[index - 1]
        preprevious = labels.iloc[index - 2]
        frame_name = str(current["file name"])
        paths = [clip_dir / str(item["file name"]) for item in (current, previous, preprevious)]
        for image_path in paths:
            if not image_path.exists():
                raise FileNotFoundError(f"manifest source image missing: {image_path}")
        try:
            frame_id = int(Path(frame_name).stem)
        except ValueError:
            frame_id = index
        rows.append(
            {
                "path1": str(paths[0].relative_to(clip_dir.parents[1])).replace("\\", "/"),
                "path2": str(paths[1].relative_to(clip_dir.parents[1])).replace("\\", "/"),
                "path3": str(paths[2].relative_to(clip_dir.parents[1])).replace("\\", "/"),
                "x-coordinate": current["x-coordinate"],
                "y-coordinate": current["y-coordinate"],
                "status": current["status"],
                "visibility": current["visibility"],
                "game": game,
                "clip": clip,
                "frame_name": frame_name,
                "frame_id": frame_id,
                "clip_key": f"{game}/{clip}",
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build the paper match-level TrackNet split.")
    parser.add_argument("--tracknet-root", default="datasets/trackNet")
    parser.add_argument("--out-dir", default="datasets/tracknet_v38_match_split")
    args = parser.parse_args()

    root = Path(args.tracknet_root)
    out_dir = Path(args.out_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    validation_keys = set()
    clip_records = []
    for game_index in range(1, 11):
        game = f"game{game_index}"
        game_dir = root / game
        clips = sorted([path for path in game_dir.iterdir() if path.is_dir()], key=lambda p: natural_key(p.name))
        if not clips:
            raise RuntimeError(f"no clips found under {game_dir}")
        if game_index <= 7:
            validation_keys.add((game, clips[-1].name))
        for clip_dir in clips:
            rows = read_clip(clip_dir, game, clip_dir.name)
            all_rows.extend(rows)
            clip_records.append({"game": game, "clip": clip_dir.name, "samples": len(rows)})

    frame = pd.DataFrame(all_rows)
    is_test = frame["game"].isin(["game8", "game9", "game10"])
    is_valid = pd.Series(False, index=frame.index)
    for game, clip in validation_keys:
        is_valid |= (frame["game"] == game) & (frame["clip"] == clip)
    is_train = ~is_test & ~is_valid
    masks = {"train": is_train, "valid": is_valid, "test": is_test}

    summary = {
        "protocol": "games1-7 development; last clip of each development game for validation; games8-10 test",
        "tracknet_root": str(root.resolve()),
        "path_order": "path1=t, path2=t-1, path3=t-2",
        "validation_clips": [f"{game}/{clip}" for game, clip in sorted(validation_keys, key=lambda x: natural_key(x[0]))],
        "splits": {},
    }
    for split, mask in masks.items():
        subset = frame[mask].copy()
        subset["split"] = split
        subset.to_csv(out_dir / f"{split}.csv", index=False)
        visible = int((pd.to_numeric(subset["visibility"], errors="coerce").fillna(0) != 0).sum())
        summary["splits"][split] = {
            "samples": int(len(subset)),
            "clips": int(subset["clip_key"].nunique()),
            "visible": visible,
            "invisible": int(len(subset) - visible),
            "games": sorted(subset["game"].unique().tolist(), key=natural_key),
        }
    pd.DataFrame(clip_records).to_csv(out_dir / "clips.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
