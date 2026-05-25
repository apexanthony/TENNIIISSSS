import argparse
import csv
import os
import random
import zipfile


def extract_zip(zip_path, images_dir):
    os.makedirs(images_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        entries = [e for e in archive.infolist() if not e.is_dir()]
        for idx, entry in enumerate(entries, 1):
            name = entry.filename.replace("\\", "/")
            if not name.startswith("Dataset/"):
                continue
            rel = name[len("Dataset/") :]
            if not rel:
                continue
            out_path = os.path.join(images_dir, *rel.split("/"))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if os.path.exists(out_path) and os.path.getsize(out_path) == entry.file_size:
                continue
            with archive.open(entry) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            if idx % 1000 == 0:
                print("extracted {}/{} files".format(idx, len(entries)), flush=True)


def read_clip_rows(label_path):
    with open(label_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_samples(images_dir):
    samples = []
    games = sorted([d for d in os.listdir(images_dir) if d.lower().startswith("game")])
    for game in games:
        game_dir = os.path.join(images_dir, game)
        if not os.path.isdir(game_dir):
            continue
        clips = sorted([d for d in os.listdir(game_dir) if d.lower().startswith("clip")])
        for clip in clips:
            clip_dir = os.path.join(game_dir, clip)
            label_path = os.path.join(clip_dir, "Label.csv")
            if not os.path.exists(label_path):
                continue
            rows = read_clip_rows(label_path)
            for idx in range(2, len(rows)):
                cur = rows[idx]
                prev = rows[idx - 1]
                preprev = rows[idx - 2]
                file_name = cur.get("file name", "")
                prev_name = prev.get("file name", "")
                preprev_name = preprev.get("file name", "")
                if not file_name or not prev_name or not preprev_name:
                    continue
                samples.append(
                    {
                        "path1": "images/{}/{}/{}".format(game, clip, file_name),
                        "path2": "images/{}/{}/{}".format(game, clip, prev_name),
                        "path3": "images/{}/{}/{}".format(game, clip, preprev_name),
                        "x-coordinate": cur.get("x-coordinate", ""),
                        "y-coordinate": cur.get("y-coordinate", ""),
                        "status": cur.get("status", ""),
                        "visibility": cur.get("visibility", "0"),
                    }
                )
    return samples


def write_csv(path, samples):
    fieldnames = ["path1", "path2", "path3", "x-coordinate", "y-coordinate", "status", "visibility"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def main():
    parser = argparse.ArgumentParser(description="Prepare TrackNet raw Dataset.zip for V2 heatmap training.")
    parser.add_argument("--zip-path", default="./Dataset.zip")
    parser.add_argument("--output-dir", default="./datasets/trackNet")
    parser.add_argument("--train-rate", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    images_dir = os.path.join(args.output_dir, "images")
    if not args.skip_extract:
        print("extracting {} -> {}".format(args.zip_path, images_dir), flush=True)
        extract_zip(args.zip_path, images_dir)

    print("building labels from {}".format(images_dir), flush=True)
    samples = build_samples(images_dir)
    if not samples:
        raise RuntimeError("no samples found; check dataset layout")

    random.Random(args.seed).shuffle(samples)
    num_train = int(len(samples) * args.train_rate)
    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "labels_train.csv"), samples[:num_train])
    write_csv(os.path.join(args.output_dir, "labels_val.csv"), samples[num_train:])

    print("samples={}, train={}, val={}".format(len(samples), num_train, len(samples) - num_train), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
