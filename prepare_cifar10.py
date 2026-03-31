import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np


URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE_NAME = "cifar-10-python.tar.gz"
USE_COMPRESSED_NPZ = True


def download_progress(block_num, block_size, total_size):
    if total_size <= 0:
        print(f"\r[Download] {block_num * block_size / (1024 * 1024):.2f} MB", end="", flush=True)
        return

    downloaded = block_num * block_size
    percent = min(downloaded / total_size * 100, 100.0)
    downloaded_mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    print(
        f"\r[Download] {percent:6.2f}% ({downloaded_mb:.2f}/{total_mb:.2f} MB)",
        end="",
        flush=True,
    )


def download_file(url: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        print(f"[Skip] Archive already exists: {save_path}")
        return

    print(f"[Download] Start: {url}")
    urllib.request.urlretrieve(url, save_path, reporthook=download_progress)
    print(f"\n[Download] Done: {save_path}")


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    target_dir = extract_to / "cifar-10-batches-py"
    if target_dir.exists():
        print(f"[Skip] Extracted folder already exists: {target_dir}")
        return

    print(f"[Extract] Start: {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        total = len(members)

        for idx, member in enumerate(members, start=1):
            tar.extract(member, path=extract_to)
            if idx == 1 or idx % 5 == 0 or idx == total:
                print(f"\r[Extract] {idx}/{total} files", end="", flush=True)

    print(f"\n[Extract] Done: {extract_to}")


def load_pickle(file_path: Path):
    import pickle

    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def reshape_images(x: np.ndarray) -> np.ndarray:
    x = x.reshape(-1, 3, 32, 32)
    x = np.transpose(x, (0, 2, 3, 1))
    return x


def save_npz(output_path: Path, x: np.ndarray, y: np.ndarray) -> None:
    print(f"[Save] Start: {output_path}")
    print(f"[Save] x shape: {x.shape}, y shape: {y.shape}")

    if USE_COMPRESSED_NPZ:
        np.savez_compressed(output_path, x=x, y=y)
    else:
        np.savez(output_path, x=x, y=y)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[Save] Done: {output_path} ({size_mb:.2f} MB)")


def build_train_npz(cifar_dir: Path, output_path: Path) -> None:
    if output_path.exists():
        print(f"[Skip] {output_path} already exists")
        return

    print("[Train] Loading training batches...")
    xs, ys = [], []
    for i in range(1, 6):
        batch_file = cifar_dir / f"data_batch_{i}"
        print(f"[Train] Reading {batch_file}")
        batch = load_pickle(batch_file)
        xs.append(batch[b"data"])
        ys.extend(batch[b"labels"])

    print("[Train] Concatenating...")
    x_train = np.concatenate(xs, axis=0)
    y_train = np.array(ys, dtype=np.int64).reshape(-1, 1)

    print("[Train] Reshaping images...")
    x_train = reshape_images(x_train)

    save_npz(output_path, x_train, y_train)


def build_test_npz(cifar_dir: Path, output_path: Path) -> None:
    if output_path.exists():
        print(f"[Skip] {output_path} already exists")
        return

    print("[Test] Reading test batch...")
    batch = load_pickle(cifar_dir / "test_batch")

    print("[Test] Reshaping images...")
    x_test = reshape_images(batch[b"data"])
    y_test = np.array(batch[b"labels"], dtype=np.int64).reshape(-1, 1)

    save_npz(output_path, x_test, y_test)


def main() -> None:
    root = Path("data") / "cifar10"
    root.mkdir(parents=True, exist_ok=True)

    archive_path = root / ARCHIVE_NAME
    extracted_dir = root / "cifar-10-batches-py"

    download_file(URL, archive_path)
    extract_archive(archive_path, root)

    build_train_npz(extracted_dir, root / "train.npz")
    build_test_npz(extracted_dir, root / "test.npz")

    print("\n[Final structure]")
    print(root)
    for item in sorted(root.iterdir()):
        print(" -", item.name)


if __name__ == "__main__":
    main()