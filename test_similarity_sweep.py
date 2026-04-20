"""Sweep DBSCAN eps for part2a_similarity on a single directory.

For each eps value, copies the input directory into a sweep workspace
and runs `components/part2a_similarity.py` with logical deletion. Files
flagged as duplicates are moved into `<workspace>/<eps_dir>/deleted_duplicates/`
so you can inspect each result side-by-side.

Usage (PowerShell):
    python test_similarity_sweep.py "D:\\tendency\\train\\z\\安藤サクラ\\person_clusters\\person_1"

Optional:
    --eps_values 0.20,0.25,0.30,0.35,0.40
    --min_samples 2
    --out_dir   <workspace>  (default: <parent>/sweep_<basename>)
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def count_images(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def run_sweep(input_dir: Path, eps_values, min_samples: int, out_dir: Path):
    part2a = Path(__file__).parent / "components" / "part2a_similarity.py"
    if not part2a.exists():
        print(f"[ERR] part2a not found: {part2a}")
        sys.exit(1)

    src_count = count_images(input_dir)
    print(f"[INFO] source={input_dir}  files={src_count}")
    if src_count == 0:
        print("[ERR] source directory has no images")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    basename = input_dir.name
    results = []

    for eps in eps_values:
        tag = f"eps_{eps:.2f}"
        work = out_dir / tag
        target = work / basename
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)
        print(f"\n[SWEEP] {tag}  copying to {target}")
        shutil.copytree(input_dir, target)

        cmd = [
            sys.executable,
            str(part2a),
            str(target),
            "--eps", f"{eps}",
            "--min_samples", f"{min_samples}",
        ]
        print(f"[SWEEP] running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERR] part2a failed for {tag}: returncode={e.returncode}")

        kept = count_images(target)
        deleted = count_images(work / "deleted_duplicates")
        print(f"[SWEEP] {tag}  kept={kept}  deleted={deleted}  (total={kept + deleted})")
        results.append((eps, kept, deleted))

    print("\n=== Summary ===")
    print(f"source files: {src_count}")
    print(f"{'eps':>6} | {'kept':>6} | {'deleted':>8}")
    print("-" * 28)
    for eps, kept, deleted in results:
        print(f"{eps:>6.2f} | {kept:>6} | {deleted:>8}")
    print(f"\nWorkspace: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sweep DBSCAN eps for similarity filter.")
    parser.add_argument("input_dir", type=str, help="Directory of images to deduplicate")
    parser.add_argument("--eps_values", type=str, default="0.20,0.25,0.30,0.35,0.40",
                        help="Comma-separated eps values. default=0.20,0.25,0.30,0.35,0.40")
    parser.add_argument("--min_samples", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="",
                        help="Workspace directory (default: <parent>/sweep_<basename>)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f"[ERR] not a directory: {input_dir}")
        sys.exit(1)

    eps_values = [float(x) for x in args.eps_values.split(",") if x.strip()]
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (input_dir.parent / f"sweep_{input_dir.name}")

    run_sweep(input_dir, eps_values, args.min_samples, out_dir)


if __name__ == "__main__":
    main()
