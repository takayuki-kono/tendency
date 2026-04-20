"""Apply components/part2a_similarity.py to all master_data person_clusters.

Iterates over master_data/<category>/<person>/person_clusters/person_* and
runs the similarity filter with a fixed eps (default 0.25, physical delete).
"""
import argparse
import functools
import subprocess
import sys
from pathlib import Path

print = functools.partial(print, flush=True)


IMG_EXT = {".jpg", ".jpeg", ".png"}


def count_images(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


def collect_targets(root: Path):
    targets = []
    if not root.is_dir():
        return targets
    for cat in sorted(root.iterdir()):
        if not cat.is_dir():
            continue
        for person in sorted(cat.iterdir()):
            pc = person / "person_clusters"
            if not pc.is_dir():
                continue
            for sub in sorted(pc.iterdir()):
                if sub.is_dir() and sub.name.startswith("person_"):
                    targets.append(sub)
    return targets


def main():
    parser = argparse.ArgumentParser(description="Batch apply part2a_similarity to master_data.")
    parser.add_argument("--root", type=str, default="master_data")
    parser.add_argument("--eps", type=float, default=0.25)
    parser.add_argument("--min_samples", type=int, default=2)
    parser.add_argument("--physical_delete", action="store_true",
                        help="Enable physical deletion (recommended for master_data).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only list targets without executing.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    part2a = Path(__file__).parent / "components" / "part2a_similarity.py"
    if not part2a.exists():
        print(f"[ERR] part2a not found: {part2a}")
        sys.exit(1)

    targets = collect_targets(root)
    if not targets:
        print(f"[WARN] no targets under {root}")
        sys.exit(0)

    print(f"[INFO] root={root}")
    print(f"[INFO] eps={args.eps}  min_samples={args.min_samples}  physical_delete={args.physical_delete}")
    print(f"[INFO] {len(targets)} target(s) found:")
    before_counts = []
    total_before = 0
    for t in targets:
        c = count_images(t)
        total_before += c
        before_counts.append(c)
        print(f"  - {t}  files={c}")
    print(f"[INFO] total images before: {total_before}")

    if args.dry_run:
        print("[INFO] dry_run: exiting before executing.")
        return

    results = []
    for t, before in zip(targets, before_counts):
        print(f"\n[RUN] {t}  (before={before})")
        cmd = [
            sys.executable,
            str(part2a),
            str(t),
            "--eps", f"{args.eps}",
            "--min_samples", f"{args.min_samples}",
        ]
        if args.physical_delete:
            cmd.append("--physical_delete")
        print(f"[RUN] cmd: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERR] returncode={e.returncode} for {t}")
            results.append((t, before, None))
            continue
        after = count_images(t)
        deleted_dir = t / "deleted_duplicates"
        logical_deleted = count_images(deleted_dir) if deleted_dir.exists() else 0
        physical_deleted = max(0, before - after - logical_deleted)
        results.append((t, before, after, logical_deleted, physical_deleted))
        print(f"[RUN] {t}  before={before}  after={after}  "
              f"logical_moved={logical_deleted}  physical_deleted={physical_deleted}")

    print("\n=== Summary ===")
    print(f"{'dir':<80} {'before':>7} {'after':>7} {'moved':>7} {'phys':>7}")
    total_after = 0
    total_moved = 0
    total_phys = 0
    for r in results:
        if len(r) == 3:
            t, before, _ = r
            print(f"{str(t):<80} {before:>7}  ERROR")
            continue
        t, before, after, moved, phys = r
        total_after += after
        total_moved += moved
        total_phys += phys
        print(f"{str(t):<80} {before:>7} {after:>7} {moved:>7} {phys:>7}")
    print(f"{'TOTAL':<80} {total_before:>7} {total_after:>7} {total_moved:>7} {total_phys:>7}")


if __name__ == "__main__":
    main()
