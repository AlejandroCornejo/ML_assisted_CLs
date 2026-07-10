from pathlib import Path
import shutil


ARCHIVE_NAME = "_archived_nonbest_experiments_20260709"

GENERATED_PREFIXES = (
    "ICKAN_training_",
    "ICKAN_prediction_",
    "ICNN_training_",
    "ICNN_prediction_",
)

GENERATED_NAMES = {
    "ICNN_smoke_principal_80",
    ".tmp",
    "__pycache__",
}

KEEP_NAMES = {
    "best_ICKAN_vs_ICNN_predictions",
    ARCHIVE_NAME,
}


def main():
    root = Path(__file__).resolve().parent
    archive = root / ARCHIVE_NAME
    archive.mkdir(exist_ok=True)

    moved = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if path.name in KEEP_NAMES:
            continue
        if not (path.name.startswith(GENERATED_PREFIXES) or path.name in GENERATED_NAMES):
            continue

        destination = archive / path.name
        if destination.exists():
            raise RuntimeError(f"Archive destination already exists: {destination}")

        shutil.move(str(path), str(destination))
        moved.append(path.name)

    manifest = archive / "ARCHIVE_MANIFEST.txt"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("Archived non-best generated ICKAN/ICNN experiments.\n")
        handle.write("Nothing was deleted; directories were moved here reversibly.\n\n")
        for name in moved:
            handle.write(f"{name}\n")

    print(f"Archive directory: {archive}")
    print(f"Moved directories: {len(moved)}")
    for name in moved:
        print(name)


if __name__ == "__main__":
    main()
