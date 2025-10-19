"""
Remove duplicate images from the facial emotion dataset using file hashing.

This script scans the train, validation, and test datasets and removes duplicate
images based on their MD5 hash values. It keeps the first occurrence of each
unique image and removes subsequent duplicates.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Dataset configuration
DATA_ROOT = "Facial_emotion_images"
SPLITS = ["train", "validation", "test"]
EMOTIONS = ["happy", "neutral", "sad", "surprise"]


def calculate_file_hash(filepath: str) -> str:
    """
    Calculate MD5 hash of a file.

    Args:
        filepath: Path to the file

    Returns:
        MD5 hash as hexadecimal string
    """
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        # Read file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_duplicates(directory: str) -> Tuple[Dict[str, List[str]], int]:
    """
    Find duplicate files in a directory based on MD5 hash.

    Args:
        directory: Path to directory containing images

    Returns:
        Tuple of (hash_to_files dict, total_files_scanned)
    """
    hash_to_files = defaultdict(list)
    total_files = 0

    # Get all image files in directory
    for filepath in Path(directory).glob("*.jpg"):
        total_files += 1
        file_hash = calculate_file_hash(str(filepath))
        hash_to_files[file_hash].append(str(filepath))

    # Also check for other common image formats
    for ext in ["*.jpeg", "*.png", "*.bmp"]:
        for filepath in Path(directory).glob(ext):
            total_files += 1
            file_hash = calculate_file_hash(str(filepath))
            hash_to_files[file_hash].append(str(filepath))

    return dict(hash_to_files), total_files


def remove_duplicates_from_split(split: str, dry_run: bool = False) -> Dict:
    """
    Remove duplicate images from a specific data split.

    Args:
        split: Dataset split name ('train', 'validation', or 'test')
        dry_run: If True, only report duplicates without deleting

    Returns:
        Dictionary with statistics about duplicates found and removed
    """
    stats = {
        "split": split,
        "emotions": {},
        "total_scanned": 0,
        "total_duplicates": 0,
        "total_removed": 0
    }

    print(f"\n{'='*60}")
    print(f"Processing {split.upper()} dataset")
    print(f"{'='*60}")

    for emotion in EMOTIONS:
        emotion_dir = os.path.join(DATA_ROOT, split, emotion)

        if not os.path.exists(emotion_dir):
            print(f"Warning: Directory not found: {emotion_dir}")
            continue

        print(f"\nScanning {emotion} images...")

        # Find duplicates
        hash_to_files, total_files = find_duplicates(emotion_dir)

        # Identify duplicates (hashes with more than one file)
        duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}

        num_unique = len(hash_to_files)
        num_duplicates = sum(len(files) - 1 for files in duplicates.values())

        print(f"  Total files scanned: {total_files}")
        print(f"  Unique images: {num_unique}")
        print(f"  Duplicate images: {num_duplicates}")

        # Remove duplicates (keep first occurrence)
        removed_count = 0
        if duplicates and not dry_run:
            for file_hash, file_list in duplicates.items():
                # Keep the first file, remove the rest
                for filepath in file_list[1:]:
                    try:
                        os.remove(filepath)
                        removed_count += 1
                        print(f"    Removed: {os.path.basename(filepath)}")
                    except Exception as e:
                        print(f"    Error removing {filepath}: {e}")

        if dry_run and duplicates:
            print(f"  [DRY RUN] Would remove {num_duplicates} duplicate files")
            for file_hash, file_list in list(duplicates.items())[:3]:  # Show first 3 examples
                print(f"    Example duplicate set:")
                for filepath in file_list:
                    print(f"      - {os.path.basename(filepath)}")

        # Update statistics
        stats["emotions"][emotion] = {
            "scanned": total_files,
            "unique": num_unique,
            "duplicates": num_duplicates,
            "removed": removed_count
        }
        stats["total_scanned"] += total_files
        stats["total_duplicates"] += num_duplicates
        stats["total_removed"] += removed_count

    return stats


def print_summary(all_stats: List[Dict]):
    """
    Print summary statistics for all dataset splits.

    Args:
        all_stats: List of statistics dictionaries from each split
    """
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    grand_total_scanned = 0
    grand_total_duplicates = 0
    grand_total_removed = 0

    for stats in all_stats:
        split = stats["split"]
        print(f"{split.upper()}:")
        print(f"  Total images scanned: {stats['total_scanned']}")
        print(f"  Total duplicates found: {stats['total_duplicates']}")
        print(f"  Total duplicates removed: {stats['total_removed']}")
        print(f"  Remaining images: {stats['total_scanned'] - stats['total_removed']}")

        # Per-emotion breakdown
        print(f"  Breakdown by emotion:")
        for emotion, emotion_stats in stats["emotions"].items():
            remaining = emotion_stats["scanned"] - emotion_stats["removed"]
            print(f"    {emotion:10s}: {emotion_stats['scanned']:5d} → {remaining:5d} "
                  f"(removed {emotion_stats['removed']})")
        print()

        grand_total_scanned += stats["total_scanned"]
        grand_total_duplicates += stats["total_duplicates"]
        grand_total_removed += stats["total_removed"]

    print(f"GRAND TOTAL:")
    print(f"  Images scanned: {grand_total_scanned}")
    print(f"  Duplicates found: {grand_total_duplicates}")
    print(f"  Duplicates removed: {grand_total_removed}")
    print(f"  Remaining images: {grand_total_scanned - grand_total_removed}")
    print(f"  Duplicate rate: {100 * grand_total_duplicates / grand_total_scanned:.2f}%")


def main():
    """Main function to remove duplicates from all dataset splits."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove duplicate images from facial emotion dataset"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report duplicates without removing them"
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        help="Process only a specific split (train, validation, or test)"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt and proceed with deletion"
    )

    args = parser.parse_args()

    # Check if data directory exists
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data directory '{DATA_ROOT}' not found!")
        print(f"Current working directory: {os.getcwd()}")
        return

    print("="*60)
    print("DUPLICATE IMAGE REMOVAL TOOL")
    print("="*60)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be deleted ***\n")
    else:
        print("\n*** WARNING: This will permanently delete duplicate files! ***")
        if not args.yes:
            response = input("Continue? (yes/no): ")
            if response.lower() not in ["yes", "y"]:
                print("Aborted.")
                return
        else:
            print("Proceeding with deletion (--yes flag provided)...\n")

    # Process specified split or all splits
    splits_to_process = [args.split] if args.split else SPLITS

    all_stats = []
    for split in splits_to_process:
        stats = remove_duplicates_from_split(split, dry_run=args.dry_run)
        all_stats.append(stats)

    # Print summary
    print_summary(all_stats)

    if args.dry_run:
        print("\nDry run complete. Run without --dry-run to actually remove duplicates.")
    else:
        print("\nDuplicate removal complete!")


if __name__ == "__main__":
    main()
