
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import soundfile as sf


# ============================================================
# CONFIGURATION
# ============================================================

SPLITS = ["train", "validation", "test"]

FILENAME_COL = "filename"
CLASS_COL = "class"
LABEL_COL = "label"
GENERATOR_COL = "generator_model"

EXPECTED_LABELS = {
    "bonafide": 1,
    "synthetic": 0
}

EXPECTED_SR = 16000
EXPECTED_CHANNELS = 1
EXPECTED_FRAMES = 64600

SPLIT_COLUMN_CANDIDATES = [
    "split", "dataset_split", "subset", "partition"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def normalize(value):
    if pd.isna(value):
        return None
    return str(value).strip()


def normalize_class(value):
    value = normalize(value)
    return value.lower() if value is not None else None


def load_metadata(path, name):
    if not path.exists():
        print(f"ERROR: {name} metadata not found: {path}")
        return None

    df = pd.read_csv(path)

    required = {
        FILENAME_COL,
        CLASS_COL,
        LABEL_COL,
        GENERATOR_COL
    }

    missing = required - set(df.columns)

    if missing:
        print(f"ERROR: {name} metadata missing columns: {sorted(missing)}")
        print("Available columns:", df.columns.tolist())
        return None

    return df


def get_class_counts(df):
    classes = df[CLASS_COL].map(normalize_class)

    real = (classes == "bonafide").sum()
    synthetic = (classes == "synthetic").sum()
    unknown = (~classes.isin(["bonafide", "synthetic"])).sum()

    return real, synthetic, unknown


def record_counter(df, columns):
    normalized = df[columns].copy()

    for col in columns:
        normalized[col] = normalized[col].map(normalize)

    return Counter(
        map(tuple, normalized.itertuples(index=False, name=None))
    )


# ============================================================
# CLASS AND LABEL VALIDATION
# ============================================================

def check_labels(df):
    errors = []

    for _, row in df.iterrows():
        cls = normalize_class(row[CLASS_COL])
        label = row[LABEL_COL]

        if cls not in EXPECTED_LABELS:
            errors.append((row[FILENAME_COL], "Unknown class", cls))
            continue

        try:
            if pd.isna(label) or float(label) != EXPECTED_LABELS[cls]:
                errors.append(
                    (row[FILENAME_COL], "Incorrect label", label)
                )
        except (ValueError, TypeError):
            errors.append(
                (row[FILENAME_COL], "Invalid label", label)
            )

    return errors


# ============================================================
# FULL AUDIO INSPECTION
# ============================================================

def inspect_audio(df, audio_dir):
    if not audio_dir.is_dir():
        print("ERROR: Audio directory does not exist:", audio_dir)
        return {
            "missing": [],
            "extra": [],
            "format_errors": [],
            "duplicate_disk": []
        }

    filenames = df[FILENAME_COL].dropna().astype(str).tolist()

    # All audio files, including files in nested directories
    disk_files = {
        p.relative_to(audio_dir).as_posix(): p
        for p in audio_dir.rglob("*")
        if p.is_file()
    }

    csv_files = set(filenames)
    disk_names = set(disk_files)

    missing = sorted(csv_files - disk_names)
    extra = sorted(disk_names - csv_files)

    print("\nAUDIO FILE INVENTORY")
    print("-" * 65)
    print(f"Metadata filenames:       {len(filenames):,}")
    print(f"Unique metadata filenames:{len(csv_files):,}")
    print(f"Audio files on disk:      {len(disk_names):,}")
    print(f"Missing audio files:      {len(missing):,}")
    print(f"Extra audio files:        {len(extra):,}")

    if missing:
        print("\nFirst missing files:")
        for f in missing[:10]:
            print("  ", f)

    if extra:
        print("\nFirst extra files:")
        for f in extra[:10]:
            print("  ", f)

    # Inspect EVERY available metadata audio file
    format_errors = []
    checked = 0

    print("\nINSPECTING ALL AUDIO FILES...")
    print("Expected: 16000 Hz | Mono | 64600 frames")

    for filename in sorted(csv_files & disk_names):
        path = disk_files[filename]
        checked += 1

        try:
            info = sf.info(str(path))

            problems = []

            if info.samplerate != EXPECTED_SR:
                problems.append(
                    f"sample_rate={info.samplerate}"
                )

            if info.channels != EXPECTED_CHANNELS:
                problems.append(
                    f"channels={info.channels}"
                )

            if info.frames != EXPECTED_FRAMES:
                problems.append(
                    f"frames={info.frames}"
                )

            if info.frames <= 0:
                problems.append("empty_audio")

            if problems:
                format_errors.append({
                    "filename": filename,
                    "problems": problems
                })

        except Exception as e:
            format_errors.append({
                "filename": filename,
                "problems": [f"unreadable: {e}"]
            })

        if checked % 5000 == 0:
            print(f"Inspected {checked:,} audio files...")

    print(f"\nTotal files inspected: {checked:,}")
    print(f"Audio format/read errors: {len(format_errors):,}")

    if format_errors:
        print("\nFirst 20 audio problems:")
        for item in format_errors[:20]:
            print(item)

    return {
        "missing": missing,
        "extra": extra,
        "format_errors": format_errors,
        "duplicate_disk": []
    }


# ============================================================
# INDIVIDUAL SPLIT ANALYSIS
# ============================================================

def analyze_split(name, df, audio_dir):
    section(f"{name.upper()} SPLIT ANALYSIS")

    real, synthetic, unknown = get_class_counts(df)

    print(f"Total records:       {len(df):,}")
    print(f"Real / bona fide:    {real:,}")
    print(f"Synthetic:           {synthetic:,}")
    print(f"Unknown class:       {unknown:,}")

    # Generator distribution
    synthetic_df = df[
        df[CLASS_COL].map(normalize_class) == "synthetic"
    ]

    generator_counts = synthetic_df[GENERATOR_COL].value_counts(
        dropna=False
    )

    generators = set(
        synthetic_df[GENERATOR_COL].dropna().astype(str).unique()
    )

    print(f"\nNumber of generation models: {len(generators)}")

    print("\nGENERATOR-WISE SAMPLE COUNTS")
    print("-" * 65)

    for generator, count in generator_counts.items():
        print(f"{str(generator):35} {count:>10,}")

    print("\nCLASS VS LABEL")
    print(pd.crosstab(
        df[CLASS_COL],
        df[LABEL_COL],
        dropna=False
    ))

    # Label checks
    label_errors = check_labels(df)

    print(f"\nClass-label mismatches: {len(label_errors):,}")

    if label_errors:
        print("Examples:", label_errors[:10])

    # Missing metadata values
    print("\nMISSING METADATA VALUES")
    print(df.isna().sum().to_string())

    missing_generators = synthetic_df[
        synthetic_df[GENERATOR_COL].isna()
    ]

    print(
        "\nSynthetic records without generator:",
        len(missing_generators)
    )

    # Duplicate filenames
    duplicates = df[
        df[FILENAME_COL].duplicated(keep=False)
    ]

    print("\nDuplicate filenames:", len(duplicates))

    if len(duplicates):
        print(duplicates[FILENAME_COL].head(10).tolist())

    # Audio inspection
    audio_result = inspect_audio(df, audio_dir)

    return {
        "generators": generators,
        "real": real,
        "synthetic": synthetic,
        "unknown": unknown,
        "label_errors": label_errors,
        "missing_generators": len(missing_generators),
        "duplicates": len(duplicates),
        **audio_result
    }


# ============================================================
# OVERALL METADATA COMPARISON
# ============================================================

def check_overall(overall, split_dfs):
    section("OVERALL METADATA CONSISTENCY CHECK")

    combined = pd.concat(
        split_dfs.values(),
        ignore_index=True
    )

    print(f"Overall metadata rows: {len(overall):,}")
    print(f"Combined split rows:   {len(combined):,}")

    # Identify master split column
    split_col = next(
        (
            col for col in SPLIT_COLUMN_CANDIDATES
            if col in overall.columns
        ),
        None
    )

    print("\nMaster split column:", split_col or "Not found")

    if split_col:
        print("\nMASTER SPLIT DISTRIBUTION")
        print(overall[split_col].value_counts(dropna=False).to_string())

    # Duplicate checks
    overall_dup = overall[
        overall[FILENAME_COL].duplicated(keep=False)
    ]

    combined_dup = combined[
        combined[FILENAME_COL].duplicated(keep=False)
    ]

    print("\nDUPLICATE RECORD CHECK")
    print("Overall duplicates:", len(overall_dup))
    print("Combined split duplicates:", len(combined_dup))

    # Compare every shared metadata column except split
    common_cols = [
        col for col in overall.columns
        if col in combined.columns and col != split_col
    ]

    print("\nColumns being compared:")
    print(common_cols)

    overall_counter = record_counter(overall, common_cols)
    combined_counter = record_counter(combined, common_cols)

    missing = overall_counter - combined_counter
    extra = combined_counter - overall_counter

    missing_count = sum(missing.values())
    extra_count = sum(extra.values())

    print("\nOVERALL VS COMBINED SPLITS")
    print("-" * 65)
    print("Records missing from splits:", missing_count)
    print("Extra records in splits:     ", extra_count)

    # Individual master-vs-split checks
    split_checks = {}

    if split_col:
        master_split = overall[split_col].map(
            lambda x: str(x).strip().lower()
            if pd.notna(x) else None
        )

        for split in SPLITS:
            master_part = overall.loc[
                master_split == split
            ].copy()

            split_part = split_dfs[split].copy()

            master_counter = record_counter(
                master_part, common_cols
            )

            split_counter = record_counter(
                split_part, common_cols
            )

            missing_split = master_counter - split_counter
            extra_split = split_counter - master_counter

            missing_n = sum(missing_split.values())
            extra_n = sum(extra_split.values())

            split_checks[split] = (
                missing_n == 0 and extra_n == 0
            )

            print(
                f"\n{split.upper()} comparison:"
                f"\n  Missing: {missing_n}"
                f"\n  Extra:   {extra_n}"
                f"\n  Result:  {'PASS' if split_checks[split] else 'FAIL'}"
            )

    else:
        print(
            "\nNo split column found. Combined records can be "
            "compared, but individual master-to-split assignments "
            "cannot be verified."
        )

    # Final metadata result
    metadata_ok = (
        len(overall) == len(combined)
        and missing_count == 0
        and extra_count == 0
        and len(overall_dup) == 0
        and len(combined_dup) == 0
        and (not split_col or all(split_checks.values()))
    )

    print("\nOVERALL METADATA RESULT:",
          "PASS" if metadata_ok else "FAIL")

    return metadata_ok


# ============================================================
# CROSS-SPLIT FILENAME LEAKAGE CHECK
# ============================================================

def check_cross_split_duplicates(split_dfs):
    section("CROSS-SPLIT FILENAME OVERLAP")

    filename_sets = {
        split: set(
            df[FILENAME_COL].dropna().astype(str)
        )
        for split, df in split_dfs.items()
    }

    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test")
    ]

    found = False

    for first, second in pairs:
        overlap = filename_sets[first] & filename_sets[second]

        print(f"{first} ∩ {second}: {len(overlap)}")

        if overlap:
            found = True
            print("Examples:", sorted(overlap)[:10])

    if not found:
        print("\nNo filename overlap found across splits.")

    return not found


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full dataset metadata and audio audit"
    )

    parser.add_argument(
        "--data_root",
        default="final_preprocessed_data"
    )

    args = parser.parse_args()

    root = Path(args.data_root)

    section("DATASET AUDIT")
    print("Dataset location:", root.resolve())
    print("Audio inspection: ALL FILES")

    # Load overall metadata
    overall = load_metadata(
        root / "metadata.csv",
        "OVERALL"
    )

    if overall is None:
        return

    # Load all split metadata
    split_dfs = {}

    for split in SPLITS:
        df = load_metadata(
            root / split / "metadata.csv",
            split.upper()
        )

        if df is None:
            return

        split_dfs[split] = df

    # Overall summary
    section("OVERALL DATASET SUMMARY")

    real, synthetic, unknown = get_class_counts(overall)

    print(f"Total records:       {len(overall):,}")
    print(f"Real / bona fide:    {real:,}")
    print(f"Synthetic:           {synthetic:,}")
    print(f"Unknown class:       {unknown:,}")

    overall_synthetic = overall[
        overall[CLASS_COL].map(normalize_class) == "synthetic"
    ]

    print(
        "\nNumber of generation models:",
        overall_synthetic[GENERATOR_COL].nunique()
    )

    print("\nOVERALL GENERATOR DISTRIBUTION")
    print(
        overall_synthetic[GENERATOR_COL]
        .value_counts(dropna=False)
        .to_string()
    )

    # Analyze each split
    results = {}

    for split in SPLITS:
        results[split] = analyze_split(
            split,
            split_dfs[split],
            root / split / "audio"
        )

    # Generator overlap
    section("GENERATOR OVERLAP ANALYSIS")

    train_g = results["train"]["generators"]
    val_g = results["validation"]["generators"]
    test_g = results["test"]["generators"]

    print("Training generators:", len(train_g), sorted(train_g))
    print("Validation generators:", len(val_g), sorted(val_g))
    print("Testing generators:", len(test_g), sorted(test_g))

    print("\nTRAIN ∩ VALIDATION:", sorted(train_g & val_g))
    print("TRAIN ∩ TEST (SEEN):", sorted(train_g & test_g))
    print("TEST ONLY (UNSEEN):", sorted(test_g - train_g))
    print("VALIDATION ONLY:", sorted(val_g - train_g))

    # Overall metadata comparison
    metadata_ok = check_overall(overall, split_dfs)

    # Filename overlap
    no_cross_split_duplicates = check_cross_split_duplicates(
        split_dfs
    )

    # Final report
    section("FINAL AUDIT SUMMARY")

    print("Overall metadata consistency:", "PASS" if metadata_ok else "FAIL")
    print(
        "Cross-split filename check:",
        "PASS" if no_cross_split_duplicates else "FAIL"
    )

    all_ok = metadata_ok and no_cross_split_duplicates

    for split, result in results.items():
        problems = (
            result["unknown"] > 0
            or len(result["label_errors"]) > 0
            or result["missing_generators"] > 0
            or result["duplicates"] > 0
            or len(result["missing"]) > 0
            or len(result["extra"]) > 0
            or len(result["format_errors"]) > 0
        )

        print(
            f"{split.upper():12}:",
            "PROBLEMS FOUND" if problems else "PASS"
        )

        if problems:
            all_ok = False

    print("\n" + "=" * 80)

    if all_ok:
        print("AUDIT COMPLETED: ALL CHECKED TESTS PASSED")
    else:
        print("AUDIT COMPLETED: PROBLEMS FOUND - REVIEW BEFORE TRAINING")

    print("=" * 80)


if __name__ == "__main__":
    main()
