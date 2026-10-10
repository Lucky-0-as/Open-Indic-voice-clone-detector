import os
import gc
import argparse
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm.auto import tqdm

# ----------------- Configuration -----------------
TARGET_SR = 16000
TARGET_SAMPLES = 64600
LANGUAGE = "Bengali"
RANDOM_SEED = 42

def aasist_pad(x, max_len=64600):
    x_len = len(x)
    if x_len >= max_len:
        return x[:max_len]
    repeats = int(np.ceil(max_len / x_len))
    return np.tile(x, repeats)[:max_len]

def preprocess_audio(filepath):
    info = sf.info(filepath)
    orig_sr = info.samplerate
    orig_dur = info.frames / orig_sr

    audio, _ = librosa.load(filepath, sr=TARGET_SR, mono=True)
    audio = aasist_pad(audio.astype(np.float32), TARGET_SAMPLES)
    return audio, orig_sr, orig_dur

def process_partition(df_subset, source_audio_dir, split_name, class_name, sub_category, config_meta, output_base, already_processed_files):
    if df_subset is None or df_subset.empty:
        return []

    out_audio_dir = os.path.join(output_base, split_name, "audio")
    os.makedirs(out_audio_dir, exist_ok=True)

    records = []
    filename_col = next((c for c in df_subset.columns if c.lower() in ["filename", "file_name"]), "file_name")

    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=f"Processing [{split_name}] [{class_name}-{sub_category}]"):
        raw_filename = str(row[filename_col])
        clean_filename = os.path.basename(raw_filename)
        
        # Checkpoint: Skip if already processed in a previous session
        if clean_filename in already_processed_files:
            continue

        in_path = os.path.join(source_audio_dir, clean_filename)
        out_path = os.path.join(out_audio_dir, clean_filename)

        if not os.path.exists(in_path):
            print(f"Warning: File not found {in_path}")
            continue

        try:
            audio, orig_sr, orig_dur = preprocess_audio(in_path)
            sf.write(out_path, audio, TARGET_SR, subtype="FLOAT")

            record = {
                "filename": clean_filename,
                "processed_path": out_path,
                "split": split_name,
                "dataset": config_meta["dataset"],
                "language": LANGUAGE,
                "label": config_meta["label"],
                "class": class_name,
                "sub_category": sub_category if sub_category else "none",
                "original_sampling_rate": orig_sr,
                "processed_sampling_rate": TARGET_SR,
                "original_duration": orig_dur,
                "processed_samples": TARGET_SAMPLES,
                "processed_duration": TARGET_SAMPLES / TARGET_SR
            }
            
            for col in df_subset.columns:
                if col not in record:
                    record[col] = row[col]

            records.append(record)
            already_processed_files.add(clean_filename)
            del audio
        except Exception as e:
            print(f"Error on {clean_filename}: {e}")
        gc.collect()

    return records

def main():
    parser = argparse.ArgumentParser(description="Resilient Audio Preprocessing Pipeline")
    parser.add_argument("--data_dir", type=str, default="./Datasets", help="Path to original Datasets directory")
    parser.add_argument("--output_dir", type=str, default="./Preprocessed", help="Path to output preprocessed directory")
    args = parser.parse_args()

    # Derived paths based on your structure: Datasets/Bonafide and Datasets/Synthetic2
    bonafide_dir = os.path.join(args.data_dir, "Bonafide")
    synthetic2_dir = os.path.join(args.data_dir, "Synthetic2")

    os.makedirs(args.output_dir, exist_ok=True)
    master_csv_path = os.path.join(args.output_dir, "metadata.csv")

    # Resume checkpointing check
    already_processed_files = set()
    existing_master_df = None
    if os.path.exists(master_csv_path):
        print(f"Found existing master metadata at {master_csv_path}. Resuming progress...")
        existing_master_df = pd.read_csv(master_csv_path)
        if "filename" in existing_master_df.columns:
            already_processed_files = set(existing_master_df["filename"].astype(str))
        print(f"Resuming: {len(already_processed_files)} files already processed.")

    all_records = []
    if existing_master_df is not None and not existing_master_df.empty:
        all_records.extend(existing_master_df.to_dict("records"))

    # ==========================================
    # 1. PROCESS BONAFIDE (5k Train, 5k Test, 1k Valid)
    # ==========================================
    bonafide_audio_dir = os.path.join(bonafide_dir, "audio")
    csv_files = [f for f in os.listdir(bonafide_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No metadata CSV found in {bonafide_dir}")
    
    bonafide_df = pd.read_csv(os.path.join(bonafide_dir, csv_files[0]))
    n_bonafide = min(11000, len(bonafide_df))
    bonafide_df = bonafide_df.sample(n=n_bonafide, random_state=RANDOM_SEED).reset_index(drop=True)

    train_n_b = min(5000, len(bonafide_df))
    test_n_b = min(5000, len(bonafide_df) - train_n_b)
    val_n_b = min(1000, len(bonafide_df) - train_n_b - test_n_b)
    
    df_train_b = bonafide_df.iloc[:train_n_b]
    df_test_b = bonafide_df.iloc[train_n_b:train_n_b+test_n_b]
    df_val_b = bonafide_df.iloc[train_n_b+test_n_b:train_n_b+test_n_b+val_n_b]

    meta_bonafide = {"dataset": "Bengali-Bonafied", "label": 1}

    all_records.extend(process_partition(df_train_b, bonafide_audio_dir, "train", "Bonafide", "", meta_bonafide, args.output_dir, already_processed_files))
    all_records.extend(process_partition(df_test_b, bonafide_audio_dir, "test", "Bonafide", "", meta_bonafide, args.output_dir, already_processed_files))
    all_records.extend(process_partition(df_val_b, bonafide_audio_dir, "validation", "Bonafide", "", meta_bonafide, args.output_dir, already_processed_files))

    # ==========================================
    # 2. PROCESS SYNTHETIC2 (17k total breakdown)
    # ==========================================
    synth_audio_dir = os.path.join(synthetic2_dir, "audio")
    synth_csv_files = [f for f in os.listdir(synthetic2_dir) if f.endswith(".csv")]
    if not synth_csv_files:
        raise FileNotFoundError(f"No metadata CSV found in {synthetic2_dir}")

    synth_df = pd.read_csv(os.path.join(synthetic2_dir, synth_csv_files[0]))
    model_col = next((c for c in synth_df.columns if c.lower() in ["generative model", "model", "generative_model"]), "Generative Model")

    synth_quotas = {
        "freevc24": {"train": 5000, "validation": 1000, "test": 0},
        "vits": {"train": 5000, "validation": 1000, "test": 0},
        "xtts": {"train": 0, "validation": 0, "test": 5000}
    }

    for model_key, quotas in synth_quotas.items():
        sub_df = synth_df[synth_df[model_col].astype(str).str.lower().str.contains(model_key)].copy()
        if sub_df.empty:
            print(f"Warning: No rows found for model filter '{model_key}' in metadata CSV.")
            continue

        total_required = quotas["train"] + quotas["validation"] + quotas["test"]
        n_sample = min(total_required, len(sub_df))
        sub_df = sub_df.sample(n=n_sample, random_state=RANDOM_SEED).reset_index(drop=True)

        meta_synth = {"dataset": f"Bengali-Synthetic-{model_key}", "label": 0}

        idx = 0
        for split_name in ["train", "validation", "test"]:
            n_needed = quotas[split_name]
            if n_needed > 0 and idx < len(sub_df):
                n = min(n_needed, len(sub_df) - idx)
                df_part = sub_df.iloc[idx:idx+n]
                all_records.extend(process_partition(df_part, synth_audio_dir, split_name, "Synthetic", model_key, meta_synth, args.output_dir, already_processed_files))
                idx += n

    # ==========================================
    # 3. SAVE METADATA FILES
    # ==========================================
    if all_records:
        master_df = pd.DataFrame(all_records)
        master_df.to_csv(master_csv_path, index=False)
        print(f"\nSaved master metadata to: {master_csv_path}")

        for split_name in ["train", "validation", "test"]:
            split_df = master_df[master_df["split"] == split_name]
            if not split_df.empty:
                split_csv_path = os.path.join(args.output_dir, split_name, "metadata.csv")
                split_df.to_csv(split_csv_path, index=False)
                print(f"Saved {split_name} metadata to: {split_csv_path}")

        print(f"\nDone! Processed {len(master_df)} total files successfully.")
    else:
        print("\nNo files were processed.")

if __name__ == "__main__":
    main()
