"""Sanity-check the dataset before training.
Usage: python check_data.py --data_root "D:\\path\\Dataset\\final_preprocessed_data"
"""
import argparse, random
from pathlib import Path
import pandas as pd
import soundfile as sf

SPLITS = {"train": "train", "validation": "validation", "test": "test"}

p = argparse.ArgumentParser()
p.add_argument("--data_root", required=True)
p.add_argument("--deep", type=int, default=300, help="files per split to open and verify")
a = p.parse_args()
root = Path(a.data_root)
random.seed(0)
ok_all = True

for name, folder in SPLITS.items():
    csv = root / folder / "metadata.csv"
    audio = root / folder / "audio"
    if not csv.exists():
        print(f"[{name}] MISSING csv: {csv}"); ok_all = False; continue
    if not audio.is_dir():
        print(f"[{name}] MISSING audio dir: {audio}"); ok_all = False; continue
    df = pd.read_csv(csv)
    on_disk = {f.name for f in audio.iterdir() if f.is_file()}
    missing = [f for f in df["filename"] if f not in on_disk]
    extra = on_disk - set(df["filename"])
    print(f"\n[{name}] csv rows={len(df)}  wavs on disk={len(on_disk)}  "
          f"missing={len(missing)}  extra(not in csv)={len(extra)}")
    print("   class counts:", df["class"].value_counts().to_dict())
    if missing:
        ok_all = False; print("   first missing:", missing[:5])
    # deep check: open a random sample
    present = [f for f in df["filename"] if f in on_disk]
    bad = 0
    for f in random.sample(present, min(a.deep, len(present))):
        try:
            info = sf.info(str(audio / f))
            if info.frames != 64600 or info.samplerate != 16000 or info.channels != 1:
                bad += 1
                if bad <= 3:
                    print(f"   odd file {f}: frames={info.frames} sr={info.samplerate} ch={info.channels}")
        except Exception as e:
            bad += 1; print(f"   unreadable {f}: {e}")
    print(f"   deep-checked {min(a.deep, len(present))} files, problems={bad}")
    ok_all &= (bad == 0)

print("\nRESULT:", "ALL GOOD" if ok_all else "PROBLEMS FOUND - fix before training")
