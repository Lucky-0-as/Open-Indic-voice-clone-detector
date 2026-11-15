"""AASIST training / zero-shot eval on the Hindi bonafide-vs-synthetic dataset.
Put this file in the root of the cloned clovaai/aasist repo.

Continued training from the pretrained ASVspoof weights (all weights trainable):
  python train_aasist.py --data_root "<...>\\final_preprocessed_data" \
      --init_weights models/weights/AASIST.pth --lr 2e-5 --epochs 8 --warmup_steps 300 --comment cont

From scratch:
  python train_aasist.py --data_root "<...>" --lr 1e-4 --epochs 30 --comment scratch

Zero-shot baseline (no training):
  python train_aasist.py --data_root "<...>" --init_weights models/weights/AASIST.pth --eval_only --comment zeroshot
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler

from evaluation import compute_eer
from models.AASIST import Model
from utils import seed_worker

NB = 64600  # samples per clip (matches nb_samp)
SPLIT_DIR = {"train": "train", "val": "validation", "test": "test"}


def fix_len(x, rand):
    n = len(x)
    if n >= NB:
        s = np.random.randint(0, n - NB + 1) if (rand and n > NB) else 0
        return x[s:s + NB]
    return np.tile(x, NB // n + 1)[:NB]


class SpoofDS(Dataset):
    def __init__(self, df, audio_dir, train):
        self.files = df["filename"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.dir = Path(audio_dir)
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x, _ = sf.read(str(self.dir / self.files[i]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        return torch.from_numpy(fix_len(x, self.train)), self.labels[i]


def load_split(root, key):
    d = Path(root) / SPLIT_DIR[key]
    return pd.read_csv(d / "metadata.csv"), d / "audio"


@torch.no_grad()
def get_scores(model, loader, device, amp):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            _, logits = model(x)
        out.append(logits[:, 1].float().cpu().numpy())  # bonafide logit = score
    return np.concatenate(out)


def eer_pct(bona, spoof):
    return compute_eer(bona, spoof)[0] * 100


def report(df, s):
    y = df["label"].values
    pooled = eer_pct(s[y == 1], s[y == 0])
    per = {}
    for g in sorted(df.loc[y == 0, "generator_model"].unique()):
        m = (df["generator_model"].values == g) & (y == 0)
        per[g] = eer_pct(s[y == 1], s[m])
    return pooled, per


def evaluate(model, df, adir, args, device, subset=0):
    if subset and subset < len(df):
        df = df.sample(subset, random_state=0).reset_index(drop=True)
    dl = DataLoader(SpoofDS(df, adir, False), batch_size=args.batch_size * 2,
                    shuffle=False, num_workers=args.num_workers, pin_memory=True,
                    persistent_workers=args.num_workers > 0)
    s = get_scores(model, dl, device, args.amp)
    pooled, per = report(df, s)
    return pooled, per, df, s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--config", default="config/AASIST.conf")
    p.add_argument("--init_weights", default=None)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lr_min_ratio", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--subsample", type=int, default=0, help="train clips per epoch (0 = all)")
    p.add_argument("--val_subsample", type=int, default=0, help="val clips per epoch (0 = all)")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--freq_aug", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output_dir", default="runs")
    p.add_argument("--comment", default=None)
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("No GPU found.")
    with open(args.config) as f:
        cfg = json.load(f)
    model = Model(cfg["model_config"]).to(device)
    print("params:", sum(q.numel() for q in model.parameters()))
    if args.init_weights:
        model.load_state_dict(torch.load(args.init_weights, map_location=device))
        print("loaded", args.init_weights)

    out = Path(args.output_dir) / (args.comment or time.strftime("%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    tr_df, tr_dir = load_split(args.data_root, "train")
    va_df, va_dir = load_split(args.data_root, "val")
    te_df, te_dir = load_split(args.data_root, "test")

    if not args.eval_only:
        n = len(tr_df); n1 = int((tr_df["label"] == 1).sum()); n0 = n - n1
        weight = torch.tensor([n / (2 * n0), n / (2 * n1)], dtype=torch.float32, device=device)
        print(f"train n={n} bona={n1} spoof={n0}  class weights [spoof,bona]={weight.tolist()}")
        crit = nn.CrossEntropyLoss(weight=weight)

        ds = SpoofDS(tr_df, tr_dir, True)
        k = args.subsample or len(ds)
        steps_per_epoch = k // args.batch_size
        total = steps_per_epoch * args.epochs
        gen = torch.Generator(); gen.manual_seed(args.seed)
        loader = DataLoader(ds, batch_size=args.batch_size,
                            sampler=RandomSampler(ds, replacement=False, num_samples=k, generator=gen),
                            drop_last=True, num_workers=args.num_workers, pin_memory=True,
                            worker_init_fn=seed_worker, persistent_workers=args.num_workers > 0)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                               weight_decay=args.weight_decay)

        def lr_lambda(step):
            if step < args.warmup_steps:
                return (step + 1) / args.warmup_steps
            t = (step - args.warmup_steps) / max(1, total - args.warmup_steps)
            r = args.lr_min_ratio
            return r + (1 - r) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
        print(f"steps/epoch={steps_per_epoch} total={total}")

        best, log = 1e9, []
        for ep in range(args.epochs):
            model.train(); t0 = time.time(); run = 0.0; cnt = 0
            for i, (x, y) in enumerate(loader):
                x = x.to(device, non_blocking=True); y = y.to(device).long()
                with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                    _, logits = model(x, Freq_aug=args.freq_aug)
                    loss = crit(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); sched.step()
                run += loss.item() * x.size(0); cnt += x.size(0)
                if i % 200 == 0:
                    print(f"  ep{ep} step{i}/{steps_per_epoch} loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
            v_eer, v_per, _, _ = evaluate(model, va_df, va_dir, args, device, args.val_subsample)
            print(f"epoch {ep}: train_loss={run/cnt:.4f} val_EER={v_eer:.3f}% time={time.time()-t0:.0f}s")
            print("   val per-generator EER:", {g: round(e, 2) for g, e in v_per.items()})
            log.append({"epoch": ep, "train_loss": run / cnt, "val_eer": v_eer, **{f"val_{g}": e for g, e in v_per.items()}})
            pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
            torch.save(model.state_dict(), out / "last.pth")
            if v_eer < best:
                best = v_eer
                torch.save(model.state_dict(), out / "best.pth")
                print("   new best -> best.pth")
        model.load_state_dict(torch.load(out / "best.pth", map_location=device))
        print("reloaded best checkpoint for final eval")

    # final evaluation (full val + test, test touched once)
    res = {}
    for name, df, adir in [("val", va_df, va_dir), ("test", te_df, te_dir)]:
        pooled, per, d, s = evaluate(model, df, adir, args, device)
        res[name] = {"pooled_eer": pooled, "per_generator_eer": per}
        d = d.assign(score=s)
        d[["filename", "label", "generator_model", "score"]].to_csv(out / f"scores_{name}.csv", index=False)
        print(f"\n{name.upper()} pooled EER = {pooled:.3f}%")
        for g, e in per.items():
            print(f"   {g:28s} {e:6.2f}%")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("\nsaved to", out)


if __name__ == "__main__":
    main()
