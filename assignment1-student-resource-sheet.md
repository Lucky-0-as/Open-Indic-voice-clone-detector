# Assignment 1 — Resource Sheet

**Scope agreed with mentor:** 3 IndicSynth languages, ~300 spoof + ~300 bona fide clips per language, AASIST pretrained checkpoint, inference only. No training. RawNet2 optional if time allows.

**Deadline:** Sunday 6 September, 11:59 PM.

---

## Do these two things today

1. **Register for ASVspoof 2019 LA** on Edinburgh DataShare and start the download. Tens of GB — this is the only task that cannot be sped up later.
2. **Start pulling a small IndicSynth slice.** Do not download all 4,000 hours.

**Fallback:** if ASVspoof access is not working by Wednesday noon, switch to IndieFake and tell your mentor. Do not spend Thursday waiting on a download.

---

## Code

**AASIST** — `github.com/clovaai/aasist`
Includes pretrained weights. Paper: arXiv:2110.01200 (ICASSP 2022). Target: roughly 0.83% EER on ASVspoof 2019 LA eval.

**RawNet2** — arXiv:2011.01108. Optional second baseline.

---

## Datasets

| Dataset | Link | Use |
|---|---|---|
| **IndicSynth** | `huggingface.co/datasets/vdivyasharma/IndicSynth` | Spoof audio. 12 languages. Synthetic only — you must source real audio separately |
| **IndicVoices-R** | `huggingface.co/datasets/ai4bharat/indicvoices_r` | Bona fide audio to pair with IndicSynth |
| **ASVspoof 2019 LA** | Edinburgh DataShare (registration required) | The English reproduction target |
| **IndieFake** | arXiv:2506.19014 | Fallback. Paired real + fake, 27 hours, Indian English speakers |

---

## Papers

**Read properly before writing code (Wednesday):**
- **AASIST** — arXiv:2110.01200. Focus on Section 3 and the results table
- **IndicSynth** — `aclanthology.org/2025.acl-long.1070` (ACL 2025 Outstanding Paper). Section 3 and the baseline table you are reproducing

**One per pair, for Related Work (Thursday):**
- RawNet2 — arXiv:2011.01108
- ASVspoof 2021 challenge summary — for EER and min-tDCF definitions
- IndieFake — arXiv:2506.19014
- Pre-trained Speech Encoders for Indic Languages — arXiv:2608.12536

**Skim abstracts, cite only:**
- Indic-CodecFake / SATYAM — arXiv:2604.19949
- SEA-Spoof — arXiv:2509.19865
- XMAD-Bench — arXiv:2506.00462
- RADAR Challenge 2026 — arXiv:2605.09568
- Does Audio Deepfake Detection Generalize? — arXiv:2203.16263

---

## The result you are reproducing

AASIST and RawNet2 achieve sub-1% EER on ASVspoof 2019 LA but exceed 50% EER on most Indic languages in IndicSynth without domain adaptation. Above 50% is worse than random guessing.

Reproduce the English success first. If you cannot hit the published English number, your pipeline is broken and the Indic number means nothing.

---

## Normalisation — do not skip this

Your real audio and fake audio come from different corpora with different recording conditions. A model can score well by learning "which dataset is this" instead of "is this synthetic." Before running anything:

```bash
# 16 kHz, mono, loudness normalised — apply identically to BOTH classes
ffmpeg -i in.wav -af loudnorm=I=-23:LRA=7:TP=-2 -ar 16000 -ac 1 out.wav
```

Also match duration distributions and trim silence the same way on both classes. Document that you did this in the report — it is the difference between a valid result and a meaningless one.

**Sanity check if you have time:** train a classifier on only the first 0.1 seconds of each clip. If it beats chance, you have a shortcut somewhere.

---

## EER

Write this yourself. You will be asked about it in the viva.

```python
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
```

Be ready to explain *why* EER rather than accuracy: the threshold is a deployment policy choice, so the metric should not depend on where it is set. Accuracy is also misleading under class imbalance.

---

## Screenshots

The submission requires screenshots of every step. **Capture as you go, starting Wednesday morning** — environment setup, download, inference, EER computation, plots. Do not plan to re-run everything on Sunday to generate them.

---

## Reproducibility

Put in the repo:
- `requirements.txt` or environment file
- Fixed random seeds
- Exact commands to reproduce every number in the report, in the README

This is cheap for you and makes the viva much easier.

---

## Viva questions to prepare

The viva is 20% and individual. Every member should be able to answer:

- What does EER mean and why use it instead of accuracy?
- What is the model actually detecting? (Vocoder artefacts — not whether the voice sounds convincing to a human.)
- **Why does the model fail on Indic languages?** Think about phoneme inventories, what the TTS systems were trained on, and where synthesis artefacts live in the signal.
- Why did you normalise loudness and duration? What breaks if you do not?
- What is the difference between a seen and an unseen attack?
- Walk through the pipeline from audio file to score.

Spend thirty minutes on Saturday going through these as a group.

---

## Day-by-day

| Day | Work |
|---|---|
| **Tue (today)** | Start downloads. Assign pairs. Two people begin the Introduction and Related Work sections |
| **Wed** | Environment working for everyone. Read the two Tier-1 papers. First inference run producing scores |
| **Thu** | Full ASVspoof eval. Write and verify the EER function. Hit the published number. Related Work drafted |
| **Fri** | Build and normalise the Indic subset. Run the same model. Record per-language EER |
| **Sat** | Analysis, plots, per-language table. Gantt chart. Remaining report sections. Viva practice |
| **Sun** | Final read, screenshot check, repo public, submit early |

---

## Team split

Four pairs. **Every pair runs the reproduction independently** — yes, redundantly. If all four get the same EER you have a foundation you can trust. If two disagree, you have found a real bug on day two instead of day five. It also means everyone can answer viva questions.

On top of that, assign ownership:

| Pair | Accountable for |
|---|---|
| 1 | Environment, GitHub repo, README, reproducibility |
| 2 | ASVspoof pipeline and the English EER |
| 3 | Indic subset assembly, normalisation, bias analysis |
| 4 | EER implementation, plots, results tables |

Two members across pairs own report writing, **starting Wednesday**.

---

## Marks reality check

Report 50%, experiment 20%, findings 10%, viva 20%. The report is worth as much as everything else combined. A perfect EER with three rushed paragraphs written on Sunday night scores worse than a partial result written up well.

Write the report in your own words, from your own reading and your own numbers.
