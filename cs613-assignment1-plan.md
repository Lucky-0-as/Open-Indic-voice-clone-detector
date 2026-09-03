# CS 613 Assignment 1 — Five-Day Work Plan

**Project:** Cross-lingual generalization of audio deepfake detection for Indian languages
**Deadline:** Sunday 6 September 2026, 11:59 PM
**Team:** 8 members
**Today:** Tuesday 1 September

---

## The core result you are reproducing

AASIST and RawNet2 achieve sub-1% EER on ASVspoof 2019 LA, but exceed 50% EER on most Indic languages in IndicSynth without domain adaptation. Above 50% EER is worse than random guessing.

Your experiment section reproduces both halves of that: the English success and the Indic collapse.

This is a good assignment-1 result because the target numbers are published, so you know immediately whether your pipeline is correct.

---

## Start tonight (Tuesday)

Two things must happen before Wednesday morning, or the week is lost:

1. **Register for ASVspoof 2019 LA** on Edinburgh DataShare and start the download. It is tens of GB.
2. **Start streaming a small IndicSynth slice** from HuggingFace (`vdivyasharma/IndicSynth`). Do not download 4,000 hours — pull 300–500 clips from 3 languages.

Also start the IndicVoices-R download (`ai4bharat/indicvoices_r`) for bona fide audio, again only a small slice.

**Fallback if ASVspoof access stalls:** use IndieFake (arXiv:2506.19014) — 27 hours, paired real and fake, downloads in about an hour. Decide by Wednesday noon. Do not let a stalled download consume Thursday.

---

## Day-by-day

### Wednesday 2 Sept — Environment and first inference
- Clone `github.com/clovaai/aasist`. Get the environment working for everyone.
- Confirm the pretrained checkpoint loads.
- Run inference on a small ASVspoof subset. You just need scores coming out, not the final number.
- **Screenshot everything.** The report requires screenshots of all steps.

### Thursday 3 Sept — The English number
- Run full ASVspoof 2019 LA eval inference.
- Write your own EER function. Do not just trust a library — you will be asked about it in the viva.
- Compare to the published ~0.83% EER. Debug until you match.
- Report writing begins in parallel (Introduction and Related Work).

### Friday 4 Sept — Build the Indic set and break the model
- Assemble the Indic eval set: roughly 300 IndicSynth spoof clips and 300 IndicVoices-R bona fide clips, across 3 languages.
- **Normalise identically:** 16 kHz mono, same loudness target, same silence trimming, matched duration distributions. If you skip this you will measure a corpus difference, not the cross-lingual gap, and the viva will catch it.
- Run the same unmodified model. Record the EER per language.

### Saturday 5 Sept — Analysis and report
- Per-language EER breakdown. Score distribution plots for English versus Indic.
- Write Experimental Plan, Datasets, Project Management sections.
- Build the Gantt chart in matplotlib.
- Clean the GitHub repository, write the README.

### Sunday 6 Sept — Finish
- Final read-through. Verify every screenshot is present.
- Confirm the repo is public and the link in the PDF works.
- Submit with time to spare.

---

## Section-by-section guidance for the report

### 1. Introduction (10%)

**Motivation.** Voice cloning fraud is an active problem in India — scam calls using cloned voices of relatives, fabricated audio of public figures. Detection tools exist but are built and validated on English. The real-world implication is that a detector deployed in India may be no better than a coin flip. The futuristic implication is that synthesis quality keeps improving while detector coverage of Indian languages does not. If the project fails, the fallback contribution is still a documented benchmark of where existing detectors break, which is useful negative evidence.

Write this in your own words. One paragraph.

**Relation with NLP.** Be prepared to defend this — it is a speech task, and the natural question is why it belongs in an NLP course. Honest framing: the artefacts a detector picks up are partly phonetic and phonotactic. A TTS system trained mostly on English mis-renders retroflex consonants, aspirated stops, and phoneme sequences that do not occur in English. The failure is language-specific because the linguistic content is language-specific. It also sits in the same self-supervised-representation family as the rest of modern NLP.

**Problem type.** Binary classification (bona fide versus spoof), evaluated as a detection problem with a tunable threshold rather than a fixed-threshold accuracy task. Say explicitly that this is why you report EER rather than accuracy.

### 2. Related Work (10%)

The rubric asks for state of the art, available baselines, and a results table from a previous paper.

Papers to actually read (not just cite):
- **AASIST** — arXiv:2110.01200, ICASSP 2022. Your baseline. Code and pretrained weights at `github.com/clovaai/aasist`
- **RawNet2** — the other standard baseline
- **IndicSynth** — ACL 2025 Outstanding Paper, `aclanthology.org/2025.acl-long.1070`. Contains the baseline table you reproduce
- **IndieFake** — arXiv:2506.19014
- **Indic-CodecFake / SATYAM** — arXiv:2604.19949
- **Evaluating Pre-trained Speech Encoders for Indic Languages** — arXiv:2608.12536
- **SEA-Spoof** — arXiv:2509.19865
- **RADAR Challenge 2026** — arXiv:2605.09568, robustness under media transformations
- **XMAD-Bench** — arXiv:2506.00462, cross-domain multilingual

For the results table: reproduce the IndicSynth baseline table showing AASIST and RawNet2 EERs per language, and cite it properly.

**Yes, baseline implementations are available** — that is a direct rubric question and the answer is a clear yes, with the repository link.

### 3. Datasets (10%)

Answer each rubric sub-question directly.

- **Available?** Yes. IndicSynth is public on HuggingFace; ASVspoof 2019 LA requires registration; IndicVoices-R is public.
- **Instances?** State the exact counts of what you actually used, not the full dataset size. Report per-language.
- **Labelled?** Yes — labels come from provenance, not annotation. Synthetic clips are labelled spoof by construction, real recordings are bona fide.
- **Bias?** This is the important one. Note two kinds: class imbalance if your subsets are uneven, and the corpus confound — real and fake come from different sources, so a model could learn recording conditions instead of synthesis artefacts. Describe your normalisation as the mitigation. **Mentioning this unprompted will read well.**
- **Crawling?** Not needed for Assignment 1. If your final project extends to collecting real-world degraded audio, say so here.

### 4. Experimental Plan (10%)

- **Splits.** For Assignment 1 you are running inference on existing eval sets, so say that clearly. For the full project, describe speaker-disjoint and generator-disjoint splits: hold out two synthesis systems entirely so you can measure unseen-attack performance.
- **Hyperparameters.** You are not tuning in Assignment 1. For the project, describe your intended approach — grid search over learning rate and batch size, or random search, with dev-set EER as the selection criterion.
- **Metric.** EER, with justification: the threshold is a deployment policy choice, so a threshold-free metric is the right comparison. Mention min-tDCF as the secondary metric used in the ASVspoof literature. Explain why plain accuracy is misleading under class imbalance.
- **System effort and demo.** Describe the final demo concretely: a Gradio app on HuggingFace Spaces where a user uploads or records audio and gets a score with a per-segment timeline. Note the responsible-release requirement — displaying the error rate prominently and never presenting a verdict as certain.

### 5. Project Management (10%)

- **Gantt chart** in matplotlib. Include Assignment 1 as the first bar, then the project phases through to submission.
- **Compute.** State honestly what you have. A single T4 or equivalent is sufficient — the trainable part of the model is roughly 300K parameters. Estimate disk for datasets.
- **Success criteria.** Define tiers: minimum is a reproduced benchmark documenting the gap; target is a fine-tuned model that closes part of it; stretch is the robustness study plus a public demo.
- **Biggest risk.** Be honest. The strongest answer is the corpus-shortcut risk — a model that appears to work but has learned recording conditions. Mitigation: identical normalisation, plus the sanity check of training a classifier on only the first 0.1 seconds of each clip. If that beats chance, there is a shortcut.
- **Split of work.** See below.

---

## Splitting 8 people over 5 days

Do not parallelise the pipeline — there is not enough of it, and a serial dependency chain with 8 people means 7 people idle.

**Structure: four pairs, all running the same reproduction independently.**

That sounds wasteful. It is not. If all four pairs report the same EER, you have a foundation the whole team trusts. If two pairs disagree, you have found a real pipeline bug in week one, and diagnosing it will teach more than any tutorial. It also means every member can answer viva questions about the pipeline, which matters — the viva is 20% and is individual.

Layer role ownership on top:

| Members | Owns |
|---|---|
| Pair 1 | Environment setup, GitHub repo, README |
| Pair 2 | ASVspoof pipeline and the English EER |
| Pair 3 | Indic subset assembly, normalisation, the bias analysis |
| Pair 4 | EER implementation, plots, results tables |
| Two members across pairs | Report writing, starting Wednesday not Saturday |

Everyone still runs the reproduction. The roles are about who is accountable for each deliverable.

---

## Team contribution section

The rubric asks for honest relative contributions and warns that inflation hurts the group. Write it as a table: name, tasks owned, and a short factual description. Do not give everyone identical percentages if the work was not identical — TAs read that as avoidance.

---

## Viva preparation

The viva is 20% and will be individual. Every member should be able to answer:

- What does EER mean, and why is it used instead of accuracy?
- What is the model actually detecting? (Vocoder artefacts, not whether the voice sounds convincing to a human.)
- Why does the model fail on Indic languages?
- Why did you normalise loudness and duration? What breaks if you do not?
- What is the difference between a seen and an unseen attack?
- Walk through the pipeline from audio file to score.

Spend thirty minutes on Saturday going through these as a group. It is the cheapest marks in the assignment.

---

## Checklist before submitting

- [ ] PDF contains screenshots of every step
- [ ] GitHub repository link is in the PDF and the repo is public
- [ ] Both EER numbers reported: ASVspoof and Indic
- [ ] Per-language breakdown table
- [ ] Results table from IndicSynth cited properly
- [ ] Gantt chart included
- [ ] Team contribution section at the end
- [ ] All report prose written by team members
