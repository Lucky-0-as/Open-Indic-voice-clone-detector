# Open-Indic-voice-clone-detector

# Equal Error Rate (EER) Implementation

The implementation calculates the **False Negative Rate (FNR)** and **False Positive Rate (FPR)** at different operating points and determines the point where the two error rates are closest. The EER is then calculated from the FNR and FPR at this point.

## Files

- `eerOnIndicSynth.py` – EER calculation implementation for the IndicSynth score files.
- `eerOnIndicSynth.conf` – Configuration file used by `eerOnIndicSynth.py`.
- `score/` – Contains the input score files from the evaluation of the model on IndicSynth.
- `result/` – Contains the generated EER results from the scores produced by the model.

The current `score/` directory contains:

```text
score/
├── Gujarati_scores.txt
├── Hindi_scores.txt
└── Marathi_scores.txt
```

## Checking the EER Implementation

1. Clone this repository.

2. Run the model evaluation on a dataset of your choice and obtain the generated score file.

3. Update the `score_file` path in `eerOnIndicSynth.conf`.

4. Run the EER implementation:

   ```bash
   python eerOnIndicSynth.py