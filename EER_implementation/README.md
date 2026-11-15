# Open-Indic-voice-clone-detector

# Equal Error Rate (EER) Implementation

The implementation calculates the **False Rejection Rate (FRR)** and **False Acceptance Rate (FAR)** at different operating points and determines the point where the two error rates are closest. The EER is then calculated from the FRR and FAR at this point.

## Files

- `eer.py` – EER calculation implementation.
- `eer.conf` – Configuration file used by `eer.py`.
- `score/` – Contains the input score files from the evaluation of the model.
- `result/` – Contains the generated EER results from the scores produced by the model.

The current `score/` directory contains:

```text
score/
├── eval_scores_using_best_dev_model.txt
├── Gujarati_scores.txt
├── Hindi_scores.txt
└── Marathi_scores.txt
```
- `eval_scores_using_best_dev_model.txt`  – Evaluation score of AASIST on ASVspoof2019 Test Dataset.
- `Gujarati_scores.txt`  – Evaluation score of AASIST on subset of IndicVoices-R and IndicSynth DataSet.
- ` Hindi_scores.txt`  – Evaluation score of AASIST on subset of IndicVoices-R and IndicSynth DataSet.
- `Marathi_scores.txt`  – Evaluation score of AASIST on subset of IndicVoices-R and IndicSynth DataSet.

## Checking the EER Implementation

1. Clone AASIST from https://github.com/clovaai/aasist.git

2. Run the model evaluation on a dataset of your choice and obtain the generated score file.

3. Clone this repository to another folder.

4. Copy the obtained score file to the `score/` folder.

3. Update the `score_file` path in `eer.conf`.

4. Run the EER implementation:

   ```bash
   python eer.py

## Dataset Source

- `ASVspoof2019 Dataset` - https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45
- `IndicSynth Dataset` - https://huggingface.co/datasets/vdivyasharma/IndicSynth
- `IndicVoices-R` - https://huggingface.co/datasets/ai4bharat/indicvoices_r
