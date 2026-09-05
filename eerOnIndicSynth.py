import configparser
import os


# ---------------------------------------------------------
# 1. Load configuration file
# ---------------------------------------------------------

config = configparser.ConfigParser()
config.read("./eerOnIndicSynth.conf")


# ---------------------------------------------------------
# 2. Get file paths from the config
# ---------------------------------------------------------

score_file = config["PATHS"]["score_file"]
output_dir = config["PATHS"]["output_file"]

source_file_name = os.path.splitext(os.path.basename(score_file))[0]
output_file = os.path.join(output_dir, source_file_name + "_EER.txt")


# ---------------------------------------------------------
# 3. Read the score file
# ---------------------------------------------------------

with open(score_file, "r") as file:
    lines = file.readlines()


# ---------------------------------------------------------
# 4. Display what was loaded
# ---------------------------------------------------------

print("Score file:", score_file)
print("Output file:", output_file)
print("Number of lines:", len(lines))

for line in lines[:5]:
    print(line.strip())


# ---------------------------------------------------------
# 5. Separate bona fide and spoof scores
# ---------------------------------------------------------

target_scores = []
nontarget_scores = []

for line in lines:

    parts = line.strip().split()

    label = parts[2]
    score = float(parts[3])

    if label == "bonafide":
        target_scores.append(score)

    elif label == "spoof":
        nontarget_scores.append(score)


print("Number of bona fide scores:", len(target_scores))
print("Number of spoof scores:", len(nontarget_scores))


# ---------------------------------------------------------
# 6. Create score-label pairs
# ---------------------------------------------------------

score_label_pairs = []

for score in target_scores:
    score_label_pairs.append([score, 1])

for score in nontarget_scores:
    score_label_pairs.append([score, 0])


# ---------------------------------------------------------
# 7. Sort score-label pairs by score
# ---------------------------------------------------------

score_label_pairs.sort()


# ---------------------------------------------------------
# 8. Extract labels from sorted score-label pairs
# ---------------------------------------------------------

labels = []

for pair in score_label_pairs:
    labels.append(pair[1])


# ---------------------------------------------------------
# 9. Calculate target prefix sums
# ---------------------------------------------------------

tar_trial_sums = []

running_target = 0

for label in labels:

    if label == 1:
        running_target = running_target + 1

    tar_trial_sums.append(running_target)


# ---------------------------------------------------------
# 10. Calculate non-target exclusive suffix sum
# ---------------------------------------------------------

nontarget_trial_sums = [0] * len(labels)

running_nontarget = 0

for i in range(len(labels) - 1, -1, -1):

    nontarget_trial_sums[i] = running_nontarget

    if labels[i] == 0:
        running_nontarget += 1


# ---------------------------------------------------------
# 11. Check the calculated values
# ---------------------------------------------------------

print("\nFirst 10 score-label pairs:")
for pair in score_label_pairs[:10]:
    print(pair)


print("\nFirst 10 labels:")
print(labels[:10])


print("\nFirst 10 target prefix sums:")
print(tar_trial_sums[:10])


print("\nFirst 10 non-target remaining counts:")
print(nontarget_trial_sums[:10])


print("\nLast 10 score-label pairs:")
for pair in score_label_pairs[-10:]:
    print(pair)


print("\nLast 10 target prefix sums:")
print(tar_trial_sums[-10:])


print("\nLast 10 non-target remaining counts:")
print(nontarget_trial_sums[-10:])


# ---------------------------------------------------------
# 12. Calculate FNR and FPR for every threshold
# ---------------------------------------------------------

n_target = len(target_scores)
n_nontarget = len(nontarget_scores)

fnr = []
fpr = []

fnr.append(0)
fpr.append(1)

for i in range(len(labels)):

    # Number of target trials incorrectly rejected
    false_negatives = tar_trial_sums[i]

    # Number of non-target trials incorrectly accepted
    false_positives = nontarget_trial_sums[i]

    current_fnr = false_negatives / n_target
    current_fpr = false_positives / n_nontarget

    fnr.append(current_fnr)
    fpr.append(current_fpr)

# ---------------------------------------------------------
# 13. Check FNR and FPR
# ---------------------------------------------------------

print("\nFirst 10 FNR values:")
print(fnr[:10])

print("\nFirst 10 FPR values:")
print(fpr[:10])

print("\nLast 10 FNR values:")
print(fnr[-10:])

print("\nLast 10 FPR values:")
print(fpr[-10:])


# ---------------------------------------------------------
# 14. Find the point where FNR and FPR are closest
# ---------------------------------------------------------

minimum_difference = float("inf")
eer_index = 0

for i in range(len(fnr)):

    difference = abs(fnr[i] - fpr[i])

    if difference < minimum_difference:
        minimum_difference = difference
        eer_index = i



# ---------------------------------------------------------
# 15. Calculate EER
# ---------------------------------------------------------

eer = (fnr[eer_index] + fpr[eer_index]) / 2

eer_threshold = score_label_pairs[eer_index][0]

print("\nEER results:")
print("EER:", eer)
print("EER percentage:", eer * 100)
print("Threshold:", eer_threshold)
print("FNR at EER:", fnr[eer_index])
print("FPR at EER:", fpr[eer_index])


os.makedirs(output_dir, exist_ok=True)

with open(output_file, "w") as file:
    file.write(f"Source file: {score_file}\n")
    file.write(f"EER: {eer}\n")
    file.write(f"EER percentage: {eer * 100}\n")
    file.write(f"Threshold: {eer_threshold}\n")
    file.write(f"FNR at EER: {fnr[eer_index]}\n")
    file.write(f"FPR at EER: {fpr[eer_index]}\n")

print("\nResult saved to:", output_file)