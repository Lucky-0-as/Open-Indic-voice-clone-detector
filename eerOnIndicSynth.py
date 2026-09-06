import configparser
import os



#config file loading
config = configparser.ConfigParser()
config.read("./eerOnIndicSynth.conf")



#get the source and output from config
score_file = config["PATHS"]["score_file"]
output_dir = config["PATHS"]["output_file"]

source_file_name = os.path.splitext(os.path.basename(score_file))[0]
output_file = os.path.join(output_dir, source_file_name + "_EER.txt")

#read the score file
with open(score_file, "r") as file:
    lines = file.readlines()

#separate bona fide and spoof scores

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

#create score-label pairs

score_label_pairs = []

for score in target_scores:
    score_label_pairs.append([score, 1])

for score in nontarget_scores:
    score_label_pairs.append([score, 0])


#sort score-label pairs by score

score_label_pairs.sort()


#extract labels from sorted score-label pairs

labels = []

for pair in score_label_pairs:
    labels.append(pair[1])


#calculate target prefix sums

tar_trial_sums = []

running_target = 0

for label in labels:

    if label == 1:
        running_target = running_target + 1

    tar_trial_sums.append(running_target)


#calculate non-target exclusive suffix sum

nontarget_trial_sums = [0] * len(labels)

running_nontarget = 0

for i in range(len(labels) - 1, -1, -1):

    nontarget_trial_sums[i] = running_nontarget

    if labels[i] == 0:
        running_nontarget += 1


#calculate frr and far for every threshold

n_target = len(target_scores)
n_nontarget = len(nontarget_scores)

frr = []
far = []

frr.append(0)
far.append(1)

for i in range(len(labels)):

    false_negatives = tar_trial_sums[i]
    false_positives = nontarget_trial_sums[i]

    current_fnr = false_negatives / n_target
    current_fpr = false_positives / n_nontarget

    frr.append(current_fnr)
    far.append(current_fpr)


#find the point where FNR and FPR are closest

minimum_difference = float("inf")
eer_index = 0

for i in range(len(frr)):

    difference = abs(frr[i] - far[i])

    if difference < minimum_difference:
        minimum_difference = difference
        eer_index = i



#calculate EER

eer = (frr[eer_index] + far[eer_index]) / 2

if eer_index == 0:
    eer_threshold = None
else:
    eer_threshold = score_label_pairs[eer_index - 1][0]

print("\nEER results:")
print("EER:", eer)
print("EER percentage:", eer * 100)
print("Threshold:", eer_threshold)
print("FNR at EER:", frr[eer_index])
print("FPR at EER:", far[eer_index])


os.makedirs(output_dir, exist_ok=True)

with open(output_file, "w") as file:
    file.write(f"Source file: {score_file}\n")
    file.write(f"EER: {eer}\n")
    file.write(f"EER percentage: {eer * 100}\n")
    file.write(f"Threshold: {eer_threshold}\n")
    file.write(f"FNR at EER: {frr[eer_index]}\n")
    file.write(f"FPR at EER: {far[eer_index]}\n")

print("\nResult saved to:", output_file)