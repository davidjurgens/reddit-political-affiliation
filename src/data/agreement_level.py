# import krippendorff
import numpy as np
import pandas as pd

file_path = '/home/kalkiek/projects/reddit-political-affiliation/src/data/Gold Samples.xlsx'

stop_row = 94

# Read in all of our annotations
bohan = pd.read_excel(file_path, sheet_name='Bohan').head(stop_row)
david = pd.read_excel(file_path, sheet_name='David').head(stop_row)
kenan = pd.read_excel(file_path, sheet_name='Kenan').head(stop_row)

print(bohan.head())


def build_array_of_labels(annotations_sheet):
    user_labels = []

    # Loop through each row
    for index, row in annotations_sheet.iterrows():

        if row['Is Liberal?']:
            user_labels.append(1)
        elif row['Is Conservative?']:
            user_labels.append(2)
        elif row['Can’t Tell / Ambiguous / Both']:
            user_labels.append(3)
        elif ['Neither']:
            user_labels.append(4)

    return user_labels


bohan_labels = build_array_of_labels(bohan)
david_labels = build_array_of_labels(david)
kenan_labels = build_array_of_labels(kenan)

labels_matrix = np.matrix([bohan_labels, david_labels, kenan_labels])
# print("Krippendorff's alpha for nominal metric: ", krippendorff.alpha(reliability_data=labels_matrix,
#                                                                       level_of_measurement='nominal'))


def get_label_name(label):
    if label == 1:
        return "Liberal"
    if label == 2:
        return "Conservative"
    if label == 3:
        return "Can’t Tell / Ambiguous / Both"
    return "Neither"


def output_disagreement_for_pair(annotations_sheet, labeler_one, labels_one, labeler_two, labels_two):
    # Find where the annotators disagreed
    disagreement_indices = np.where(np.not_equal(labels_one, labels_two))[0]

    for index in disagreement_indices:
        row = annotations_sheet.iloc[index]
        labeler_one_value = get_label_name(labels_one[index])
        labeler_two_value = get_label_name(labels_two[index])
        print("Annotators {} and {} disagreed on the comment: {}".format(labeler_one, labeler_two, row['text']))
        print(
            "{} labelled it with: {}. While {} labelled it with: {}".format(labeler_one, labeler_one_value, labeler_two,
                                                                            labeler_two_value))
        print()


output_disagreement_for_pair(david, 'David', david_labels, 'Bohan', bohan_labels)
output_disagreement_for_pair(david, 'David', david_labels, 'Kenan', kenan_labels)
output_disagreement_for_pair(david, 'Bohan', bohan_labels, 'Kenan', kenan_labels)
