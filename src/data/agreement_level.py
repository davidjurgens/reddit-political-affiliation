import krippendorff
import numpy as np
import pandas as pd

file_path = '/home/kalkiek/projects/reddit-political-affiliation/src/data/Gold Samples.xlsx'

stop_row = 94

# Read in all of our annotations
bohan = pd.read_excel(file_path, sheet_name='Bohan').head(stop_row)
david = pd.read_excel(file_path, sheet_name='David').head(stop_row)
kenan = pd.read_excel(file_path, sheet_name='Kenan').head(stop_row)


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
print(labels_matrix)
print("Krippendorff's alpha for nominal metric: ", krippendorff.alpha(reliability_data=labels_matrix,
                                                                      level_of_measurement='nominal'))

# List indices of disagreements

