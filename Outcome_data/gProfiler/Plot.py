# -*- coding: utf-8 -*-
"""ISEF Graphs

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1l6InQQ6mIR4g2eBimBsaB_qMydpr77ZO
"""

import csv

#Known
known_file = open(r'C:\Users\George\Desktop\ISEF-2023\Outcome_data\gProfiler\gProfiler_known.csv', "r")
known_reader = csv.reader(known_file)

MF_List = []
BP_List = []
KEGG_List = []
REAC_List = []

for row in known_reader:
  if row[0] == "GO:MF":
    MF_List.append(row)
  elif row[0] == "GO:BP":
    BP_List.append(row)
  elif row[0] == "KEGG":
    KEGG_List.append(row)
  elif row[0] == "REAC":
    REAC_List.append(row)
  else:
    continue

#Unknown
unknown_file = open(r'C:\Users\George\Desktop\ISEF-2023\Outcome_data\gProfiler\gProfiler_unknown.csv', "r")
unknown_reader = csv.reader(unknown_file)

Unknown_MF_List = []
Unknown_BP_List = []
Unknown_KEGG_List = []
Unknown_REAC_List = []

for row in unknown_reader:
  if row[0] == "GO:MF":
    Unknown_MF_List.append(row)
  elif row[0] == "GO:BP":
    Unknown_BP_List.append(row)
  elif row[0] == "KEGG":
    Unknown_KEGG_List.append(row)
  elif row[0] == "REAC":
    Unknown_REAC_List.append(row)
  else:
    continue



def find_intersection(list1, list2):
    set1 = set(item[1] for item in list1)
    set2 = set(item[1] for item in list2)

    intersection_set = set1.intersection(set2)

    intersection_list1 = [item for item in list1 if item[1] in intersection_set]
    intersection_list2 = [item for item in list2 if item[1] in intersection_set]

    return intersection_list1, intersection_list2

Known_List_MF, Our_Predictions_MF = find_intersection(MF_List, Unknown_MF_List)
Known_List_BP, Our_Predictions_BP = find_intersection(BP_List, Unknown_BP_List)
Known_List_REAC, Our_Predictions_REAC = find_intersection(REAC_List, Unknown_REAC_List)
Known_List_KEGG, Our_Predictions_KEGG = find_intersection(KEGG_List, Unknown_KEGG_List)


def pair_lists(A, B):
    dict_B = {}
    for sublist in B:
        key = sublist[1]
        if key in dict_B:
            dict_B[key].append(sublist)
        else:
            dict_B[key] = [sublist]
    result = []
    for sublist in A:
        key = sublist[1]
        if key in dict_B:
            result.extend([[sublist, match] for match in dict_B[key]])
    return result

MF_result = pair_lists(Known_List_MF, Our_Predictions_MF)
BP_result = pair_lists(Known_List_BP, Our_Predictions_BP)
KEGG_result = pair_lists(Known_List_KEGG, Our_Predictions_KEGG)
REAC_result = pair_lists(Known_List_REAC, Our_Predictions_REAC)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import pandas as pd

def create_graph(data,name):
    x_values = []
    known_scores = []
    unknown_scores = []

    for idx, sublist in enumerate(data):
        known_score = float(sublist[0][5])
        unknown_score = float(sublist[1][5])

        x_values.append(idx + 1)
        known_scores.append(known_score)
        unknown_scores.append(unknown_score)

    # Plotting the known scores
    #plt.scatter(x_values, known_scores, color='blue', label='Known Score (Actual)')

    # Plotting the unknown scores
    #plt.scatter(x_values, unknown_scores, color='orange', label='Unknown Score (Predicted)')

    plt.xlabel('Pathways')
    plt.ylabel('P-Values, -Log_10')
    plt.title('Approved and Predicted Targets Enrichment for ' + str(name))

    print(len(x_values))
    print(len(unknown_scores))
    print(len(known_scores))
    window_size = 8  # Adjust this for more or less smoothing
    y_smoothed = pd.Series(unknown_scores).rolling(window=window_size).mean()
    plt.plot(x_values, y_smoothed, color='#99b3ff', label='Moving Average')
    y_smoothed = pd.Series(known_scores).rolling(window=window_size).mean()
    plt.plot(x_values, y_smoothed, color='#0000ff', label='Moving Average')
    #plt.scatter(x_values, known_scores, color='pink', label='Targets of FDA Approved Drugs')
    #plt.scatter(x_values, unknown_scores, color='orange', label='Inferred Proteins')
    plt.legend(loc='upper right')
    plt.ylim(0, 256)
    plt.yscale('symlog', base=2)
    #plt.yticks([2, 5, 10, 20, 40, 100])


    plt.show()

    

    # Plot the original data and the LOWESS smoothed data
    #plt.scatter(x_values, unknown_scores, color='blue', alpha=0.6, label='unknown')'''
    '''
    lowess = sm.nonparametric.lowess
    frac = 0.05  
    y_lowess = lowess(unknown_scores, x_values, frac=frac)
    plt.plot(y_lowess[:, 0], y_lowess[:, 1], color='green', label='LOWESS')
    y_lowess = lowess(known_scores, x_values, frac=frac)
    plt.plot(y_lowess[:, 0], y_lowess[:, 1], color='red', label='LOWESS')
    #plt.scatter(x_values, known_scores, color='blue', label='Known Score (Actual)')
    #plt.scatter(x_values, unknown_scores, color='orange', label='Unknown Score (Predicted)')
    plt.legend(loc='upper right')
    plt.ylim(bottom=0)

    plt.show()'''

#MF_graph = create_graph(MF_result,'Molecular Function')

#BP_graph = create_graph(BP_result,'Biological Process')

KEGG_graph = create_graph(KEGG_result,"KEGG")

REAC_graph = create_graph(REAC_result,"REAC")