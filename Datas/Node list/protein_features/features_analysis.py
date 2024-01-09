import json
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
from skfeature.function.similarity_based import fisher_score
from imblearn.over_sampling import SMOTE

with open(r"C:\Users\George\Desktop\ISEF-2023\Datas\Node list\protein_features\protein_features_uniprot_blast.json", "r") as json_file:
    protein_features = json.load(json_file)

with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\labels\alzheimer_disease\label_dictionary.json', 'r') as json_file:
    alzheimer_labels = json.load(json_file)

with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Signal pathway\back up data\cell_communication_edge_node_relation_cleaned copy.json', 'r') as json_file:
    signal_trans = json.load(json_file)

keys = []

for i in protein_features.keys():
    keys += list(protein_features[i].keys())
    keys = set(keys)
    keys = list(keys)


rk = ['Glycosylation','COFACTOR', 'Domain', 'DOMAIN', 'Peptide', 'Non-terminal residue', 'MASS SPECTROMETRY', 'ALLERGEN', 'BIOTECHNOLOGY', 'PHARMACEUTICAL', 'RNA EDITING', 'BIOPHYSICOCHEMICAL PROPERTIES', 'WEB RESOURCE', 'Lipidation', 'Motif', 'POLYMORPHISM', 'Intramembrane', 'Non-standard residue', 'Sequenence conflict', 'Topological domain', 'PATHWAY', 'Chain', 'Site', 'CAUTION', 'DEVELOPMENTAL STAGE', 'Signal', 'INDUCTION', 'Repeat', 'Transit peptide', 'ACTIVITY REGULATION', 'Propeptide', 'DNA binding', 'Zinc finger', 'MISCELLANEOUS', 'Turn', 'Beta strand', 'Coiled coil', 'Cross-link', 'Region', 'SEQUENCE CAUTION', 'Alternative sequence', 'Compositional bias', 'Initiator methionine', 'Helix']
keys = [i for i in keys if i not in rk]
print(keys)

protein_feature  = {}
for protein in protein_features.keys():
    local_features = {}
    x = protein_features[protein]
    for feature in keys:
        if feature in x:
            local_features[feature] = x[feature]
        else:
            local_features[feature] = 0

    if protein in signal_trans.keys():
        local_features["STP involvement"] = len(signal_trans[protein]['pathways'])
    else:
        local_features["STP involvement"] = 0

    local_features["label"] = alzheimer_labels[protein]

    protein_feature[protein] = local_features

df = pd.DataFrame.from_dict(protein_feature, orient='index')
df.to_csv(r"C:\Users\George\Desktop\ISEF-2023\Datas\Node list\protein_features\protein_features_beforefilter.csv", index=True)
print("DataFrame saved as 'protein_features.csv' successfully.")


feature_columns = df.columns[1:-1]  # Assuming the feature columns are from the second column to the second-to-last column
label_column = df.columns[-1]  # Assuming the label column is the last column


#smote

# Apply SMOTE to address class imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(df[feature_columns], df[label_column])
df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=feature_columns), pd.DataFrame(y_resampled, columns=[label_column])], axis=1)
df = df_resampled

#smote

feature_columns = df.columns[0:-1]  # Assuming the feature columns are from the second column to the second-to-last column
label_column = df.columns[-1]  # Assuming the label column is the last column


save_directory = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\protein_features'

# Set the font size and font family
plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})

# Calculate the variation for each column
column_variation = df[feature_columns].var()
column_variation = column_variation.sort_values(ascending=True)
plt.figure(figsize=(20, 12)) 
column_variation.plot(kind='bar')
plt.xlabel('Columns')
plt.ylabel('Variation')
plt.title('Variation of each column')
plt.savefig(save_directory + '\\column_variation.png', dpi=300, bbox_inches='tight')

importances = mutual_info_classif(df[feature_columns], df[label_column])
feat_importances = pd.Series(importances, index=feature_columns)
feat_importances = feat_importances.sort_values(ascending=True)
plt.figure(figsize=(20, 12))  # Set the figure size for the plot
feat_importances.plot(kind='barh')
plt.xlabel('Information Gain')
plt.ylabel('Features')
plt.title('Information Gain for Features')
plt.savefig(save_directory + '\\information_gain.png', dpi=300, bbox_inches='tight')

# Plot the correlation matrix as a heatmap
correlation_matrix = df[feature_columns].corr()
plt.figure(figsize=(20, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".3f", square=True)
plt.title('Pearson Correlation Heatmap')
plt.tight_layout()
plt.savefig(save_directory + '\\correlation_coefficient.png', dpi=300, bbox_inches='tight')



df_label_1 = df[df[label_column] == 1]

# Get the feature columns
ranks_1 = fisher_score.fisher_score(df_label_1[feature_columns].values, df_label_1[label_column].values)
feat_importances_1 = pd.Series(ranks_1, index=feature_columns)
feat_importances_1 = feat_importances_1.sort_values(ascending=True)
# Filter the DataFrame for label value 0
df_label_0 = df[df[label_column] == 0]
ranks_0 = fisher_score.fisher_score(df_label_0[feature_columns].values, df_label_0[label_column].values)
feat_importances_0 = pd.Series(ranks_0, index=feature_columns)
feat_importances_0 = feat_importances_0.sort_values(ascending=True)
# Calculate the Fisher score for both label values combined
ranks_both = fisher_score.fisher_score(df[feature_columns].values, df[label_column].values)
feat_importances_both = pd.Series(ranks_both, index=feature_columns)
feat_importances_both = feat_importances_both.sort_values(ascending=True)

plt.figure(figsize=(20, 12))

# Plot the Fisher score for label value 1
plt.subplot(1, 3, 1)
feat_importances_1.plot(kind='barh')
plt.xlabel('Fisher Score')
plt.ylabel('Features')
plt.title('Fisher Score for Label 1')

# Plot the Fisher score for label value 0
plt.subplot(1, 3, 2)
feat_importances_0.plot(kind='barh')
plt.xlabel('Fisher Score')
plt.ylabel('Features')
plt.title('Fisher Score for Label 0')

# Plot the Fisher score for both label values combined
plt.subplot(1, 3, 3)
feat_importances_both.plot(kind='barh')
plt.xlabel('Fisher Score')
plt.ylabel('Features')
plt.title('Fisher Score for Both Labels')

plt.tight_layout()
plt.savefig(save_directory + '\\fisher_score.png', dpi=300, bbox_inches='tight')

































































