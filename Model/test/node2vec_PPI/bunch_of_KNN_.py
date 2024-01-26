import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, log_loss, matthews_corrcoef, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_recall_fscore_support as prf_support
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


def run_KNNss_100(csv_file_path):
    json_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\labels\alzheimer_disease\label_dictionary.json'
    json_file_path2 = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\back up\current_protein_Signal+meta+targets.json'
    # Load the embedding DataFrame
    embedding = pd.read_csv(csv_file_path)
    # Load the label dictionary
    with open(json_file_path, 'r') as file:
        label_dictionary = json.load(file)
    # Load the protein list
    with open(json_file_path2, 'r') as file:
        protein_list = json.load(file)



    # Match the features and labels based on protein names
    features = []
    labels = []

    for protein_name in embedding.iloc[:, 0]:
        if protein_name in label_dictionary:
            features.append(embedding.loc[embedding.iloc[:, 0] == protein_name].values[:, 1:])
            labels.append(label_dictionary[protein_name])

    # Convert the lists to numpy arrays
    features = np.concatenate(features)
    labels = np.array(labels)

    # Constants
    num_classifiers = 100
    num_false_samples = 2000

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # Lists to store KNN classifiers and their predictions
    knn_classifiers = []
    knn_predictions = []

    # Loop to train KNN classifiers
    for i in range(num_classifiers):
        # Randomly sample 300 nodes with the label "false"
        false_indices = np.where(y_train == False)[0]
        random_false_indices = np.random.choice(false_indices, size=num_false_samples, replace=False)

        # Combine with all nodes with the label "true"
        true_indices = np.where(y_train == True)[0]
        train_indices = np.concatenate((random_false_indices, true_indices))

        # Create the training set
        X_train_sampled = X_train[train_indices]
        y_train_sampled = y_train[train_indices]

        # Apply SMOTE for balancing the training data
        smote = SMOTE(sampling_strategy=0.5, random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_sampled, y_train_sampled)

        # Create and train the KNN classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=5)
        knn_classifier.fit(X_train_balanced, y_train_balanced)

        # Store the trained classifier
        knn_classifiers.append(knn_classifier)

        # Make predictions on the test set
        predictions = knn_classifier.predict_proba(X_test)[:, 1]
        pre = knn_classifier.predict(X_test)
        knn_predictions.append(predictions)

        precisionsss = precision_score(y_test, pre)
        recallsss = recall_score(y_test, pre)

        #print(f'KNN{i} with Precision: {precisionsss} and Recall: {recallsss}')

    # Calculate the average prediction
    y_pred_prob = np.mean(knn_predictions, axis=0)

    threshold_probability = 0.5
    threshold_points = [0.5, 0.99]  # Example threshold values

    # Apply the threshold probability for classification
    average_predictions_classified = (y_pred_prob >= threshold_probability).astype(int)


    y_pred = average_predictions_classified


    #metric

    # Calculate precision, recall, accuracy, F1 score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculate the confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)

    # Calculate the Matthews correlation coefficient
    matthews_corr = matthews_corrcoef(y_test, y_pred)

    # Calculate macro-averaged precision, recall, F1 score
    macro_precision, macro_recall, macro_f1, _ = prf_support(y_test, y_pred, average="macro")

    # Print the evaluation metrics
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("Accuracy: {:.2f}".format(accuracy))
    print("F1 Score: {:.2f}".format(f1))
    print("Confusion Matrix:")
    print(confusion_mat)
    print("Matthews Correlation Coefficient: {:.4f}".format(matthews_corr))
    print("Macro-averaged Precision: {:.2f}".format(macro_precision))
    print("Macro-averaged Recall: {:.2f}".format(macro_recall))
    print("Macro-averaged F1 Score: {:.2f}".format(macro_f1))

    # Calculate the false positive rate (FPR), true positive rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Calculate the metric for each threshold
    metric_values = fpr * tpr * 10000

    # Find the index of the threshold that maximizes the metric
    max_metric_index = np.argmax(metric_values)

    # Get the corresponding threshold value
    optimal_threshold = thresholds[max_metric_index]

    print("Optimal Threshold: {:.4f}".format(optimal_threshold))
    print("Max Metric Value: {:.4f}".format(metric_values[max_metric_index]))

    # Calculate the area under the ROC curve (AUC)
    auc_score = roc_auc_score(y_test, y_pred_prob)

    # Plot the ROC curve
    plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc_score))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    for threshold in threshold_points:
        idx = np.argwhere(thresholds >= threshold)[0]
        plt.scatter(fpr[idx], tpr[idx], marker='o', label=f'Threshold = {threshold:.2f}')
    plt.legend(loc='lower right')
    plt.show()

run_KNNss_100(r'C:\Users\George\Desktop\ISEF-2023\Model\test\node2vec_PPI\node2vec_embedding_100walks.csv')