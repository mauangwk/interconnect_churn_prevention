# general_functions.py

import sys
import os
sys.path.append(os.getcwd())

# Librerias ----------------------------------------
import params as params

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
#import seaborn as sns

"""
 Analiza si el customerID se encuentra mas de una vez en el data set 
 lo que implicaría que es una relacion 1:n
"""

def analizaDatosLlave(df):
    customer_counts = df['customerID'].value_counts()
    duplicate_ids = customer_counts[customer_counts > 1].index.tolist()

    print("customerIDs que aparecen más de una vez:", duplicate_ids)
    print("-" * 30)


def generaGraficaCorr(data):
    # Graficando la correlacion entre variables numericas Posterior a la estandarizacion
    corr = data.corr()  # Calculate the correlation matrix
    sns.heatmap(corr, annot=False, cmap='coolwarm')  # Create a heatmap
    plt.show()


def displayClassFrequency(y_train):
    class_frequency = y_train.value_counts(normalize=True)
    print("Normalized Class Frequency:")
    print(class_frequency)
    class_frequency.plot(kind='bar')
    plt.savefig(params.IMAGES_OUTPUT_PATH + 'target_class_frequency.png')


def printClassFrequency(y_train):
    class_frequency = y_train.value_counts(normalize=True)
    print("Normalized Class Frequency:")
    print(class_frequency)


# A function to generate oversampling
def generate_oversamples(features, target, nrepeat):

    if (target[target == 0].count() < target[target == 1].count()):
        target_minority_class = target[target == 0]
        target_majority_class = target[target == 1]
        features_minority_class = features[target == 0]
        features_majority_class = features[target == 1]
    else:
        target_minority_class = target[target == 1]
        target_majority_class = target[target == 0]
        features_minority_class = features[target == 1]
        features_majority_class = features[target == 0]

    diff = 0
    if (nrepeat == 0):
        nrepeat = int(target_majority_class.count() /
                      target_minority_class.count())
        diff = target_majority_class.count() % target_minority_class.count()

    features_upsampled = pd.concat(
        [features_majority_class] +
        [features_minority_class] * nrepeat
    )

    target_upsampled = pd.concat(
        [target_majority_class] +
        [target_minority_class] * nrepeat
    )

    if diff > 0:
        features_upsampled = pd.concat(
            [features_upsampled] +
            [features_minority_class.sample(diff, random_state=12345)]
        )
        target_upsampled = pd.concat(
            [target_upsampled] +
            [target_minority_class.sample(diff, random_state=12345)]
        )

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345
    )

    return features_upsampled, target_upsampled



def evaluate_model(model, train_features, train_target, test_features, test_target, metrics_file):

    eval_stats = {}

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):

        eval_stats[type] = {}

        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]

        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba >= threshold)
                     for threshold in f1_thresholds]

        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)
        eval_stats[type][params.roc_auc_label] = roc_auc



        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type][params.aps_label] = aps
        eval_stats[type][params.accuracy_label] = metrics.accuracy_score(target, pred_target)
        eval_stats[type][params.recall_label] = metrics.recall_score(target, pred_target)
        eval_stats[type][params.f1_label] = metrics.f1_score(target, pred_target)

       
        # Grafica de las estadisticas
    
        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # Valor F1
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color,
                label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx],
                    color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'Valor F1')

        # ROC
        ax = axs[1]
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx],
                    color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower center')
        ax.set_title(f'Curva ROC')

        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx],
                    color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')
    
    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=(
        params.f1_label, 
        params.accuracy_label, 
        params.recall_label,
        params.aps_label, 
        params.roc_auc_label
        ))

    df_eval_stats.to_csv(metrics_file)

    print("\nMetrics\n", df_eval_stats, "\n")
    plt.savefig(params.IMAGES_OUTPUT_PATH+"model_evaluation.png")

    return
