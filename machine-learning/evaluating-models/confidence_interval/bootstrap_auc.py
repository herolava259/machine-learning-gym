import pandas as pd
import numpy as np

from sklearn import roc_auc_score

def bootstrap_auc(y, pred, classes, bootstraps = 1000, fold_size=1000):

    stats = np.zeros((len(classes), bootstraps))


    for c in range(len(classes)):

        df = pd.DataFrame(columns = ['y', 'pred'])

        df.loc[:, 'y'] = y[:, c]
        df.loc[:, "pred"] = pred[:, c]

        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]

        prevalence = len(df_pos) / len(df)

        for i in range(bootstraps):

            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace = True)
            neg_sample = df_neg.sample(n = int(fold_size * prevalence), replace = False)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])

            score = roc_auc_score(y_sample, pred_sample)
            stats[c][i] = score


    return stats



