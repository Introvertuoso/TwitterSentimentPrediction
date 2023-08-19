import numpy as np
import sklearn.metrics as skm


def eval_clf(clf, datasets):
    for split_name, dataset in zip(['train', 'validation'], datasets):
        X_i, y_i = dataset
        if X_i is not None:
            y_pred = clf.predict(X_i)
            print(f'\nSplit: {split_name}')
            print(skm.classification_report(y_i, y_pred))


def eval_reg(reg, datasets):
    # Evaluate our predictor quantitatively
    for split_name, dataset in zip(['train', 'valididation'], datasets):
        X_i, y_i = dataset
        if X_i is not None:
            y_pred = reg.predict(X_i)

            rmse = np.sqrt(skm.mean_squared_error(y_i, y_pred))
            print(f'\nSplit: {split_name}')
            print(f"\tRMSE: {rmse:.2f}")
            mae = skm.mean_absolute_error(y_i, y_pred)
            print(f"\tMAE: {mae:.2f}")
