from pathlib import Path

import numpy as np
import pandas as pd

from cardiospike import SMART_MODEL_PATH, TEST_PATH, WELLTORY_PATH, API_E2E_ARTIFACTS_DIR, TRAIN_DATA_PATH, \
    EVALUATION_REPORTS_DIR, EVALUATION_REPORTS_PATH
from cardiospike.api.inference import SmartModel
from cardiospike.api.models import Predictions


def load_test_dataset(version="1.0"):
    if version == '1.0':
        df = pd.read_csv(Path(TEST_PATH))
        # wt = pd.read_csv(Path(WELLTORY_PATH))
        # df = pd.concat((df, wt))
        users = [str(u) for u in df.id.unique()]
        return df, users
    raise NotImplementedError


def load_train_dataset(version="1.0"):
    if version == '1.0':
        df = pd.read_csv(Path(TRAIN_DATA_PATH))
        users = [str(u) for u in df.id.unique()]
        return df, users
    raise NotImplementedError


def load_analytics_report(version="1.0"):
    if version == '1.0':
        df = pd.read_csv(Path(EVALUATION_REPORTS_PATH))
        users = [str(u) for u in df.id.unique()]
        return df, users
    raise NotImplementedError


def get_user_data(dataset_df, user_id):
    return dataset_df.loc[dataset_df["id"] == int(user_id)].sort_values("time").reset_index(drop=True)


def classify_sequence(model, user_id, dataset_df):
    user_data_df = get_user_data(dataset_df, user_id)
    anomaly_proba, anomaly_thresh, errors, error_thresh = model.predict(user_data_df["x"].tolist())

    return Predictions(
        study=user_id,
        anomaly_proba=anomaly_proba,
        errors=errors,
        anomaly_thresh=anomaly_thresh,
        error_thresh=error_thresh,
    )


def run_e2e(user_id, dataset_type, dataset_version):
    data_df, users, model = initialize_environment(dataset_type, dataset_version)

    predictions = classify_sequence(model, user_id, data_df)

    return predictions


def initialize_environment(dataset_type, dataset_version):
    model = SmartModel(str(SMART_MODEL_PATH))

    if dataset_type == 'test':
        data_df, users = load_test_dataset(version=dataset_version)
    else:  # 'train'
        data_df, users = load_train_dataset(version=dataset_version)
    return data_df, users, model


def create_single_user_report_df(data_df, user_id, predictions):
    user_df = get_user_data(data_df, user_id)
    user_df['raw_probs'] = predictions.anomaly_proba
    user_df['y_predicted'] = np.where(user_df['raw_probs'] > predictions.anomaly_thresh, 1, 0)
    user_df.drop('raw_probs', axis=1, inplace=True)
    return user_df


def create_analytics_report():
    data_df, users, model = initialize_environment(dataset_type='train', dataset_version='1.0')

    frames = []
    for i, user_id in enumerate(users):
        print(f'Processing user {{{user_id}}} ({i + 1}/{len(users)} ...')
        predictions = classify_sequence(model, user_id, data_df)
        user_df = create_single_user_report_df(data_df, user_id, predictions)
        frames.append(user_df)
    res_df = pd.concat(frames)
    res_df.to_csv(EVALUATION_REPORTS_DIR / 'full_report.csv', index=False)


def filter_df(df, settings):
    res_df = df.loc[(df['y'] == settings['y']) & (df['y_predicted'] == settings['y_predicted'])]
    if settings['id'] is not None:
        res_df = df.loc[(df['id'] == int(settings['id']))]
    return res_df


def analyze_report(user_id=None):
    report_df, _ = load_analytics_report()

    tp_df = filter_df(report_df, dict(y=1, y_predicted=1, id=user_id))
    tp = len(tp_df)

    fp_df = filter_df(report_df, dict(y=0, y_predicted=1, id=user_id))
    fp = len(fp_df)

    tn_df = filter_df(report_df, dict(y=0, y_predicted=0, id=user_id))
    tn = len(tn_df)

    fn_df = filter_df(report_df, dict(y=1, y_predicted=0, id=user_id))
    fn = len(fn_df)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def main():
    # create_analytics_report()

    precision, recall, f1 = analyze_report()
    precision, recall, f1 = analyze_report()
    print(f'Report:')
    print(f'    precision: {precision:.4f}')
    print(f'    recall: {recall:.4f}')
    print(f'    f1: {f1:.4f}')
    print(f'    Rule: TP is when anomaly (y=1) was predicted by the model as well')
    print(f'    model: {SMART_MODEL_PATH}')
    print(f'    dataset: {TRAIN_DATA_PATH}')


if __name__ == '__main__':
    main()
