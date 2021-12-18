from pathlib import Path

import numpy as np
import pandas as pd

from cardiospike import SMART_MODEL_PATH, TEST_PATH, TRAIN_DATA_PATH, EVALUATION_REPORTS_PATH
from cardiospike.api.inference import SmartModel
from cardiospike.api.models import Predictions


class NoSuchUserInDatasetError(Exception):
    pass


def load_test_dataset(version="1.0"):
    if version == '1.0':
        return pd.read_csv(TEST_PATH)
    raise NotImplementedError


def load_train_dataset(version="1.0"):
    if version == '1.0':
        return pd.read_csv(TRAIN_DATA_PATH)
    raise NotImplementedError


def load_analytics_report(version="1.0", reports_path=None):
    if version == '1.0' and reports_path:
        df = pd.read_csv(reports_path)
        return df
    raise NotImplementedError


def get_unique_user_ids(report_df):
    return [str(u) for u in report_df.id.unique()]


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


def initialize_environment(dataset_type, dataset_version):
    model = SmartModel(str(SMART_MODEL_PATH))

    if dataset_type == 'test':
        data_df = load_test_dataset(version=dataset_version)
    else:  # 'train'
        data_df = load_train_dataset(version=dataset_version)
    return data_df, model


def create_single_user_report_df(data_df, user_id, predictions):
    user_df = get_user_data(data_df, user_id)
    user_df['raw_probs'] = predictions.anomaly_proba
    user_df['y_predicted'] = np.where(user_df['raw_probs'] > predictions.anomaly_thresh, 1, 0)
    user_df.drop('raw_probs', axis=1, inplace=True)
    return user_df


def create_analytics_report(dataset_type, dataset_version, user_ids=None, dump=True, report_path: Path = None):
    data_df, model = initialize_environment(dataset_type=dataset_type, dataset_version=dataset_version)

    frames = []
    users_to_iterate = get_unique_user_ids(data_df) if user_ids is None else user_ids
    for i, user_id in enumerate(users_to_iterate):
        print(f'Processing user {{{user_id}}} ({i + 1}/{len(users_to_iterate)}) ...')
        predictions = classify_sequence(model, user_id, data_df)
        user_df = create_single_user_report_df(data_df, user_id, predictions)
        frames.append(user_df)
    res_df = pd.concat(frames)

    if dump:
        res_df.to_csv(report_path, index=False)
    return res_df


def filter_df(df, settings):
    res_df = df.loc[(df['y'] == settings['y']) & (df['y_predicted'] == settings['y_predicted'])]
    if settings['id'] is not None:
        res_df = res_df.loc[(res_df['id'] == int(settings['id']))]
    return res_df


def analyze_report(user_id=None, report_df=None, report_path=None):
    if report_df is None:
        report_df = load_analytics_report(reports_path=report_path)

    if user_id is not None:
        res_df = report_df.loc[(report_df['id'] == int(user_id))]
        if not len(res_df):
            raise NoSuchUserInDatasetError()

    tp_df = filter_df(report_df, dict(y=1, y_predicted=1, id=user_id))
    tp = len(tp_df)

    fp_df = filter_df(report_df, dict(y=0, y_predicted=1, id=user_id))
    fp = len(fp_df)

    tn_df = filter_df(report_df, dict(y=0, y_predicted=0, id=user_id))
    tn = len(tn_df)

    fn_df = filter_df(report_df, dict(y=1, y_predicted=0, id=user_id))
    fn = len(fn_df)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fp) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def main():
    report_path = EVALUATION_REPORTS_PATH
    create_analytics_report(dataset_type='train', dataset_version='1.0',
                            user_ids=None, dump=True, report_path=report_path)

    precision, recall, f1 = analyze_report(report_path=report_path)
    print(f'Report:')
    print(f'    precision: {precision:.4f}')
    print(f'    recall: {recall:.4f}')
    print(f'    f1: {f1:.4f}')
    print(f'    Rule: TP is when anomaly (y=1) was predicted by the model as well')
    print(f'    model: {SMART_MODEL_PATH}')
    print(f'    dataset: {TRAIN_DATA_PATH}')


if __name__ == '__main__':
    main()
