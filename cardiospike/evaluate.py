from pathlib import Path

import pandas as pd

from cardiospike import SMART_MODEL_PATH, TEST_PATH, WELLTORY_PATH, API_E2E_ARTIFACTS_DIR, TRAIN_DATA_PATH
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


def run_e2e():
    user_id = '9'
    model = SmartModel(str(SMART_MODEL_PATH))
    data_df, _ = load_test_dataset(version='1.0')
    predictions = classify_sequence(model, user_id, data_df)

    print(predictions.dict().keys())

    return predictions


def main():
    predictions = run_e2e()

    data_df = load_test_dataset(version='1.0')
    data_df = load_train_dataset(version='1.0')
    data_df = data_df.sort_values("time").reset_index(drop=True)


    data_df
    print(data_df.columns)


if __name__ == '__main__':
    main()
