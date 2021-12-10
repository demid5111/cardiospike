from pathlib import Path

import pandas as pd

from cardiospike import SMART_MODEL_PATH, TEST_PATH, WELLTORY_PATH, API_E2E_ARTIFACTS_DIR
from cardiospike.api.inference import SmartModel
from cardiospike.api.models import Predictions
from cardiospike.evaluate import run_e2e


def test_clasify():
    predictions = run_e2e(user_id=9, dataset_type='test', dataset_version='1.0')

    expected_predictions = Predictions.parse_file(API_E2E_ARTIFACTS_DIR / 'predict_9.json')

    assert predictions == expected_predictions, 'Predictions should be the same'
