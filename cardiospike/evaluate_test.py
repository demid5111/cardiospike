from pandas._testing import assert_frame_equal
import pytest
from cardiospike import API_E2E_ARTIFACTS_DIR, EVALUATION_REPORTS_PATH
from cardiospike.api.models import Predictions
from cardiospike.evaluate import create_analytics_report, analyze_report, initialize_environment, \
    create_single_user_report_df


@pytest.mark.serial
def test_evaluate_against_rest_api_server():
    data_df, model = initialize_environment(dataset_type='test', dataset_version='1.0')
    user_id = 9

    predictions_df = create_analytics_report(dataset_type='test', dataset_version='1.0',
                                             user_ids=('9',), dump=False)

    expected_predictions = Predictions.parse_file(API_E2E_ARTIFACTS_DIR / 'predict_9.json')
    expected_predictions_df = create_single_user_report_df(data_df, user_id, expected_predictions)

    assert_frame_equal(predictions_df, expected_predictions_df), 'Predictions should be the same'


@pytest.mark.serial
def test_evaluate_against_own_prediction_capability():
    report_df = create_analytics_report(dataset_type='train', dataset_version='1.0',
                                        user_ids=('1', '10', '265'), dump=False)
    precision, recall, f1 = analyze_report(user_id="265", report_df=report_df)

    expected_precision = .8857
    expected_recall = .9118
    expected_f1 = .8986

    accepted_threshold = 1e-4

    assert pytest.approx(precision, abs=accepted_threshold) == expected_precision
    assert pytest.approx(recall, abs=accepted_threshold) == expected_recall
    assert pytest.approx(expected_f1, abs=accepted_threshold) == expected_f1


@pytest.mark.slow
def test_evaluate_against_full_train_dataset():
    report_path = EVALUATION_REPORTS_PATH
    create_analytics_report(dataset_type='train', dataset_version='1.0',
                            user_ids=None, dump=True, report_path=report_path)

    precision, recall, f1 = analyze_report(report_path=report_path)

    expected_precision = .9782
    expected_recall = .9532
    expected_f1 = .9656

    accepted_threshold = 1e-4

    assert pytest.approx(precision, abs=accepted_threshold) == expected_precision
    assert pytest.approx(recall, abs=accepted_threshold) == expected_recall
    assert pytest.approx(expected_f1, abs=accepted_threshold) == expected_f1
