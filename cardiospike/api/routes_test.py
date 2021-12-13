import json
import pytest
from pathlib import Path
from typing import Dict

import pandas as pd
from fastapi.testclient import TestClient

from .app import app
from .. import TEST_PATH, WELLTORY_PATH, API_E2E_ARTIFACTS_DIR

client = TestClient(app)


def write_reference_response(api_route, user_id, json_content):
    with open(API_E2E_ARTIFACTS_DIR / f'{api_route}_{user_id}.json', 'w') as f:
        json.dump(json_content, f)


def load_reference_response(api_route, user_id):
    with open(API_E2E_ARTIFACTS_DIR / f'{api_route}_{user_id}.json') as f:
        return json.load(f)


@pytest.mark.serial
def test_read_main():
    user_id = '9'
    api_route = 'predict'

    df = pd.read_csv(Path(TEST_PATH))
    wt = pd.read_csv(Path(WELLTORY_PATH))
    df = pd.concat((df, wt))
    t = df.loc[df["id"] == int(user_id)].sort_values("time").reset_index(drop=True)

    json_data = {
        'study': '9',
        'sequence': t["x"].tolist(),
    }

    response = client.post(f'/{api_route}',
                           headers={"Content-Type": "application/json"},
                           json=json_data)

    # write_reference_response(api_route, user_id, response.json())

    reference_json = load_reference_response(api_route, user_id)
    assert reference_json == response.json()
