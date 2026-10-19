"""Exercise API failure paths and UI wiring without downloading model weights."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from demo.backend.cache_config import TASK_NAMES, discover_caches
from demo.backend.main_fastapp_backend import app
from demo.frontend.frontend_utils import response_request


class DemoTests(unittest.TestCase):
    def test_missing_assets_are_visible_and_do_not_allow_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"NEUROSTEER_CACHE_DIR": directory}):
                with TestClient(app) as client:
                    self.assertEqual(client.get('/health').status_code, 503)
                    response = client.post('/process', json={"input_dict": {}, "input_str": "Hello"})
                    self.assertEqual(response.status_code, 503)
                    self.assertIn('sentiment cache', response.json()['detail'])

    def test_validation_and_successful_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"NEUROSTEER_CACHE_DIR": directory}):
                with TestClient(app) as client:
                    invalid_payloads = [
                        {"input_dict": {}, "input_str": " "},
                        {"input_dict": {"unknown": 1}, "input_str": "Hello"},
                        {"input_dict": {"sentiment_pos-neg": -1}, "input_str": "Hello"},
                        {"input_dict": {"sentiment_pos-neg": 301}, "input_str": "Hello"},
                    ]
                    for payload in invalid_payloads:
                        self.assertEqual(client.post('/process', json=payload).status_code, 422)
                    # A deterministic stub tests API wiring only, never model quality.
                    app.state.engine = (lambda *arguments: arguments[1] + ' test', None, None, None, {})
                    self.assertEqual(client.get('/health').status_code, 200)
                    response = client.post('/process', json={"input_dict": {}, "input_str": "Hello"})
                    self.assertEqual(response.json(), {"result": "Hello test"})

    def test_cache_discovery_ignores_unrelated_files_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            for task_name in TASK_NAMES:
                task_folder = Path(directory) / f'gpt2-small_{task_name}_layer_6_datasize_ALL_batchsize32_topK_100'
                task_folder.mkdir()
                (task_folder / f'{task_name}_neuron_info_cache_of_gpt2-small_l6.pkl').touch()
            (Path(directory) / '.DS_Store').touch()
            with patch.dict(os.environ, {"NEUROSTEER_CACHE_DIR": directory}):
                self.assertEqual(set(discover_caches()), set(TASK_NAMES))
                (task_folder / 'extra_neuron_info.pkl').touch()
                with self.assertRaises(ValueError):
                    discover_caches()

    @patch('demo.frontend.frontend_utils.requests.post')
    def test_frontend_direction_mapping_and_timeout(self, post_request):
        post_request.return_value.json.return_value = {"result": "real service output"}
        with patch.dict(os.environ, {"NEUROSTEER_BACKEND_URL": 'http://localhost:8000/'}):
            response = response_request('Hello', {"sen": 10, "tox": 5, "sta": 0, "pol": -3})
        self.assertEqual(response, 'real service output')
        arguments = post_request.call_args
        self.assertEqual(arguments.args[0], 'http://localhost:8000/process')
        self.assertEqual(arguments.kwargs['json']['input_dict'], {
            'sentiment_pos-neg': 10, 'toxicity_neg-pos': 5, 'polite_neg-pos': 3,
        })
        self.assertEqual(arguments.kwargs['timeout'], (5, 180))

    @patch('demo.frontend.frontend_utils.requests.post')
    def test_connection_error_is_readable(self, post_request):
        import requests
        post_request.side_effect = requests.ConnectionError()
        self.assertIn('无法连接', response_request('Hello', {"sen": 0}))


if __name__ == '__main__':
    unittest.main()
