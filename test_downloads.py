import unittest
from unittest.mock import Mock

import requests

from download_utils import exception_message, is_permanent_error, response_error


class DownloadErrorTests(unittest.TestCase):
    def test_response_error_preserves_ollama_error_and_status(self):
        response = Mock(status_code=400)
        response.json.return_value = {"error": "failed to fetch: repository not found"}

        self.assertEqual(
            response_error(response, "Pulling hf.co/example/model:Q4_K_M"),
            "Pulling hf.co/example/model:Q4_K_M failed (HTTP 400): failed to fetch: repository not found",
        )

    def test_response_error_falls_back_to_plain_text(self):
        response = Mock(status_code=400)
        response.json.side_effect = ValueError
        response.text = "Bad Request"

        self.assertEqual(
            response_error(response, "Hugging Face download"),
            "Hugging Face download failed (HTTP 400): Bad Request",
        )

    def test_http_4xx_is_permanent_but_5xx_is_retryable(self):
        permanent = requests.HTTPError(response=Mock(status_code=400))
        transient = requests.HTTPError(response=Mock(status_code=503))

        self.assertTrue(is_permanent_error(permanent))
        self.assertFalse(is_permanent_error(transient))
        self.assertIn("HTTP 400", exception_message(permanent, "Hugging Face download"))


if __name__ == "__main__":
    unittest.main()
