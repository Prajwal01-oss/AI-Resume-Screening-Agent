import os
import tempfile
import unittest

from src.parser import parse_resume
from src.similarity import compute_similarity


class PipelineTests(unittest.TestCase):
    def test_parse_resume_returns_text_for_text_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("Experienced Python developer with AWS and Docker")
            path = handle.name

        try:
            text = parse_resume(path)
            self.assertIn("Experienced Python developer", text)
        finally:
            os.unlink(path)

    def test_compute_similarity_returns_value_between_zero_and_one(self):
        score = compute_similarity([1.0, 0.0], [0.8, 0.2])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
