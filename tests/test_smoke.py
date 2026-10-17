"""Basic smoke test verifying importability of core dependencies."""
import unittest


class TestDependencies(unittest.TestCase):
    def test_imports(self):
        import numpy as np
        import pandas as pd
        import plotly
        import sklearn
        import streamlit

        self.assertIsNotNone(streamlit.__version__)
        self.assertIsNotNone(pd.__version__)
        self.assertIsNotNone(np.__version__)
        self.assertIsNotNone(sklearn.__version__)
        self.assertIsNotNone(plotly.__version__)


if __name__ == "__main__":
    unittest.main()
