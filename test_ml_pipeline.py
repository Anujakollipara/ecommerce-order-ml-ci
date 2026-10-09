
import json
import os
import unittest

import joblib
import pandas as pd


class TestEcommerceMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(os.path.exists("ecommerce.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("ecommerce_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_model_prediction(self):
        model = joblib.load("ecommerce_model.pkl")
        data = pd.read_csv("ecommerce.csv")

        sample = data.drop(
            columns=[
                "Order_Status",
                "Order_ID",
                "Customer_ID",
                "Customer_Name"
            ],
            errors="ignore"
        ).head(5)

        predictions = model.predict(sample)

        self.assertEqual(len(predictions), len(sample))
        self.assertTrue(
            set(predictions).issubset(set(model.classes_))
        )


if __name__ == "__main__":
    unittest.main()
