
import unittest
from app import app


class TestEcommercePredictionAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_valid_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "Gender": "Female",
                "Product_Category": "Electronics",
                "Quantity": 2,
                "Price": 1500.0,
                "Payment_Method": "Credit Card",
                "City": "Guntur",
                "Customer_Rating": 4
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            response.get_json()["predicted_order_status"],
            ["Delivered", "Pending", "Cancelled", "Returned"]
        )

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={"Gender": "Female", "Quantity": 2}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())

    def test_invalid_json_body(self):
        response = self.client.post(
            "/predict",
            data="not json",
            content_type="text/plain"
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
