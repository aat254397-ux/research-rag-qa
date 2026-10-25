import unittest

from fastapi.testclient import TestClient

from api import create_app


class TestWebPage(unittest.TestCase):
    def test_home_serves_question_page(self):
        with TestClient(create_app(index_path=None)) as client:
            response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Research QA", response.text)
        self.assertIn('/assets/app.js', response.text)
        self.assertIn('/assets/favicon.svg', response.text)

    def test_favicon_and_legacy_path_serve_the_icon(self):
        with TestClient(create_app(index_path=None)) as client:
            icon = client.get("/assets/favicon.svg")
            legacy = client.get("/favicon.ico", follow_redirects=True)
        for response in (icon, legacy):
            self.assertEqual(response.status_code, 200)
            self.assertIn("image/svg+xml", response.headers["content-type"])
            self.assertIn("Research QA", response.text)

    def test_page_assets_are_available(self):
        with TestClient(create_app(index_path=None)) as client:
            javascript = client.get("/assets/app.js")
            stylesheet = client.get("/assets/styles.css")
        self.assertEqual(javascript.status_code, 200)
        self.assertEqual(stylesheet.status_code, 200)
        self.assertIn("requestJSON", javascript.text)
        self.assertIn(".workspace", stylesheet.text)

    def test_static_route_does_not_serve_project_secrets_or_indexes(self):
        with TestClient(create_app(index_path=None)) as client:
            for path in ("/assets/.env", "/assets/data/indexes/retrieval_index.json"):
                with self.subTest(path=path):
                    self.assertEqual(client.get(path).status_code, 404)


if __name__ == "__main__":
    unittest.main()
