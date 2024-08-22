import base64
import requests
import uuid
import time
import os
import json
from dotenv import load_dotenv

load_dotenv()


class OCRService:
    def __init__(self):
        self.api_url = os.getenv("CLOVA_OCR_API")
        self.secret_key = os.getenv("CLOVA_OCR_API_KEY")

    def extract_text(self, file_bytes, file_extension, file_title):
        base64_data = self._convert_to_base64(file_bytes)
        return self._ocr_request(file_extension, file_title, base64_data)

    def _convert_to_base64(self, file_bytes):
        return base64.b64encode(file_bytes).decode("utf-8")

    def _ocr_request(self, file_extension, file_title, base64_data):
        request_json = {
            "images": [
                {"format": file_extension, "name": file_title, "data": base64_data}
            ],
            "requestId": str(uuid.uuid4()),
            "version": "V2",
            "timestamp": int(round(time.time() * 1000)),
        }
        headers = {"X-OCR-SECRET": self.secret_key, "Content-Type": "application/json"}
        response = requests.post(
            self.api_url,
            headers=headers,
            data=json.dumps(request_json).encode("UTF-8"),
            timeout=15,
        )
        return self._extract_text_from_response(response.text)

    def _extract_text_from_response(self, response_text):
        response_data = json.loads(response_text)
        return " ".join(
            field.get("inferText", "")
            for image in response_data.get("images", [])
            for field in image.get("fields", [])
        )
