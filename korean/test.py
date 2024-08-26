from clova_ocr_service import OCRService
from io import BytesIO
from PIL import Image
import os  # os 모듈 추가


def read_image(uploaded_image, file_path):
    ocr_service = OCRService()
    # ocr 인식용 이미지 사전 처리
    file_bytes = uploaded_image.getvalue()
    file_extension = os.path.splitext(file_path)[1][1:]  # 파일 경로에서 확장자 추출
    file_title = "problem_image"

    # 이미지에서 추출된 문제
    extracted_text = ocr_service.extract_text(file_bytes, file_extension, file_title)

    return extracted_text


# 이미지 파일 경로
image_path = "/Users/collegenie/Desktop/스크린샷 2024-08-23 13.32.06.jpeg"

# 이미지 파일을 열기
with open(image_path, "rb") as image_file:
    image_bytes = image_file.read()

# BytesIO 객체로 변환
image_stream = BytesIO(image_bytes)

# 함수 호출, 파일 경로도 함께 전달
extracted_text = read_image(image_stream, image_path)

print(extracted_text)
