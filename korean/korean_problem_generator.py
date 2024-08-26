import random


class ProblemGenerator:
    def __init__(self, **kwargs):
        self.ocr_service = kwargs.get("ocr_service")
        self.embeddings_model = kwargs.get("embeddings_model")
        self.index = kwargs.get("index")

    def read_image(self, uploaded_image):
        # ocr 인식용 이미지 사전 처리
        file_bytes = uploaded_image.getvalue()
        file_extension = uploaded_image.type.split("/")[1]
        file_title = "problem_image"

        # 이미지에서 추출된 문제
        extracted_text = self.ocr_service.extract_text(
            file_bytes, file_extension, file_title
        )

        return extracted_text

    def vectorize_and_search_similar(self, extracted_text, category, type):
        # 텍스트를 벡터로 변환
        extracted_problem_vector = self.embeddings_model.embed_query(extracted_text)

        # 검색 필터 정의
        query_filter = {
            "text_type": {"$eq": f"{type}"},
            "main_category": {"$eq": f"{category}"},
        }

        # 필터를 적용한 검색
        query_results = self.index.query(
            vector=[extracted_problem_vector],
            top_k=5,
            include_metadata=True,
            filter=query_filter,  # 필터 추가
        )

        # 결과에서 "text" 필드만 추출하여 결합
        temp_text = [match["metadata"]["text"] for match in query_results["matches"]]
        result_text = "\n\n".join(temp_text)

        return result_text

    def get_random_masterpiece(self, extracted_text, type):
        extracted_problem_vector = self.embeddings_model.embed_query(extracted_text)

        query_filter = {"text_type": {"$eq": f"{type}"}}
        query_results = self.index.query(
            vector=[extracted_problem_vector],
            top_k=2,  # 임의로 큰 숫자를 설정하여 최대한 많은 결과를 가져옴
            include_metadata=True,
            filter=query_filter,  # 필터 추가
        )
        if len(query_results["matches"]) > 1:
            second_most_similar = query_results["matches"][
                1
            ]  # 두 번째 유사한 결과 선택
            result_text = second_most_similar["metadata"]["text"]
        else:
            result_text = None  # 두 번째 유사한 결과가 없을 경우 None 반환

        return result_text
