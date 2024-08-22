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

    def vectorize_and_search_similar(self, extracted_text):
        extracted_problem_vector = self.embeddings_model.embed_query(extracted_text)

        query_results = self.index.query(
            vector=[extracted_problem_vector], top_k=4, include_metadata=True
        )

        temp_text = [match["metadata"]["text"] for match in query_results["matches"]]
        result_text = "\n\n".join(temp_text)

        return result_text
