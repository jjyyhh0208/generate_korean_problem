import streamlit as st
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from clova_ocr_service import OCRService
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from korean_vectorizer import ImageReaderAndVectorizer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from prompt_maker import PromptMaker
from gpt_answer_formatter import GptAnswerFormatter
from gpt_chat_completer import GptChatCompleter
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_REGION_NAME = os.getenv("AWS_S3_REGION_NAME")
AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


def main():
    # OCR, Embedding Model, LLM, Pinecone 설정
    ocr_service = OCRService()
    embeddings_model = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4o-2024-08-06")
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    index = pinecone.Index(PINECONE_INDEX_NAME, PINECONE_HOST)
    gpt_chat_completer = GptChatCompleter(llm)

    # 데이터 작업기 설정
    params = {
        "ocr_service": ocr_service,
        "embeddings_model": embeddings_model,
        "index": index,
    }
    image_reader_data_vectorizer = ImageReaderAndVectorizer(**params)
    problem_type = [
        "빈칸 채우기",
        "주제/내용 파악하기",
        "작품 비교하기",
        "표현법 분석하기",
        "문학: <보기> 분석하기",
        "언어의 본질",
        "품사",
    ]
    category_type = ["문법", "문학"]

    # Streamlit 세션 상태 초기화
    st.session_state.setdefault("masterpiece_uploaded", False)
    st.session_state.setdefault("article_uploaded", False)
    st.session_state.setdefault("problem_type_literature", False)
    st.session_state.setdefault("problem_type_grammar", False)
    st.session_state.setdefault("read_masterpiece_finished", False)
    st.session_state.setdefault("read_article_finished", False)
    st.session_state.setdefault("extracted_problem_text", "")
    st.session_state.setdefault("gpt_classified_data", {})

    # 시작--------------------------------------------------------------------
    st.title("교과서 기반 국어 문제 생성")
    uploaded_image = st.file_uploader(
        "문제 이미지를 제공하주세요.",
        type=["jpg", "jpeg", "png"],
        key="problem_uploader",
    )
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        if st.button("이미지 인식"):
            extracted_problem_text = image_reader_data_vectorizer.read_image(
                uploaded_image
            )

            # 전체 세션에 추출된 문제 텍스트 저장
            st.session_state.extracted_problem_text = extracted_problem_text

            # gpt의 답변 형식 지정
            required_gpt_format = {
                "main_category": "분류된 분야",
                "problem_type": "분류된 유형",
            }
            message = [
                ("system", "당신은 국어 문제를 분류하는 교사입니다."),
                (
                    "assistant",
                    "주어진 문제를 바탕으로 주어진 분야 중에서 어떤 분야인지, 주어진 유형 중에서 어떤 유형인지 제시하세요. 반드시 주어진 유형 안에서 골라야합니다. 반드시 결과를 키와 문자열 값을 큰따옴표로 감싼 올바른 JSON 형식으로 출력해 주세요.- 출력 형식: JSON, 출력 형태 (반드시 제공된 키 형식을 지켜주세요): {required_format}",
                ),
                (
                    "user",
                    "주어진 문제: {extracted_problem_text}, 주어진 분야:{category_type}, 주어진 유형: {problem_type}",
                ),
            ]

            gpt_chat_source = {
                "message": message,
                "required_format": required_gpt_format,
                "extracted_problem_text": extracted_problem_text,
                "category_type": category_type,
                "problem_type": problem_type,
                "masterpiece_text": None,
                "result_concept": None,
            }

            gpt_classified_data = gpt_chat_completer.complete_chat(**gpt_chat_source)

            # 전체 세션에 gpt가 분류한 문제 정보 저장
            st.session_state.gpt_classified_data = gpt_classified_data

            st.write(gpt_classified_data)

            # gpt가 분류한 결과에 따라 다른 조건 부여
            if gpt_classified_data["main_category"] == "문학":
                st.session_state.problem_type_literature = True

            elif gpt_classified_data["main_category"] == "문법":
                st.session_state.problem_type_grammar = True

        # 문제가 문학 문제라면
        if st.session_state.problem_type_literature:
            uploaded_masterpiece = st.file_uploader(
                "출제하고자 하는 지문을 올려주세요.",
                type=["jpg", "jpeg", "png"],
                key="masterpiece_uploader",
            )

            if uploaded_masterpiece is not None:
                st.image(uploaded_masterpiece, "Uploaded Image", use_column_width=True)
                st.session_state.masterpiece_uploaded = True

                # gpt_classified_data == {"main_category": "문학", "problem_type": "유형"}
                gpt_classified_data = st.session_state.gpt_classified_data

                # 문제 유형 저장
                problem_type = gpt_classified_data["problem_type"]

                # 문제 유형별 프롬프트 제작 인스턴스, 답변 형식 생성 인스턴스 등 설정
                prompt_maker = PromptMaker(problem_type)
                gpt_answer_formatter = GptAnswerFormatter(problem_type)

                if st.session_state.masterpiece_uploaded:
                    # 작품 인식 가동
                    if st.button("지문 인식"):
                        masterpiece_text = image_reader_data_vectorizer.read_image(
                            uploaded_masterpiece
                        )
                        st.session_state.read_masterpiece_finished = True

                    # 작품 인식 완료
                    if st.session_state.read_masterpiece_finished:
                        # 문제와 지문을 하나의 query_text 변수에 넣어둠
                        extracted_problem_text = st.session_state.extracted_problem_text
                        query_text = masterpiece_text + extracted_problem_text

                        # 해당 문제와 지문으로 DB에 있는 개념 추출
                        result_concept = (
                            image_reader_data_vectorizer.vectorize_and_search_similar(
                                query_text,
                                gpt_classified_data["main_category"],
                                "개념",
                            )
                        )

                        # 문제 유형별로 프롬프트 다르게 생성
                        if problem_type == "문학: <보기> 분석하기":
                            required_format = (
                                gpt_answer_formatter.create_gpt_result_format()
                            )

                            message = prompt_maker.create_prompt()

                            st.write(problem_type)

                            gpt_chat_source = {
                                "message": message,
                                "required_format": required_format,
                                "extracted_problem_text": extracted_problem_text,
                                "masterpiece_text": masterpiece_text,
                                "result_concept": result_concept,
                                "problem_type": problem_type,
                            }

                            gpt_final_problem = gpt_chat_completer.complete_chat(
                                **gpt_chat_source
                            )
                            st.write(gpt_final_problem)

                        elif problem_type == "작품 비교하기":
                            # 비교할 작품 가져오기
                            compared_masterpiece = (
                                image_reader_data_vectorizer.get_random_masterpiece(
                                    extracted_problem_text, "작품"
                                )
                            )
                            required_format = (
                                gpt_answer_formatter.create_gpt_result_format(
                                    masterpiece_text, compared_masterpiece
                                )
                            )
                            message = prompt_maker.create_prompt()

                            st.write(problem_type)

                            gpt_chat_source = {
                                "message": message,
                                "required_format": required_format,
                                "extracted_problem_text": extracted_problem_text,
                                "masterpiece_text": masterpiece_text,
                                "compared_masterpiece": compared_masterpiece,
                                "result_concept": result_concept,
                                "problem_type": problem_type,
                            }

                            gpt_final_problem = gpt_chat_completer.complete_chat(
                                **gpt_chat_source
                            )
                            st.write(gpt_final_problem)

                        else:
                            required_format = (
                                gpt_answer_formatter.create_gpt_result_format()
                            )
                            message = prompt_maker.create_prompt()

                            st.write(problem_type)

                            # gpt 문제 생성에 필요한 데이터들
                            gpt_chat_source = {
                                "message": message,
                                "required_format": required_format,
                                "extracted_problem_text": extracted_problem_text,
                                "masterpiece_text": masterpiece_text,
                                "result_concept": result_concept,
                                "problem_type": problem_type,
                            }
                            gpt_final_problem = gpt_chat_completer.complete_chat(
                                **gpt_chat_source
                            )
                            st.write(gpt_final_problem)

        # 문제가 문법 문제라면
        elif st.session_state.problem_type_grammar:
            # 세션에서 문제 불러오기
            extracted_problem_text = st.session_state.extracted_problem_text
            is_article = st.radio(label="유형 선택", options=["지문", "지문X"])

            gpt_classified_data = st.session_state.gpt_classified_data
            problem_type = gpt_classified_data["problem_type"]

            # 문제 유형별 프롬프트 제작 인스턴스, 답변 형식 생성 인스턴스 등 설정
            prompt_maker = PromptMaker(problem_type)
            gpt_answer_formatter = GptAnswerFormatter(problem_type)

            if is_article == "지문X":
                if st.button("문제 생성"):
                    result_concept = (
                        image_reader_data_vectorizer.vectorize_and_search_similar(
                            extracted_problem_text,
                            gpt_classified_data["main_category"],
                            "개념",
                        )
                    )

                    required_format = gpt_answer_formatter.create_gpt_result_format()
                    message = prompt_maker.create_prompt()

                    st.write(problem_type)

                    gpt_chat_source = {
                        "message": message,
                        "required_format": required_format,
                        "extracted_problem_text": extracted_problem_text,
                        "result_concept": result_concept,
                        "problem_type": problem_type,
                    }

                    gpt_final_problem = gpt_chat_completer.complete_chat(
                        **gpt_chat_source
                    )
                    st.write(gpt_final_problem)

            elif is_article == "지문":
                uploaded_article = st.file_uploader(
                    "출제하고자 하는 지문을 올려주세요.",
                    type=["jpg", "jpeg", "png"],
                    key="masterpiece_uploader",
                )

                if uploaded_article is not None:
                    st.image(uploaded_article, "Uploaded Image", use_column_width=True)
                    st.session_state.article_uploaded = True

                    if st.session_state.article_uploaded:
                        # 작품 인식 가동
                        if st.button("지문 인식"):
                            article_text = image_reader_data_vectorizer.read_image(
                                uploaded_article
                            )
                            st.session_state.read_article_finished = True

                        # 작품 인식 완료
                        if st.session_state.read_masterpiece_finished:
                            # 문제와 지문을 하나의 query_text 변수에 넣어둠
                            query_text = article_text + extracted_problem_text

                            # 해당 문제와 지문으로 DB에 있는 개념 추출
                            result_concept = image_reader_data_vectorizer.vectorize_and_search_similar(
                                query_text,
                                gpt_classified_data["main_category"],
                                "개념",
                            )

                            required_format = (
                                gpt_answer_formatter.create_gpt_result_format()
                            )
                            message = prompt_maker.create_prompt()

                            st.write(problem_type)

                            gpt_chat_source = {
                                "message": message,
                                "required_format": required_format,
                                "extracted_problem_text": extracted_problem_text,
                                "result_concept": result_concept,
                                "problem_type": problem_type,
                            }

                            gpt_final_problem = gpt_chat_completer.complete_chat(
                                **gpt_chat_source
                            )
                            st.write(gpt_final_problem)


if __name__ == "__main__":
    main()
