import streamlit as st
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from clova_ocr_service import OCRService
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from korean_problem_generator import ProblemGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from korean_prompt_maker import PromptMaker
from korean_chat_completions import ChatCompleter
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

    # 데이터 작업기 설정
    params = {
        "ocr_service": ocr_service,
        "embeddings_model": embeddings_model,
        "index": index,
    }
    problem_generator = ProblemGenerator(**params)
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
    if "image_uploaded" not in st.session_state:
        st.session_state.image_uploaded = False

    if "masterpiece_uploaded" not in st.session_state:
        st.session_state.masterpiece_uploaded = False

    if "problem_type_literature" not in st.session_state:
        st.session_state.problem_type_literature = False

    if "problem_type_grammar" not in st.session_state:
        st.session_state.problem_type_grammar = False

    if "read_masterpiece_finished" not in st.session_state:
        st.session_state.read_masterpiece_finished = False

    if "extracted_text" not in st.session_state:
        st.session_state.extracted_text = ""

    if "gpt_classified_data" not in st.session_state:
        st.session_state.gpt_classified_data = {}

    # 시작
    st.title("교과서 기반 국어 문제 생성")
    uploaded_image = st.file_uploader(
        "문제 이미지를 제공하주세요.",
        type=["jpg", "jpeg", "png"],
        key="problem_uploader",
    )
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.session_state.image_uploaded = True
        if st.button("이미지 인식"):
            extracted_text = problem_generator.read_image(uploaded_image)
            st.session_state.extracted_text = extracted_text

            required_gpt_format = {"main_category": "분야", "problem_type": "유형"}
            message = [
                ("system", "당신은 국어 문제를 분류하는 교사입니다."),
                (
                    "assistant",
                    "주어진 문제를 바탕으로 주어진 분야 중에서 어떤 분야인지, 주어진 유형 중에서 어떤 유형인지 제시하세요. 반드시 주어진 유형 안에서 골라야합니다. 반드시 결과를 키와 문자열 값을 큰따옴표로 감싼 올바른 JSON 형식으로 출력해 주세요.- 출력 형식: JSON, 출력 형태: {required_format}",
                ),
                (
                    "user",
                    "주어진 문제: {extracted_text}, 주어진 분야:{category_type}, 주어진 유형: {problem_type}",
                ),
            ]
            chat_prompt = ChatPromptTemplate.from_messages(message)
            chain = chat_prompt | llm | JsonOutputParser()
            gpt_classified_data = chain.invoke(
                {
                    "required_format": required_gpt_format,
                    "extracted_text": extracted_text,
                    "category_type": category_type,
                    "problem_type": problem_type,
                }
            )
            st.session_state.gpt_classified_data = gpt_classified_data
            st.write(gpt_classified_data)

            if gpt_classified_data["main_category"] == "문학":
                st.session_state.problem_type_literature = True

            elif gpt_classified_data["main_category"] == "문법":
                st.session_state.problem_type_grammar = True

        if st.session_state.problem_type_literature:
            uploaded_masterpiece = st.file_uploader(
                "출제하고자 하는 지문을 올려주세요.",
                type=["jpg", "jpeg", "png"],
                key="masterpiece_uploader",
            )

            if uploaded_masterpiece is not None:
                st.image(uploaded_masterpiece, "Uploaded Image", use_column_width=True)
                st.session_state.masterpiece_uploaded = True
                gpt_classified_data = st.session_state.gpt_classified_data

                if st.session_state.masterpiece_uploaded:
                    if st.button("지문 인식"):
                        masterpiece_text = problem_generator.read_image(
                            uploaded_masterpiece
                        )
                        st.session_state.read_masterpiece_finished = True

                    if st.session_state.read_masterpiece_finished:
                        extracted_text = st.session_state.extracted_text
                        query_text = masterpiece_text + extracted_text
                        result_concept = problem_generator.vectorize_and_search_similar(
                            query_text,
                            gpt_classified_data["main_category"],
                            "개념",
                        )
                        problem_type = gpt_classified_data["problem_type"]

                        if "<보기> 분석하기" not in problem_type:
                            if problem_type != "작품 비교하기":
                                required_format = {
                                    "문제": "유사 문제",
                                    "선택지": {
                                        "1": "1번 선택지 내용",
                                        "2": "2번 선택지 내용",
                                        "3": "3번 선택지 내용",
                                        "4": "4번 선택지 내용",
                                        "5": "5번 선택지 내용",
                                    },
                                    "정답": "유사 문제에 대한 정답",
                                    "해설": "답에 대한 근거와 해설",
                                }

                                message = [
                                    (
                                        "system",
                                        "당신은 국어 문학 문제를 출제하는 교사입니다.",
                                    ),
                                    (
                                        "assistant",
                                        "주어진 문제와 주어진 지문을 바탕으로 반드시 주어진 개념 안에서 주어진 유형의 변형된 문제와 정답, 해설과 함께 제시하세요. 반드시 주어진 유형의 5지선다(선택지 5개) 문제를 제시하되, 문제(질문)는 유사하게, 선택지는 제공된 문제와 달라야합니다. 반드시 결과를 키와 문자열 값을 큰따옴표로 감싼 올바른 JSON 형식으로 출력해 주세요.- 출력 형식: JSON, 출력 형태: {required_format}",
                                    ),
                                    (
                                        "user",
                                        "주어진 문제: {extracted_text}, 주어진 지문: {masterpiece_text}, 주어진 개념: {result_concept} 주어진 유형:{problem_type}",
                                    ),
                                ]

                                st.write(problem_type)

                                chat_prompt = ChatPromptTemplate.from_messages(message)
                                chain = chat_prompt | llm | JsonOutputParser()
                                gpt_final_problem = chain.invoke(
                                    {
                                        "required_format": required_format,
                                        "extracted_text": extracted_text,
                                        "masterpiece_text": masterpiece_text,
                                        "result_concept": result_concept,
                                        "problem_type": problem_type,
                                    }
                                )
                                st.write(gpt_final_problem)
                            elif problem_type == "작품 비교하기":
                                compared_masterpiece = (
                                    problem_generator.get_random_masterpiece(
                                        extracted_text, "작품"
                                    )
                                )
                                required_format = {
                                    "가": masterpiece_text,
                                    "나": compared_masterpiece,
                                    "문제": "유사 문제",
                                    "선택지": {
                                        "1": "1번 선택지 내용",
                                        "2": "2번 선택지 내용",
                                        "3": "3번 선택지 내용",
                                        "4": "4번 선택지 내용",
                                        "5": "5번 선택지 내용",
                                    },
                                    "정답": "유사 문제에 대한 정답",
                                    "해설": "답에 대한 근거와 해설",
                                }
                                message = [
                                    (
                                        "system",
                                        "당신은 국어 문학 문제를 출제하는 교사입니다.",
                                    ),
                                    (
                                        "assistant",
                                        "주어진 문제와 주어진 지문을 바탕으로 반드시 주어진 개념 안에서 주어진 지문과 비교할 지문을 비교하는 주어진 유형의 유사 문제, 유사 보기와 정답, 해설과 함께 제시하세요. 여기서 중요한 점은 보기와 선택지, 문제는 형태만 유사해야 합니다. 반드시 주어진 유형의 5지선다(선택지 5개) 문제를 제시해야합니다. 반드시 결과를 키와 문자열 값을 큰따옴표로 감싼 올바른 JSON 형식으로 출력해 주세요.- 출력 형식: JSON, 출력 형태: {required_format}",
                                    ),
                                    (
                                        "user",
                                        "주어진 문제: {extracted_text}, 주어진 지문: {masterpiece_text}, 비교할 지문: {compared_masterpiece} 주어진 개념: {result_concept} 주어진 유형:{problem_type}",
                                    ),
                                ]

                                chat_prompt = ChatPromptTemplate.from_messages(message)
                                chain = chat_prompt | llm | JsonOutputParser()
                                gpt_final_problem = chain.invoke(
                                    {
                                        "required_format": required_format,
                                        "extracted_text": extracted_text,
                                        "masterpiece_text": masterpiece_text,
                                        "compared_masterpiece": compared_masterpiece,
                                        "result_concept": result_concept,
                                        "problem_type": problem_type,
                                    }
                                )
                                st.write(gpt_final_problem)

                        else:
                            required_format = {
                                "문제": "유사 문제",
                                "보기": "유사 보기",
                                "선택지": {
                                    "1": "1번 선택지 내용",
                                    "2": "2번 선택지 내용",
                                    "3": "3번 선택지 내용",
                                    "4": "4번 선택지 내용",
                                    "5": "5번 선택지 내용",
                                },
                                "정답": "유사 문제에 대한 정답",
                                "해설": "답에 대한 근거와 해설",
                            }

                            message = [
                                (
                                    "system",
                                    "당신은 국어 문학 문제를 출제하는 교사입니다.",
                                ),
                                (
                                    "assistant",
                                    "주어진 문제와 주어진 지문을 바탕으로 반드시 주어진 개념 안에서 주어진 유형의 유사 문제, 유사 보기와 정답, 해설과 함께 제시하세요. 여기서 중요한 점은 보기와 선택지, 문제는 형태만 유사해야 합니다. 반드시 주어진 유형의 5지선다(선택지 5개) 문제를 제시해야합니다. 반드시 결과를 키와 문자열 값을 큰따옴표로 감싼 올바른 JSON 형식으로 출력해 주세요.- 출력 형식: JSON, 출력 형태: {required_format}",
                                ),
                                (
                                    "user",
                                    "주어진 문제: {extracted_text}, 주어진 지문: {masterpiece_text}, 주어진 개념: {result_concept} 주어진 유형:{problem_type}",
                                ),
                            ]

                            st.write(problem_type)

                            chat_prompt = ChatPromptTemplate.from_messages(message)
                            chain = chat_prompt | llm | JsonOutputParser()
                            gpt_final_problem = chain.invoke(
                                {
                                    "required_format": required_format,
                                    "extracted_text": extracted_text,
                                    "masterpiece_text": masterpiece_text,
                                    "result_concept": result_concept,
                                    "problem_type": problem_type,
                                }
                            )
                            st.write(gpt_final_problem)
        elif st.session_state.problem_type_grammar:
            result_concept = problem_generator.vectorize_and_search_similar(
                extracted_text,
                gpt_classified_data["main_category"],
                "개념",
            )
            problem_type = gpt_classified_data["problem_type"]
            required_format = {
                "문제": "유사 문제",
                "선택지": {
                    "1": "1번 선택지 내용",
                    "2": "2번 선택지 내용",
                    "3": "3번 선택지 내용",
                    "4": "4번 선택지 내용",
                    "5": "5번 선택지 내용",
                },
                "정답": "유사 문제에 대한 정답",
                "해설": "답에 대한 근거와 해설",
            }
            message = [
                ("system", "당신은 국어 문법 문제를 출제하는 교사입니다."),
                (
                    "assistant",
                    "주어진 문제를 바탕으로 반드시 주어진 개념 안에서 주어진 유형의 유사 문제와 정답, 해설과 함께 제시하세요. 반드시 주어진 유형의 5지선다(선택지 5개) 문제를 제시해야합니다. 반드시 결과를 키와 문자열 값을 큰따옴표로 감싼 올바른 JSON 형식으로 출력해 주세요.- 출력 형식: JSON, 출력 형태: {required_format}",
                ),
                (
                    "user",
                    "주어진 문제: {extracted_text},  주어진 개념: {result_concept} 주어진 유형:{problem_type}",
                ),
            ]
            st.write(problem_type)
            chat_prompt = ChatPromptTemplate.from_messages(message)
            chain = chat_prompt | llm | JsonOutputParser()
            gpt_final_problem = chain.invoke(
                {
                    "required_format": required_format,
                    "extracted_text": extracted_text,
                    "result_concept": result_concept,
                    "problem_type": problem_type,
                }
            )
            st.write(gpt_final_problem)


if __name__ == "__main__":
    main()
