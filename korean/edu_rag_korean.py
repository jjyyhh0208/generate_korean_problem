import streamlit as st
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from clova_ocr_service import OCRService
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from korean_problem_generator import ProblemGenerator
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

    # streamlit 페이지 시작
    st.title("교과서 기반 국어 문제 생성")
    problem_type = st.radio("생성 유형을 선택하세요:", ["문법", "문학"])
    prompt_maker = PromptMaker(problem_type)
    gpt_chat_completer = ChatCompleter(problem_type, llm)

    params = {
        "ocr_service": ocr_service,
        "embeddings_model": embeddings_model,
        "index": index,
    }
    problem_generator = ProblemGenerator(**params)

    if problem_type == "문법":
        input_type = st.radio("입력 유형을 선택하세요:", ["문제 사진"])

        if input_type == "문제 사진":
            uploaded_image = st.file_uploader(
                "문제 이미지를 제공해주세요..", type=["jpg", "jpeg", "png"]
            )
            if uploaded_image is not None:
                st.image(
                    uploaded_image, caption="Uploaded Image", use_column_width=True
                )

                if st.button("유사 문제 생성"):
                    extracted_problem = problem_generator.read_image(uploaded_image)
                    st.write("이미지 처리 완료, 문제 생성중..")

                    # 추출된 문제 백터화 및 유사도 검사, 추출
                    result_text = problem_generator.vectorize_and_search_similar(
                        extracted_problem
                    )
                    required_format = {
                        "문제": "유사한문제",
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
                    message = prompt_maker.create_prompt()

                    chat_completion_params = {
                        "message": message,
                        "main_category": problem_type,
                        "sub_catagory": "품사",
                        "required_format": required_format,
                        "extracted_problem": extracted_problem,
                        "result_text": result_text,
                    }

                    finalized_problem = gpt_chat_completer.complete_gpt_chat(
                        **chat_completion_params
                    )

                    st.write(
                        finalized_problem,
                        unsafe_allow_html=True,
                    )

    elif problem_type == "문학":
        if "final_text" not in st.session_state:
            st.session_state.final_text = ""
        problem_subtype = st.radio("문제 유형을 선택해주세요", ["운문", "산문"])

        if problem_subtype == "운문":
            uploaded_work = st.file_uploader(
                "출제하고자 하는 작품을 입력해주세요:",
                type=["jpg", "jpeg", "png"],
                key="work_uploader",
            )

            if uploaded_work is not None:
                st.image(
                    uploaded_work, caption="Uploaded Work Image", use_column_width=True
                )

                if st.button("작품 인식 처리"):
                    # OCR로 작품 추출
                    extracted_poem = problem_generator.read_image(uploaded_work)

                    st.write("작품 처리 완료")
                    st.session_state.final_text += extracted_poem

            uploaded_problem = st.file_uploader(
                "문제를 올려주세요 (작품 인식부터 해주세요)",
                type=["jpg", "jpeg", "png"],
            )
            if uploaded_problem is not None:
                st.image(
                    uploaded_problem,
                    caption="Uploaded Problem Image",
                    use_column_width=True,
                )

                if st.button("유사 문제 생성"):
                    extracted_problem = problem_generator.read_image(uploaded_problem)
                    st.write("문제 인식 완료, 문제 생성중...")
                    st.session_state.final_text += "\n\n" + extracted_problem

                    final_prompt_reference = st.session_state.final_text

                    result_text = problem_generator.vectorize_and_search_similar(
                        final_prompt_reference
                    )
                    st.write(result_text)

                    extracted_poem = final_prompt_reference.split("\n\n")[0]
                    extracted_problem = final_prompt_reference.split("\n\n")[1]
                    required_format = {
                        "문제": "유사한문제",
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
                    message = prompt_maker.create_prompt()

                    chat_completion_params = {
                        "message": message,
                        "main_category": problem_type,
                        "sub_catagory": "운문",
                        "required_format": required_format,
                        "extracted_problem": extracted_problem,
                        "extracted_work": extracted_poem,
                        "result_text": result_text,
                    }

                    finalized_problem = gpt_chat_completer.complete_gpt_chat(
                        **chat_completion_params
                    )

                    st.write(
                        finalized_problem,
                        unsafe_allow_html=True,
                    )

                    # 세션 초기화
                    st.session_state.final_text = ""

        elif problem_subtype == "산문":
            uploaded_work = st.file_uploader(
                "출제하고자 하는 작품을 입력해주세요:",
                type=["jpg", "jpeg", "png"],
                key="work_uploader",
            )

            if uploaded_work is not None:
                st.image(
                    uploaded_work, caption="Uploaded Work Image", use_column_width=True
                )

                if st.button("작품 인식 처리"):
                    # OCR로 작품 추출
                    extracted_story = problem_generator.read_image(uploaded_work)

                    st.write("작품 처리 완료")
                    st.session_state.final_text += extracted_story

            uploaded_problem = st.file_uploader(
                "문제를 올려주세요 (작품 인식부터 해주세요)",
                type=["jpg", "jpeg", "png"],
            )
            if uploaded_problem is not None:
                st.image(
                    uploaded_problem,
                    caption="Uploaded Problem Image",
                    use_column_width=True,
                )

                if st.button("유사 문제 생성"):
                    extracted_problem = problem_generator.read_image(uploaded_problem)
                    st.write("문제 인식 완료, 문제 생성중...")
                    st.session_state.final_text += "\n\n" + extracted_problem

                    final_prompt_reference = st.session_state.final_text

                    result_text = problem_generator.vectorize_and_search_similar(
                        final_prompt_reference
                    )

                    extracted_story = final_prompt_reference.split("\n\n")[0]
                    extracted_problem = final_prompt_reference.split("\n\n")[1]
                    required_format = {
                        "문제": "유사한문제",
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
                    message = prompt_maker.create_prompt()

                    chat_completion_params = {
                        "message": message,
                        "main_category": problem_type,
                        "sub_catagory": "운문",
                        "required_format": required_format,
                        "extracted_problem": extracted_problem,
                        "extracted_work": extracted_story,
                        "result_text": result_text,
                    }

                    finalized_problem = gpt_chat_completer.complete_gpt_chat(
                        **chat_completion_params
                    )

                    st.write(
                        finalized_problem,
                        unsafe_allow_html=True,
                    )

                    # 세션 초기화
                    st.session_state.final_text = ""


if __name__ == "__main__":
    main()
