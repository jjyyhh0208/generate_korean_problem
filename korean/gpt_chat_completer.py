from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


class GptChatCompleter:
    def __init__(self, llm):
        self.llm = llm

    def complete_chat(self, **chat_source):
        message = chat_source.get("message")
        extracted_problem_text = chat_source.get("extracted_problem_text")
        required_gpt_format = chat_source.get("required_format")
        category_type = chat_source.get("category_type")
        problem_type = chat_source.get("problem_type")
        masterpiece_text = chat_source.get("masterpiece_text")
        result_concept = chat_source.get("result_concept")

        chat_prompt = ChatPromptTemplate.from_messages(message)
        chain = chat_prompt | self.llm | JsonOutputParser()
        gpt_result = chain.invoke(
            {
                "required_format": required_gpt_format,
                "extracted_problem_text": extracted_problem_text,
                "category_type": category_type,
                "problem_type": problem_type,
                "masterpiece_text": masterpiece_text,
                "result_concept": result_concept,
            }
        )

        return gpt_result
