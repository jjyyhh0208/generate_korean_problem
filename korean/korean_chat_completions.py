from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class ChatCompleter:
    def __init__(self, problem_type, llm):
        self.problem_type = problem_type
        self.llm = llm
        
    def complete_gpt_chat(self, **kwargs):
        self.message = kwargs.get('message')
        self.main_category = kwargs.get('main_category')
        self.sub_category = kwargs.get('sub_category')
        self.required_format = kwargs.get('required_format')
        self.extracted_problem = kwargs.get('extracted_problem')
        self.result_text = kwargs.get('result_text')
        self.extracted_work = kwargs.get('extracted_work')
        
        if self.problem_type == "문법":
            return self.grammar_gpt_complete()
        elif self.problem_type == "문학":
            return self.literature_gpt_complete()

    def grammar_gpt_complete(self):
        chat_prompt = ChatPromptTemplate.from_messages(self.message)
        chain = chat_prompt | self.llm | JsonOutputParser()
        gpt_result = chain.invoke(
            {
                "main_category": self.main_category,
                "sub_category": self.sub_category,
                "required_format": self.required_format,
                "extracted_problem": self.extracted_problem,
                "result_text": self.result_text,
            }
        )
        
        if "보기" in gpt_result:
            reference = f"보기: {gpt_result["보기"]}"
        else:
            reference = None
        problem = gpt_result["문제"]
        options = gpt_result["선택지"]
        formatted_options = "<br>".join(
            f"{key}. {value}" for key, value in options.items()
        )
        answer = gpt_result["정답"]
        explanation = gpt_result["해설"]
        finalized_problem = f"문제:<br>{problem}<br>"
        if reference:
            finalized_problem += f"{reference}<br>"
        finalized_problem += f"{formatted_options}<br><br>답:<br>{answer}<br>해설: {explanation}"
        
        return finalized_problem

    def literature_gpt_complete(self):
        chat_prompt = ChatPromptTemplate.from_messages(self.message)
        chain = chat_prompt | self.llm | JsonOutputParser()
        gpt_result = chain.invoke(
            {
                "main_category": self.main_category,
                "sub_category": self.sub_category,
                "required_format": self.required_format,
                "extracted_work": self.extracted_work,
                "extracted_problem": self.extracted_problem,
                "result_text": self.result_text,
            }
        )
        
        if "보기" in gpt_result:
            reference = f"보기: {gpt_result["보기"]}"
        else:
            reference = None
        problem = gpt_result["문제"]
        options = gpt_result["선택지"]
        formatted_options = "<br>".join(
            f"{key}. {value}" for key, value in options.items()
        )
        answer = gpt_result["정답"]
        explanation = gpt_result["해설"]
        finalized_problem = f"문제:<br>{problem}<br>"
        if reference:
            finalized_problem += f"{reference}<br>"
        finalized_problem += f"{formatted_options}<br><br>답:<br>{answer}<br>해설: {explanation}"

        return finalized_problem
