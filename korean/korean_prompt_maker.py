class PromptMaker:
    def __init__(self, problem_type):
        self.problem_type = problem_type

    def create_prompt(self):
        if self.problem_type == "문법":
            return self.grammar_prompt()
        elif self.problem_type == "문학":
            return self.literature_prompt()

    def grammar_prompt(self):
        message = [
            ("system", "당신은 국어 문제를 출제하는 교사입니다."),
            (
                "assistant",
                "입력받은 중학생 국어 {main_category} {sub_category} 문제를 입력 받은 텍스트의 개념 내용을 벗어나지 않고 내용 안에서 비슷하게 바꿔서 반드시 같은 질문 형식으로 유사한 문제와 선택지를 만들어 주고 답과 해설도 반드시 제시하세요. 형식: Json-{required_format}",
            ),
            (
                "user",
                "입력받은 문제: {extracted_problem}, 입력받은 문제의 개념 내용: {result_text}, 출력 형태: JSON - 문제",
            ),
        ]

        return message

    def literature_prompt(self):
        message = [
            ("system", "당신은 국어 문제를 출제하는 교사입니다."),
            (
                "assistant",
                "입력받은 작품과 입력받은 중학생 국어 {main_category} {sub_category} 문제를 입력 받은 텍스트의 개념 내용을 벗어나지 않고 내용 안에서 비슷하게 바꿔서 반드시 같은 질문 형식으로 유사한 문제와 선택지를 만들어 주고 답과 해설도 반드시 제시하세요. 반드시 동일하거나 동일한 말을 다른말로 풀어낸 문제가 아닌 유사한 문제를 제시하세요. 형식: Json-{required_format}",
            ),
            (
                "user",
                "입력받은 작품: {extracted_work}, 입력받은 문제: {extracted_problem}, 입력받은 문제의 개념 내용: {result_text}, 출력 형태: JSON - 문제",
            ),
        ]

        return message
