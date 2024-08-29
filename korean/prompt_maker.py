class PromptMaker:
    def __init__(self, problem_type):
        self.problem_type = problem_type

    def create_prompt(self):
        if self.problem_type == "문학: <보기> 분석하기":
            return self.analyze_example_prompt()

        elif self.problem_type == "작품 비교하기":
            return self.compare_masterpiece_prompt()

        elif self.problem_type == "품사" or "언어의 본질":
            return self.grammar_prompt()

        else:
            return self.other_prompt()

    def other_prompt(self):
        message = [
            (
                "system",
                "당신은 국어 문학 문제를 출제하는 교사입니다.",
            ),
            (
                "assistant",
                "주어진 문제와 주어진 지문을 바탕으로 반드시 주어진 개념 안에서 주어진 유형의 변형된 문제와 정답, 해설과 함께 제시하세요. 반드시 선택지는 제공한 글과 동어 제시 절대 금지입니다. 반드시 결과를 키와 문자열 값을 큰따옴표로 감싼 올바른 JSON 형식으로 출력해 주세요.- 출력 형식: JSON, 출력 형태: {required_format}",
            ),
            (
                "user",
                "주어진 문제: {extracted_problem_text}, 주어진 지문: {masterpiece_text}, 주어진 개념: {result_concept} 주어진 유형:{problem_type}",
            ),
        ]
        return message

    def analyze_example_prompt(self):
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
                "주어진 문제: {extracted_problem_text}, 주어진 지문: {masterpiece_text}, 주어진 개념: {result_concept} 주어진 유형:{problem_type}",
            ),
        ]

        return message

    def compare_masterpiece_prompt(self):
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
                "주어진 문제: {extracted_problem_text}, 주어진 지문: {masterpiece_text}, 비교할 지문: {compared_masterpiece} 주어진 개념: {result_concept} 주어진 유형:{problem_type}",
            ),
        ]

        return message

    def grammar_prompt(self):
        message = [
            ("system", "당신은 국어 문법 문제를 출제하는 교사입니다."),
            (
                "assistant",
                "주어진 문제를 바탕으로 반드시 주어진 개념 안에서 주어진 유형의 유사 문제와 정답, 해설과 함께 제시하세요. 반드시 주어진 유형의 5지선다(선택지 5개) 문제를 제시해야합니다. 반드시 결과를 키와 문자열 값을 큰따옴표로 감싼 올바른 JSON 형식으로 출력해 주세요.- 출력 형식: JSON, 출력 형태: {required_format}",
            ),
            (
                "user",
                "주어진 문제: {extracted_problem_text},  주어진 개념: {result_concept} 주어진 유형:{problem_type}",
            ),
        ]

        return message
