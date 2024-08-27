class GptAnswerFormatter:
    def __init__(self, problem_type):
        self.problem_type = problem_type

    def create_gpt_result_format(
        self, masterpiece_text=None, compared_masterpiece=None
    ):
        if self.problem_type == "문학: <보기> 분석하기":
            return self.analyze_example_formatter()

        elif self.problem_type == "작품 비교하기":
            return self.compare_masterpiece_formatter(
                masterpiece_text, compared_masterpiece
            )

        elif self.problem_type == "품사" or "언어의 본질":
            return self.grammar_formatter()

        else:
            return self.other_formatter()

    def analyze_example_formatter(self):
        gpt_format = {
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
        return gpt_format

    def compare_masterpiece_formatter(self, masterpiece_text, compared_masterpiece):
        gpt_format = {
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

        return gpt_format

    def other_formatter(self):
        gpt_format = {
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
        return gpt_format

    def grammar_formatter(self):
        gpt_format = {
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

        return gpt_format
