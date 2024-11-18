import esconv_load
import json


def formatted_json_item(question: str, text: str) -> str:
    formatted_str = f"""
    "question": "{question}",
    "choices": {{
        "label": [
            "A",
            "B",
            "C",
            "D"
        ],
        "text": [
            "{text}"
        ]
    }},
    "answerKey": "这个字段暂时还不知道有啥用，需要读论文，暂时可以用一段固定的str做占位符"
    """
    return formatted_str


def formatted_question(dialog: str, problem_type: str, text: str) -> str:
    formatted_str = f"""
    Let's define such a classification task:
    <your task>
    1. Firstly, I will provide you with a conversation between a seeker and a supporter.
    2. The seeker in the conversation is currently experiencing some psychological issues, which is {problem_type}.
    3. Please analyze the above conversation history and select from the choices after the dialog what strategy the counselor used to address the {problem_type} issue, and provide reasons.
    4. When answering, please provide a letter representing the option and the corresponding text in the first line, and then provide the reason why you chose this option in the second line. Please remember that you can only provide two lines of text to answer the question
    </your task>

    Here is the conversation between the seeker and the supporter:
    <conversation>
    {dialog}
    </conversation>

    The choices are as follows:
    {text}
    """
    return formatted_str


def formatted_text():
    pass


"""
all_dialog是一个：
        list: 一个包含字典的列表，每个字典包含以下键：
            - 'dialog' (str): 连接的对话，每个回合格式为 "speaker: content"。
            - 'situation' (str): 对话中的情境描述。
            - 'problem_type' (str): 对话中讨论的问题类型。
            - 'emotion' (str): 与对话相关的情绪类型。


"""


def main():
    all_dialog = esconv_load.load_esconv(
        "/home/szk/szk_2024/Database/Emotional-Support-Conversation/ESConv.json"
    )
    for dialog in all_dialog:
        json_content = ""
        for turn in dialog:
            json_content.append(
                formatted_json_item(turn["dialog"], turn["problem_type"])
            )
        with open("processed_esconv.json", "w", encoding="utf-8") as f:
            json.dump(json_content, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
