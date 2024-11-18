import json


def formatted_question(dialog: str, problem_type: str) -> str:
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
    """.strip()
    return formatted_str


def formatted_json_item(question: str) -> str:
    formatted_dict = {
        "question": question,
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["认知行为疗法", "人本主义疗法", "精神动力学疗法", "家庭治疗"],
        },
        "answerKey": "answerKey占位符",
    }
    return formatted_dict


def formatted_text():
    pass


def main():
    file_path = "/home/szk/szk_2024/Database/Emotional-Support-Conversation/ESConv.json"
    with open(file_path, "r", encoding="utf-8") as f:
        all_dialog = json.load(f)
    esconv_result = []
    for conv in all_dialog:
        dialog = ""
        for turn in conv["dialog"]:
            dialog += turn["speaker"] + ": " + turn["content"].strip() + "\n"
        dialog.strip()
        formatted_dict = formatted_json_item(
            formatted_question(dialog, conv["problem_type"])
        )
        esconv_result.append(formatted_dict)
    with open("esconv_result.json", "w", encoding="utf-8") as f:
        json.dump(esconv_result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
