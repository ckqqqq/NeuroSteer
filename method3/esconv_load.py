import json


def load_esconv(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        esconv = json.load(f)
    return esconv


def process_esconv(file_path):
    """
    处理给定文件路径中的 esconv 数据。

    参数:
        file_path (str): esconv 数据文件的路径。

    返回:
        list: 一个包含字典的列表，每个字典包含以下键：
            - 'dialog' (str): 连接的对话，每个回合格式为 "speaker: content"。
            - 'situation' (str): 对话中的情境描述。
            - 'problem_type' (str): 对话中讨论的问题类型。
            - 'emotion' (str): 与对话相关的情绪类型。
    """
    esconv = load_esconv(file_path)
    all_dialog = []
    print(len(esconv))
    for conv in esconv:
        dialog = ""
        for turn in conv["dialog"]:
            dialog += turn["speaker"] + ": " + turn["content"].strip() + "\n"
        dialog.strip()
        all_dialog.append(
            {
                "dialog": dialog,
                "situation": conv["situation"],
                "problem_type": conv["problem_type"],
                "emotion": conv["emotion_type"],
            }
        )
    return all_dialog


if __name__ == "__main__":
    file_path = "/home/szk/szk_2024/Database/Emotional-Support-Conversation/ESConv.json"
    esconv = process_esconv(file_path)  # 获取所有的对话历史数据，转换成str格式
    print(len(esconv))
    print(esconv[0])
