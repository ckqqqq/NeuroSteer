import json

def load_esconv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        esconv = json.load(f)
    return esconv

def process_esconv(file_path):
    esconv = load_esconv(file_path)
    all_dialog = []
    print(len(esconv))
    for conv in esconv:
        dialog = ''
        for turn in conv['dialog']:
            dialog += turn['speaker'] + ': ' + turn['content'].strip() + '\n'
        dialog.strip()
        all_dialog.append({'dialog': dialog, 'situation': conv['situation'], 'problem_type': conv['problem_type'], 'emotion': conv['emotion_type']})
    return all_dialog

if __name__ == '__main__':
    file_path = '/home/szk/szk_2024/Database/Emotional-Support-Conversation/ESConv.json'
    esconv = process_esconv(file_path)    # 获取所有的对话历史数据，转换成str格式
    print(len(esconv))
    print(esconv[0])