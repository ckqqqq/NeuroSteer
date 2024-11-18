from openai import OpenAI
from esconv_load import *

client = OpenAI(
    base_url = 'http://10.110.147.66:11434/v1',
    api_key='ollama', # required, but unused
)

file_path = '/home/szk/szk_2024/Database/Emotional-Support-Conversation/ESConv.json'    # esconv路径按需调整
esconv = process_esconv(file_path)

system_prompt = "You are a helpful assistant."
# prompt还需要继续调整
user_prompt = '''
Let's define such a self judgment task:
<your task>
1. Firstly, I will provide you with a conversation between a seeker and a supporter.
2. The seeker in the conversation is currently experiencing some psychological issues, which is {problemtype}
3. The seeker's current mood is {emotion}, and the seeker described his current situation in this way: "{situation}"
4. As a helpful psychological supporter, please carefully consider the following two questions: 
[1] What are the problems faced by the seeker? 
[2] What are your confidence level that you can solve the seeker's problem if you were asked to play the role of a supporter in the conversation?
<your task>

The conversation between the seeker and the supporter:
<conversation>
{dialog}
</conversation>

Next, please analyze this self judgment task according to the steps and finally output a number representing the confidence level.
Please do not provide any additional explanatory information other than the number representing confidence level.(A number between 0 and 10)

Your confidence level score is: 
'''.strip()

for conv in esconv[:2]:
    problemtype, emotion, situation, dialog = conv['problem_type'], conv['emotion'], conv['situation'], conv['dialog']

    response = client.chat.completions.create(
        model="phi3.5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(problemtype=problemtype, emotion=emotion, situation=situation, dialog=dialog)},
        ]
    )
    print(response.choices[0].message.content)