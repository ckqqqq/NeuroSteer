import requests
def response_request(prompt,val_info)->str:
    trans={"sen":"sentiment","pol":"polite","sta":"debate","tox":"toxicity"}
    alphas={}
    for key,val in val_info.items():
        task=trans[key]
        if task=="toxicity":
            val=-val
        if val>0:
            alphas[f'{task}_pos-neg']=abs(val)
        else:
            alphas[f'{task}_neg-pos']=abs(val)
    # response=generate_response(prompt,alphas)
    payload = {
            "input_dict": alphas,
            "input_str": prompt
        }
    FASTAPI_URL = "http://localhost:10478/process"
    # 发送POST请求到FastAPI
    print("response send",val_info)
    response = requests.post(FASTAPI_URL, json=payload)
    
    if response.status_code == 200:
        # 显示返回的结果
        result = response.json()
        response=result['result']
    else:
        response=f"请求失败，check backend: {response.status_code}"
    return response