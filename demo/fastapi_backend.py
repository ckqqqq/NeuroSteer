from fastapi import FastAPI
from pydantic import BaseModel
from neuro_steer_backend import model_preprocess,delta_h,neuron_steer_generate
# 定义请求体模型
class RequestData(BaseModel):
    input_dict: dict  # 接收字典
    input_str: str   # 接收字符串

app = FastAPI()
model=None
sae=None
tokenizer=None
delta_h_info=None
# 在应用启动时调用a函数来加载A
@app.on_event("startup")
async def startup():
    global model,sae,tokenizer,delta_h_info
     # 这里调用a函数并将其结果赋值给A
    print("模型已经加载完毕")
    LLM = "gpt2-small"
    layer = 6
    model,sae,tokenizer=model_preprocess(LLM,layer)
    delta_h_info=delta_h(sae,LLM,layer)

@app.post("/process")
async def process_data(data: RequestData):
    """
    处理输入数据示例：
    - 输入：{"input_dict": {"key": "value"}, "input_str": "test"}
    - 输出：处理后的字符串
    """
    # 处理逻辑（示例：拼接字典和字符串）
    result=neuron_steer_generate(data.input_dict,data.input_str,model,sae,tokenizer,delta_h_info)
    return {"result": result}