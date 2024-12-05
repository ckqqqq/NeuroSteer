import utils as utils

def test1():
    print("测试所有模型的加载并打印参数")
    model,tokenizer=utils.get_model_and_tokenizer("modelA")
    print(model)
    print(tokenizer)
    # model,tokenizer=model_factory.get_model_and_tokenizer("modelB")
    # print(model)
    # model,tokenizer=model_factory.get_model_and_tokenizer("modelC")
    # print(model)

if __name__ == "__main__":
    test1()