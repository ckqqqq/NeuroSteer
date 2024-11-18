# 方法三
用GPT对比一下结果，验证下列猜想：

*　大模型和小模型谁会更对不确定性的评估更加准确

此目录下各文件的作用：
esconv_load.py：定义一个加载ESconv数据集的方法，目前只用到了原数据集中的四个字段：dialog、situation、problem_type、emotion
self_eval.py: 初步测试实验，写了一个简单的prompt，让模型自己评估自己是否有解决这个问题的能力，目前这段代码应该不用了
papera_code.py: 论文A实验代码的重构版本，可以参考这段代码进行我们自己实验代码的编写，这段原代码可以阅读以理解其原理
method3_experiment.py: 目前实验3的初步代码，但是因为缺乏匹配格式的数据集，目前还没有运行过，需要处理好数据之后再运行同时debug