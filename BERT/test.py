"""
@Project ：machine-learning 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：jinke
@Date    ：2025/5/13 17:45 
"""
import torch
from transformers import BertTokenizer, BertForQuestionAnswering, AutoConfig, AutoModel

# 是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 下载未经微调的BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='model')
model = BertForQuestionAnswering.from_pretrained('./').to(device)
# model = BertForQuestionAnswering.from_pretrained('modelscope').to(device)
print(model)

# 修改输入维度，微调 embedding
# config = AutoConfig.from_pretrained("./")
# config.max_position_embeddings = 5120
# model = AutoModel.from_config(config)
# print(model)

# 评估未经微调的BERT的性能
def china_capital():
    question, text = "What is the population of Shenzhen? ", "The population of Shenzhen is approximately 13 million."
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs.to(device))
    answer_start_index = torch.argmax(outputs.start_logits)
    answer_end_index = torch.argmax(outputs.end_logits) + 1
    predict_answer_tokens = inputs['input_ids'][0][answer_start_index:answer_end_index]
    predicted_answer = tokenizer.decode(predict_answer_tokens)
    print("What is the population of Shenzhen? ", predicted_answer)


china_capital()
