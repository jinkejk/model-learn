"""
@Project ：machine-learning 
@File    ：finetune.py
@IDE     ：PyCharm 
@Author  ：jinke
@Date    ：2025/5/13 16:17
基于 Pytorch 微调 BERT 实现问答任务
"""
import pickle

import torch
from modelscope import AutoTokenizer, AutoModel
from modelscope.models.nlp.mglm.data_utils.wordpiece import BertTokenizer
from transformers.data.processors.squad import SquadV2Processor, squad_convert_examples_to_features
from transformers import BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


if __name__ == '__main__':
    #模型下载
    # from modelscope import snapshot_download
    # model_dir = snapshot_download('AI-ModelScope/bert-base-uncased', local_dir='model')
    # tokenizer = AutoTokenizer.from_pretrained('modelscope')
    # model = AutoModel.from_pretrained("modelscope")

    # 加载SQuAD 2.0数据集
    processor = SquadV2Processor()
    train_examples = processor.get_train_examples('squad2.0')
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased', cache_dir='modelscope')

    # 将SQuAD 2.0示例转换为BERT输入特征
    train_features = squad_convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=True,
        return_dataset=False,
        threads=8
    )

    # 将特征保存到磁盘上
    with open('training_features.pkl', 'wb') as f:
        pickle.dump(train_features, f)
