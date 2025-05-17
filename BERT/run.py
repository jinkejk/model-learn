"""
@Project ：machine-learning 
@File    ：run.py
@IDE     ：PyCharm 
@Author  ：jinke
@Date    ：2025/5/13 17:38

https://blog.csdn.net/FrenzyTechAI/article/details/131958410
"""
import pickle

import torch
from modelscope import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import BertForQuestionAnswering


if __name__ == '__main__':
    with open('training_features.pkl', 'rb') as f:
        train_features = pickle.load(f)

    # 定义训练参数
    train_batch_size = 8
    num_epochs = 3
    learning_rate = 3e-5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将特征转换为PyTorch张量
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

    # 全量数据训练
    # train_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_start_positions,
    #                               all_end_positions)
    num_samples = 100
    train_dataset = TensorDataset(
        all_input_ids[:num_samples],
        all_attention_mask[:num_samples],
        all_token_type_ids[:num_samples],
        all_start_positions[:num_samples],
        all_end_positions[:num_samples])
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    # 加载BERT模型和优化器
    model = BertForQuestionAnswering.from_pretrained('modelscope').to(device)
    # tokenizer = AutoTokenizer.from_pretrained('modelscope')
    # model = AutoModel.from_pretrained("modelscope")
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 微调BERT
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            input_ids, attention_mask, token_type_ids, start_positions, end_positions = tuple(t.to(device) for t in batch)
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            start_positions=start_positions,
                            end_positions=end_positions)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # Print the training loss every 500 steps
            if step % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

    # 保存微调后的模型
    model.save_pretrained('./')

