import torch.nn as nn
from transformers import BertModel

class NeuralNet(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(768, hidden_size) 
        self.l2 = nn.Linear(hidden_size, num_classes) 
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        out = self.l1(pooled_output)
        out = self.relu(out)
        out = self.l2(out)
        return out

# import torch
# import torch.nn as nn
# from transformers import BertModel

# class NeuralNet(nn.Module):
#     def __init__(self, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.hidden_size = hidden_size
#         self.l1 = nn.Linear(hidden_size, hidden_size) 
#         self.l2 = nn.Linear(hidden_size, num_classes) 
#         self.relu = nn.ReLU()
#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids, attention_mask)
#         pooled_output = outputs.pooler_output
#         out = self.l1(pooled_output)
#         out = self.relu(out)
#         out = self.l2(out)
#         # no activation and no softmax at the end
#         return out
#     # def __init__(self, input_size, hidden_size, num_classes):
#     #     super(NeuralNet, self).__init__()
#     #     self.l1 = nn.Linear(input_size, hidden_size) 
#     #     self.l2 = nn.Linear(hidden_size, hidden_size) 
#     #     self.l3 = nn.Linear(hidden_size, num_classes)
#     #     self.relu = nn.ReLU()
    
#     # def forward(self, x):
#     #     out = self.l1(x)
#     #     out = self.relu(out)
#     #     out = self.l2(out)
#     #     out = self.relu(out)
#     #     out = self.l3(out)
#     #     # no activation and no softmax at the end
#     #     return out
