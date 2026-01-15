import torch
from transformers.models.llama import LlamaModel,LlamaConfig

# llamaconfig = LlamaConfig(vocab_size=32000,
#                         hidden_size=4096//2,
#                         intermediate_size=11008//2,
#                         num_hidden_layers=32//2,
#                         num_attention_heads=32//2,
#                         max_position_embeddings=4096//2)

llamaconfig = LlamaConfig(vocab_size=1000,
                        hidden_size=16,
                        intermediate_size=40,
                        num_hidden_layers=4,
                        num_attention_heads=4,
                        max_position_embeddings=16)

llamamodel = LlamaModel(config=llamaconfig)
input_ids = torch.randint(low=0,high=llamaconfig.vocab_size, size=(4,30))
res = llamamodel(input_ids)
print(res)

