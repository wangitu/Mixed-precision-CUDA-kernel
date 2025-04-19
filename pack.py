import torch

from llama.configuration_llama_quant import LlamaConfig
from llama.modeling_llama_quant import LlamaForCausalLM
from llama.cherry_linear import QuantLinear
from utils import pack_model, save_quantized


base = '/data/wangqianle/models/llama2-7b'
config = LlamaConfig.from_pretrained(
    base, pretraining_tp=1,
    w_bits=4, group_size=128, cherryq=True, cherry_fraction=1/256
)
model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=base,
    config=config,
    torch_dtype=torch.float16,
    device_map={'': 0},
)

cherry_indices_mapping = torch.load('processed_data/cherry_indices/llama2-7b-impact.pt')
for name, module in model.named_modules():
    if isinstance(module, QuantLinear):
        module.register_cherry_indices(cherry_indices_mapping[name])
        
pack_model(model)
save_quantized(model, 'processed_data/quantized_model_4bit')
