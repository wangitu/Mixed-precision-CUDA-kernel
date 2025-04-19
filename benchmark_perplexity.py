import time
import torch
import torch.nn as nn

from llama.qlinear import QuantLinear
from utils import from_quantized


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    # traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )
    # trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, testenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = testenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, ...


DEV = torch.device('cuda:0')


def get_model(model, model_type):
    if model_type == "opt":
        return model.model.decoder
    else:
        assert model_type == "llama" or model_type == "mistral"
        return model.model
    
def get_layers(model, model_type):
    _model = get_model(model, model_type)
    if model_type == "opt":
        return _model.layers
    else:
        assert model_type == "llama" or model_type == "mistral"
        return _model.layers
    

# function for benchmarking runtime
def benchmark(model, input_ids, check=False):
    layers = get_layers(model, 'llama')

    input_ids = input_ids.to(model.gpus[0] if hasattr(model, "gpus") else DEV)
    torch.cuda.synchronize()

    cache = {"past": None}

    def clear_past(i):
        def tmp(layer, inp, out):
            if cache["past"]:
                cache["past"][i] = None

        return tmp

    for i, layer in enumerate(layers):
        layer.register_forward_hook(clear_past(i))

    print("Benchmarking ...")

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.0

    def sync():
        if hasattr(model, "gpus"):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
        # for i in range(128):
            tick = time.time()
            out = model(
                input_ids[:, i : i + 1],
                past_key_values=cache["past"],
                attention_mask=attention_mask[:, : (i + 1)].reshape((1, -1)),
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            max_memory = max(max_memory, torch.cuda.memory_allocated() / 1024 / 1024)
            if check and i != input_ids.numel() - 1:
                tot += loss(
                    out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)
                ).float()
            cache["past"] = list(out.past_key_values)
            del out
        sync()
        import numpy as np

        print("Median:", np.median(times))
        if check:
            print("PPL:", torch.exp(tot / (input_ids.numel() - 1)).item())
            print("max memory(MiB):", max_memory)


dataloader, testloader = get_wikitext2(128, 0, 2048, '/data/wangqianle/models/llama2-7b')
input_ids = next(iter(dataloader))[0][:, :128]
model = from_quantized('processed_data/quantized_model_4bit', torch_dtype=torch.float16, device_map={'': 0})
model.eval()


with QuantLinear.dmode(layerwise_dequantize=True):
    benchmark(model, input_ids, check=True)

with QuantLinear.dmode(layerwise_dequantize=False):
    benchmark(model, input_ids, check=True)
