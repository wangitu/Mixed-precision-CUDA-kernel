import math

import torch
import torch.nn as nn

from .quantizer import (
    RestrictedSymQuantizer, 
    FullSymQuantizer
)


class QuantLinear(nn.Linear):
    def __init__(
        self,
        *args,
        bias=False,
        w_bits=16,
        weight_layerwise=False,
        group_size=-1,
        cherryq=False,
        cherry_fraction=1/256
    ):
        super().__init__(*args, bias=bias)
        
        self.w_bits = w_bits
        self.weight_layerwise = weight_layerwise
        self.group_size = group_size
        self.cherryq = cherryq
        self.cherry_fraction = cherry_fraction
        
        if self.w_bits < 16 and self.w_bits > 1:
            if self.cherryq:
                self.num_cherries = math.ceil(self.weight.shape[-1] * cherry_fraction) // 8 * 8
                self.register_buffer("cherry_indices", -torch.ones(self.num_cherries, dtype=torch.int32))
                self.cherry_indices_registered = False
            self.quantizer = RestrictedSymQuantizer if self.w_bits > 4 else FullSymQuantizer
            self.dequantized = False
            
    def init_weights_(self):
        pass
            
    def _expand_scaling_factors(self, s, weight):
        if s.ndimension() == 0 or s.shape[-1] == 1:
            return s.expand_as(weight)
        else: # group
            return s.repeat_interleave(repeats=self.group_size, dim=-1)[:, :weight.shape[-1]] # (out, in)
    
    @torch.inference_mode()
    def prepare_weight_for_inference(self, weight):
        assert not self.training and not self.dequantized, "Prepare during training or prepare an already dequantized model is not allowed."
        assert not self.cherryq or (self.cherry_indices >= 0).all(), "`cherry_indices` is invalid."
        
        if self.cherryq:
            self.weight.data.copy_(self._adjust_weight(weight, ste=False))
        else:
            weight = weight.reshape(weight.shape[0], -1, self.group_size)
            s = self.quantizer.get_scaling_factor(weight, None, self.w_bits, self.weight_layerwise)
            self.weight.data.copy_(self.quantizer.transform(weight, s, self.w_bits).flatten(1))
        
        self.dequantized = True
        return self.weight
    
    def register_cherry_indices(self, cherry_indices):
        assert cherry_indices.shape[-1] >= self.cherry_indices.shape[-1], "The passed indices are invalid."
        
        self.cherry_indices = cherry_indices.to(dtype=torch.int32, device=self.weight.device)
        self.cherry_indices_registered = True
        
    def _get_scaling_factors(self, weight):
        scaling_factors = []
        for i in range(0, weight.shape[-1], self.group_size):
            scaling_factors.append(
                self.quantizer.get_scaling_factor(weight[:, i: i + self.group_size], None, self.w_bits, self.weight_layerwise)
            )
        return torch.cat(scaling_factors, dim=-1) # (out, num_group)
        
    def _adjust_weight(self, weight: torch.Tensor, ste=True):
        """
        We do not apply weight clipping since
        any clipping-based method will lead to exceptionally high perplexity scores (i.e., > 10000),
        causing a substantial loss of information that proves to be difficult to recover through fine-tuning.
        """
        assert not self.training or self.cherry_indices_registered, "`cherry_indices` is not registered yet."
        
        cherry_indices = self.cherry_indices.long()
        quant_indices = torch.ones(weight.shape[1], dtype=torch.bool, device=weight.device).scatter(-1, cherry_indices, 0)
        quant_indices = quant_indices.nonzero().squeeze(1)
        quant_weight = weight.detach().index_select(-1, quant_indices)
        
        s = self._get_scaling_factors(quant_weight) # (out, num_group)
        s = self._expand_scaling_factors(s, quant_weight)
            
        quant_weight = self.quantizer.transform(quant_weight, s, self.w_bits) # (out, num_group * group_size)
        real_weight = weight.detach().clone()
        real_weight[..., quant_indices] = quant_weight
        
        if ste:
            real_weight = real_weight - weight.detach() + weight # Straight Through Estimator
        return real_weight
    
    def quant_forward(self, input_):
        # quantize weight
        assert self.weight.ndimension() == 2
        
        real_weight = self.weight
        
        if self.w_bits >= 16:
            weight = self.weight
        elif self.w_bits > 1:
            if self.cherryq:
                if self.training:
                    real_weight = self._adjust_weight(real_weight)
                elif not self.dequantized:
                    real_weight = self.prepare_weight_for_inference(real_weight)
            else: # naive QAT
                if self.training:
                    real_weight = real_weight.reshape(real_weight.shape[0], -1, self.group_size) # (out, num_group, group_size)
                    s = self.quantizer.get_scaling_factor(real_weight, None, self.w_bits, self.weight_layerwise) # (out, num_group, 1)
                elif not self.dequantized:
                    real_weight = self.prepare_weight_for_inference(real_weight)
            
            if self.training and not self.cherryq:
                # Backward compatibility for naive QAT
                weight = self.quantizer.apply(
                    real_weight, None, self.w_bits, self.weight_layerwise, s
                ) # (out, num_group, group_size)
                weight = weight.flatten(1) # (out, in)
            else:
                weight = real_weight
        
        else:
            raise NotImplementedError(f"Quantization for {self.w_bits}bit is not implemented yet.")

        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias
        return out
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.quant_forward(x, **kwargs)
    