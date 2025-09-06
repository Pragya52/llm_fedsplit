"""
model_architecture.py
Split LLaMA 3.2 Model Architecture for Federated Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    LlamaForCausalLM,
    LlamaConfig
)
from typing import Optional, Tuple
import logging
import warnings
import os
import math

logger = logging.getLogger(__name__)
# Suppress warnings
warnings.filterwarnings("ignore")


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Generate frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x, position_ids=None):
        seq_len = x.shape[-2]  # Handle different tensor shapes
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device, dtype=torch.long)
        
        # Handle different position_ids shapes
        if position_ids.dim() == 2:
            # [batch_size, seq_len] -> [seq_len]
            position_ids = position_ids[0]
        elif position_ids.dim() == 1 and position_ids.shape[0] != seq_len:
            # If it's 1-D but wrong length, create new one
            position_ids = torch.arange(seq_len, device=x.device, dtype=torch.long)
        
        # Ensure it's 1-D and convert to float
        position_ids = position_ids.flatten().float()
        
        # Generate frequencies
        freqs = torch.outer(position_ids, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Shape: [1, 1, seq_len, rotary_dim]
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embedding to query and key tensors."""
    # Ensure cos and sin match the head dimension
    head_dim = q.shape[-1]
    if cos.shape[-1] != head_dim:
        cos = cos[..., :head_dim]
        sin = sin[..., :head_dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class ClientModel(nn.Module):
    """
    Client-side model containing embeddings and first N transformer layers
    """
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B", num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.model_name = model_name
        
        try:
            # Try to load Llama 3.2 1B model
            print(f"Loading {model_name}...")
            
            # Check if HF token is set
            hf_token = os.environ.get("HF_TOKEN", None)
            
            if "meta-llama/Llama-3.2" in model_name:
                # Loading actual Llama model
                full_model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float32,
                    device_map="auto",
                    token=hf_token,
                    low_cpu_mem_usage=True
                )
            else:
                # Generic model loading
                full_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float32,
                    device_map="auto",
                    token=hf_token,
                    low_cpu_mem_usage=True
                )
            
        except Exception as e:
            print(f"Could not load {model_name}: {e}")
            print("\nUsing a smaller fallback model for testing...")
            print("To use Llama 3.2 1B:")
            print("1. Get access at: https://huggingface.co/meta-llama/Llama-3.2-1B")
            print("2. Set your HF token: export HF_TOKEN='your_token_here'")
            print("3. Or login with: huggingface-cli login\n")
            
            # Fallback to a smaller public model
            model_name = "EleutherAI/pythia-70m"  # Very small model for testing
            full_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
                device_map="cpu"
            )
        
        # Extract client components
        self.config = full_model.config
        
        # Handle Llama 3.2 architecture
        if hasattr(full_model, 'model'):  # Llama architecture
            self.embed_tokens = full_model.model.embed_tokens
            
            # Only take first N layers
            available_layers = len(full_model.model.layers)
            layers_to_use = min(num_layers, available_layers)
            
            # Use custom LlamaDecoderLayer for better control
            self.layers = nn.ModuleList()
            for i in range(layers_to_use):
                if "llama" in model_name.lower():
                    # Use our custom layer with proper attention
                    layer = LlamaDecoderLayer(self.config)
                    # Copy weights from original layer if available
                    if hasattr(full_model.model.layers[i], 'state_dict'):
                        try:
                            # Get original state dict
                            orig_state_dict = full_model.model.layers[i].state_dict()
                            # Create a new state dict with matched keys
                            new_state_dict = {}
                            for key, value in orig_state_dict.items():
                                # Handle potential key differences
                                if key.startswith('self_attn.'):
                                    new_state_dict[key] = value
                                elif key.startswith('mlp.'):
                                    new_state_dict[key] = value
                                elif key.startswith('input_layernorm.'):
                                    new_state_dict[key] = value
                                elif key.startswith('post_attention_layernorm.'):
                                    new_state_dict[key] = value
                            layer.load_state_dict(new_state_dict, strict=False)
                        except Exception as e:
                            print(f"Could not load weights for layer {i}: {str(e)}")
                            print("Using initialized weights")
                else:
                    # Use original layer for non-Llama models
                    layer = full_model.model.layers[i]
                self.layers.append(layer)
            
        else:  # Fallback architecture (GPT-style)
            if hasattr(full_model, 'gpt_neox'):  # Pythia model
                self.embed_tokens = full_model.gpt_neox.embed_in
                available_layers = len(full_model.gpt_neox.layers)
                layers_to_use = min(num_layers, available_layers)
                self.layers = nn.ModuleList([
                    full_model.gpt_neox.layers[i] for i in range(layers_to_use)
                ])
            else:  # Generic transformer
                self.embed_tokens = full_model.transformer.wte
                available_layers = len(full_model.transformer.h)
                layers_to_use = min(num_layers, available_layers)
                self.layers = nn.ModuleList([
                    full_model.transformer.h[i] for i in range(layers_to_use)
                ])
        
        self.hidden_size = full_model.config.hidden_size
        self.actual_num_layers = len(self.layers)
        
        # Delete full model to save memory
        del full_model
        torch.cuda.empty_cache()
        
        logger.info({
            "model/client_layers": self.actual_num_layers,
            "model/client_model": model_name,
            "model/hidden_size": self.hidden_size
        })
        
        print(f"Client model initialized with {self.actual_num_layers} layers")
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass through client layers
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)

        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long, device=device
            )
        
        # Create causal mask for Llama
        if "llama" in self.model_name.lower():
            # Create 4D causal attention mask
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), hidden_states, 0
            )
        else:
            # Expand attention mask for other architectures
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
                attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Pass through client layers
        for layer in self.layers:
            if "llama" in self.model_name.lower() or hasattr(layer, 'self_attn'):
                # Llama-style layer
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                hidden_states = layer_outputs[0]
            else:
                # Generic transformer layer
                outputs = layer(hidden_states, attention_mask=attention_mask)
                hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        
        return hidden_states, None
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """Create causal attention mask for decoder"""
        # Create causal mask
        batch_size, seq_length = input_shape
        mask = torch.full((seq_length, seq_length), torch.finfo(inputs_embeds.dtype).min, device=inputs_embeds.device)
        mask_cond = torch.arange(mask.size(-1), device=inputs_embeds.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(inputs_embeds.dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(seq_length, past_key_values_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device), mask], dim=-1)
        
        # Expand to batch size and add head dimension
        mask = mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length + past_key_values_length)
        
        # Combine with attention mask
        if attention_mask is not None:
            # Expand attention mask
            if attention_mask.dim() == 2:
                expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length)
                # Convert to float and apply large negative value where mask is 0
                expanded_mask = expanded_mask.to(dtype=inputs_embeds.dtype)
                expanded_mask = (1.0 - expanded_mask) * torch.finfo(inputs_embeds.dtype).min
                mask = mask + expanded_mask

        return mask
    
    def get_num_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ServerModel(nn.Module):
    """
    Server-side model containing remaining transformer layers and output head
    """
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B", start_layer: int = 4):
        super().__init__()
        self.start_layer = start_layer
        self.model_name = model_name
        
        try:
            # Try to load Llama 3.2 1B model
            print(f"Loading server model {model_name}...")
            
            # Check if HF token is set
            hf_token = os.environ.get("HF_TOKEN", None)
            
            if "meta-llama/Llama-3.2" in model_name:
                # Loading actual Llama model
                full_model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float32,
                    device_map="auto",
                    token=hf_token,
                    low_cpu_mem_usage=True
                )
            else:
                # Generic model loading
                full_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float32,
                    device_map="auto",
                    token=hf_token,
                    low_cpu_mem_usage=True
                )
                
        except Exception as e:
            print(f"Could not load {model_name}: {e}")
            print("Using fallback model for server...")
            
            # Fallback to smaller model
            model_name = "EleutherAI/pythia-70m"
            full_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
                device_map="auto"
            )
        
        # Extract server components
        self.config = full_model.config
        
        # Handle Llama 3.2 architecture
        if hasattr(full_model, 'model'):  # Llama architecture
            total_layers = len(full_model.model.layers)
            
            # Take layers from start_layer to end
            self.layers = nn.ModuleList()
            for i in range(min(start_layer, total_layers), total_layers):
                if "llama" in model_name.lower():
                    # Use our custom layer with proper attention
                    layer = LlamaDecoderLayer(self.config)
                    # Copy weights from original layer if available
                    if hasattr(full_model.model.layers[i], 'state_dict'):
                        try:
                            # Get original state dict
                            orig_state_dict = full_model.model.layers[i].state_dict()
                            # Create a new state dict with matched keys
                            new_state_dict = {}
                            for key, value in orig_state_dict.items():
                                # Handle potential key differences
                                if key.startswith('self_attn.'):
                                    new_state_dict[key] = value
                                elif key.startswith('mlp.'):
                                    new_state_dict[key] = value
                                elif key.startswith('input_layernorm.'):
                                    new_state_dict[key] = value
                                elif key.startswith('post_attention_layernorm.'):
                                    new_state_dict[key] = value
                            layer.load_state_dict(new_state_dict, strict=False)
                        except Exception as e:
                            print(f"Could not load weights for layer {i}: {str(e)}")
                            print("Using initialized weights")
                else:
                    # Use original layer for non-Llama models
                    layer = full_model.model.layers[i]
                self.layers.append(layer)
            
            self.norm = full_model.model.norm
            self.lm_head = full_model.lm_head
            
        else:  # Fallback architecture
            if hasattr(full_model, 'gpt_neox'):  # Pythia model
                total_layers = len(full_model.gpt_neox.layers)
                self.layers = nn.ModuleList([
                    full_model.gpt_neox.layers[i]
                    for i in range(min(start_layer, total_layers), total_layers)
                ])
                self.norm = full_model.gpt_neox.final_layer_norm
                self.lm_head = full_model.embed_out
            else:  # Generic transformer
                total_layers = len(full_model.transformer.h)
                self.layers = nn.ModuleList([
                    full_model.transformer.h[i]
                    for i in range(min(start_layer, total_layers), total_layers)
                ])
                self.norm = full_model.transformer.ln_f
                self.lm_head = full_model.lm_head
        
        self.actual_num_layers = len(self.layers)
        
        # Delete full model to save memory
        del full_model
        torch.cuda.empty_cache()
        
        logger.info({
            "model/server_layers": self.actual_num_layers,
            "model/server_start": start_layer,
            "model/server_model": model_name
        })
        
        print(f"Server model initialized with {self.actual_num_layers} layers (from layer {start_layer})")
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """Create causal attention mask for decoder"""
        # Create causal mask
        batch_size, seq_length = input_shape
        mask = torch.full((seq_length, seq_length), torch.finfo(inputs_embeds.dtype).min, device=inputs_embeds.device)
        mask_cond = torch.arange(mask.size(-1), device=inputs_embeds.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(inputs_embeds.dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(seq_length, past_key_values_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device), mask], dim=-1)
        
        # Expand to batch size and add head dimension
        mask = mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length + past_key_values_length)
        
        # Combine with attention mask
        if attention_mask is not None:
            # Expand attention mask
            if attention_mask.dim() == 2:
                expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length)
                # Convert to float and apply large negative value where mask is 0
                expanded_mask = expanded_mask.to(dtype=inputs_embeds.dtype)
                expanded_mask = (1.0 - expanded_mask) * torch.finfo(inputs_embeds.dtype).min
                mask = mask + expanded_mask

        return mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, None, None]:
        """
        Forward pass through server layers
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Prepare attention mask for server layers
        if "llama" in self.model_name.lower() and attention_mask is not None:
            # Create causal mask if not already provided
            if attention_mask.dim() == 2:
                attention_mask = self._prepare_decoder_attention_mask(
                    attention_mask, (batch_size, seq_length), hidden_states, 0
                )
        
        # Pass through server layers
        for layer in self.layers:
            if "llama" in self.model_name.lower() or hasattr(layer, 'self_attn'):
                # Llama-style layer
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                hidden_states = layer_outputs[0]
            else:
                # Generic transformer layer
                outputs = layer(hidden_states, attention_mask=attention_mask)
                hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        return logits, None, None
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100
    ) -> torch.Tensor:
        """
        Compute language modeling loss
        """
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten and compute loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
    
    def get_num_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SplitLLaMA3Model:
    """
    Manager class for split Llama 3.2 model architecture
    """
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B", client_layers: int = 4):
        self.model_name = model_name
        self.client_layers = client_layers
        
        print(f"\nInitializing Split Model Architecture")
        print(f"Model: {model_name}")
        print(f"Client layers: {client_layers}")
        
        # Initialize tokenizer
        try:
            hf_token = os.environ.get("HF_TOKEN", None)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token
            )
        except Exception as e:
            print(f"Could not load tokenizer for {model_name}, using fallback")
            # Use a fallback tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load configuration
        try:
            hf_token = os.environ.get("HF_TOKEN", None)
            if "meta-llama/Llama-3.2" in model_name:
                self.config = LlamaConfig.from_pretrained(model_name, token=hf_token)
            else:
                self.config = AutoConfig.from_pretrained(model_name, token=hf_token)
            self.total_layers = self.config.num_hidden_layers
        except:
            print("Using default configuration")
            self.total_layers = 16  # Default for small models
        
        # Ensure we don't exceed available layers
        self.client_layers = min(client_layers, max(1, self.total_layers - 2))
        
        logger.info({
            "model/name": model_name,
            "model/total_layers": self.total_layers,
            "model/client_layers": self.client_layers,
            "model/server_layers": self.total_layers - self.client_layers
        })
        
        print(f"Total layers: {self.total_layers}")
        print(f"Client will use layers 0-{self.client_layers-1}")
        print(f"Server will use layers {self.client_layers}-{self.total_layers-1}")
    
    def create_client_model(self) -> ClientModel:
        """Create and return client model"""
        return ClientModel(self.model_name, self.client_layers)
    
    def create_server_model(self) -> ServerModel:
        """Create and return server model"""
        return ServerModel(self.model_name, self.client_layers)
    
    def validate_split(self, client_model: ClientModel, server_model: ServerModel) -> bool:
        """Validate the split architecture"""
        client_params = client_model.get_num_parameters()
        server_params = server_model.get_num_parameters()
        total_params = client_params + server_params
        
        print(f"\nModel Split Validation:")
        print(f"Client parameters: {client_params:,}")
        print(f"Server parameters: {server_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Client/Server ratio: {client_params/total_params:.1%}/{server_params/total_params:.1%}")
        
        logger.info({
            "model/client_parameters": client_params,
            "model/server_parameters": server_params,
            "model/total_parameters": total_params,
            "model/client_param_ratio": client_params/total_params,
            "model/server_param_ratio": server_params/total_params
        })
        
        return True
