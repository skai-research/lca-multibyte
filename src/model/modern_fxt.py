import torch
import torch.nn.functional as F
import itertools
import math
import copy
from typing import Tuple, List

import torch.nn as nn
from src.model.shortening import downsample, upsample
from src.model.utils import compute_mean_with_padding, apply_rotary_pos_embd, freeze_module, unfreeze_module, RotaryEmbedding, RMSNorm


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, group_ids):
        """
        Args:
            group_ids: (B, T) long tensor of group ids e.g. [0,0,0,1,1,1,2,...]
        Returns:
            (B, T, demb) sinusoidal encoding of intra-group position
        """
        B, T = group_ids.shape
        pos = torch.arange(T, device=group_ids.device).unsqueeze(0)
        is_start = F.pad(group_ids[:, 1:] != group_ids[:, :-1], (1, 0), value=True)
        intra_idx = pos - (pos * is_start).cummax(1).values          # (B, T) → [0,1,2, 0,1,2,3, ...]
        sinusoid = torch.ger(intra_idx.float().flatten(), self.inv_freq)
        enc = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)
        return enc.view(B, T, -1)



# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L214
# https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L382
class LanguageModelGroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention (GQA) as used in some transformer-based language models.

    GQA reduces computation by using fewer key-value heads than query heads,
    grouping multiple query heads to share the same key-value heads.

    Args:
        cfg: Configuration object containing:
            - lm_n_heads (int): Number of query heads.
            - lm_n_kv_heads (int): Number of key-value heads.
            - lm_hidden_dim (int): Hidden embedding dimension.
            - lm_dropout (float): Dropout rate.
    """
    def __init__(self, cfg):
        super().__init__()

        self.n_heads = cfg.lm_n_heads
        self.n_kv_heads = cfg.lm_n_kv_heads
        self.embd_dim = cfg.lm_hidden_dim
        self.dropout = cfg.lm_dropout

        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by num_heads"

        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = self.embd_dim // self.n_heads

        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.k_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.v_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Use scaled dot product attention if available
        self.sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.sdpa:
            print("Warning: scaled dot product attention not available, using standard attention in LM.")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask=None, block_kv_cache=None) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass for grouped query attention.

        Args:
            x (Tensor): Input tensor of shape (B, T_curr, C), where
                        B = batch size,
                        T_curr = current sequence length,
                        C = embedding dimension.
            cos (Tensor): Rotary embedding cosines, shape compatible with q and k.
            sin (Tensor): Rotary embedding sines, shape compatible with q and k.
            attention_mask (Tensor, optional): Attention mask tensor of shape (B, total_kv_length),
                                               with 1 for tokens to attend to and 0 for padding.
            block_kv_cache (dict, optional): Cache dict with 'key' and 'value' tensors for autoregressive decoding.

        Returns:
            tuple[Tensor, dict]:
                - Output tensor after attention and projection, shape (B, T_curr, C).
                - Updated block_kv_cache dict for caching key-value states.
        """
        is_prefill = block_kv_cache is None
        # breakpoint() # Check if we are in prefill (no cache) or decode (with cache) mode

        B, T_curr, C = x.size() # T_curr is the sequence length of the current input x
        q_curr = self.q_proj(x).view(B, T_curr, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T_curr, head_dim)
        k_curr = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv_heads, T_curr, head_dim)
        v_curr = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv_heads, T_curr, head_dim)

        # Apply rotary embeddings to the current q and k
        q, k_rotated = apply_rotary_pos_embd(q_curr, k_curr, cos, sin)

        # Check if we can use cached keys and values
        if not is_prefill and block_kv_cache['key'] is not None:
            # Concatenate with cached K, V
            # k_rotated and v_curr are for the new token(s)
            k = block_kv_cache['key']
            v = block_kv_cache['value']
            k = torch.cat([k, k_rotated], dim=2)
            v = torch.cat([v, v_curr], dim=2)
            block_kv_cache['key'] = k
            block_kv_cache['value'] = v
        else:
            # No cache, this is the first pass (prefill)
            k = k_rotated
            v = v_curr
            block_kv_cache = {'key': k, 'value': v}

        # Repeat K, V for Grouped Query Attention
        k_exp = k.repeat_interleave(self.n_kv_groups, dim=1) # (B, n_heads, T_kv, head_dim)
        v_exp = v.repeat_interleave(self.n_kv_groups, dim=1) # (B, n_heads, T_kv, head_dim)
        
        T_kv = k_exp.size(2) # Total sequence length of keys/values

        # Prepare attention mask for SDPA or manual path
        # attention_mask is (B, T_kv_total_length), 1 for attend, 0 for pad
        additive_attn_mask = None
        if attention_mask is not None:
            additive_attn_mask = attention_mask

            # # The current `attention_mask` parameter is assumed to be `[B, total_sequence_length_kv]`
            # # Let's make it `[B, 1, 1, T_kv]` for SDPA.
            # mask_for_keys = attention_mask[:, :T_kv] # Ensure mask matches key length [B, T_kv]
            # additive_attn_mask = (1.0 - mask_for_keys.unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min
            # This additive_attn_mask shape is [B, 1, 1, T_kv]
        if self.sdpa and x.device.type != 'mps':
            # During decode, no additional masking needed as [1, T_kv] is naturally causal
            # When an explicit attn_mask is provided, is_causal must be False (PyTorch SDPA constraint)
            is_causal = (T_curr == T_kv and T_curr > 1) #and additive_attn_mask is None
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k_exp, v_exp,
                attn_mask=additive_attn_mask, 
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal
            )
        else:
            # Manual attention implementation
            attn = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(self.head_dim) # (B, n_heads, T_curr, T_kv)
            # During decode: no additional masking needed as [1, T_kv] is naturally causal
            if T_curr == T_kv and T_curr > 1:
                causal_mask_val = torch.tril(torch.ones(T_curr, T_curr, device=x.device, dtype=torch.bool)).view(1, 1, T_curr, T_curr)
                attn = attn.masked_fill(~causal_mask_val, float('-inf'))

            if additive_attn_mask is not None: # Additive padding mask
                # additive_attn_mask is [B,1,1,T_kv], needs to be broadcast to [B, n_heads, T_curr, T_kv]
                attn = attn + additive_attn_mask 

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v_exp
            
        y = y.transpose(1, 2).contiguous().view(B, T_curr, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y, block_kv_cache

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L160
class LanguageModelMLP(nn.Module):
    """
    Implements the feed-forward network (MLP) block used in transformer-based language models.

    This MLP uses a gated activation mechanism where two separate linear projections
    are applied to the input: one passed through an activation function (gate_proj),
    and the other as is (up_proj). Their element-wise product is then projected back
    to the embedding dimension (down_proj).

    Args:
        cfg: Configuration object containing:
            - lm_hidden_dim (int): The embedding dimension size.
            - lm_inter_dim (int): The intermediate dimension size for the MLP.

    Attributes:
        activation_fn (Callable): The activation function used (SiLU).
        gate_proj (nn.Linear): Linear projection for gating pathway.
        up_proj (nn.Linear): Linear projection for upscaling pathway.
        down_proj (nn.Linear): Linear projection for downscaling back to embedding dim.
    """

    def __init__(self, cfg):
        super().__init__()
        self.embd_dim = cfg.lm_hidden_dim
        self.inter_dim = cfg.lm_inter_dim

        self.activation_fn = nn.SiLU()
        self.gate_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.up_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.down_proj = nn.Linear(self.inter_dim, self.embd_dim, bias=False)

    def forward(self, x):
        """
        Forward pass through the gated MLP block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embd_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, embd_dim),
                    after gated MLP transformation.
        """
        gate = self.activation_fn(self.gate_proj(x))
        x = self.up_proj(x)
        x = self.down_proj(gate * x)

        return x

# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L222
class LanguageModelBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = LanguageModelMLP(cfg)
        self.attn = LanguageModelGroupedQueryAttention(cfg)
        self.norm1 = RMSNorm(cfg) # Input Norm
        self.norm2 = RMSNorm(cfg) # Post Attention Norm
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: torch.Tensor=None, block_kv_cache: dict=None):
        """
        Forward pass of the Transformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            cos (Tensor): Cosine positional embeddings for rotary embedding, shape
                matching sequence length and head dimension.
            sin (Tensor): Sine positional embeddings for rotary embedding, same shape as cos.
            attention_mask (Tensor, optional): Attention mask of shape (batch_size, total_kv_length),
                with 1 indicating tokens to attend to and 0 for padding tokens.
            block_kv_cache (dict, optional): Key-value cache dict for cached keys and values
                during decoding. If None, no cache is used.

        Returns:
            Tuple[Tensor, dict]: Output tensor after the block (same shape as input),
                and the updated key-value cache dictionary.
        """
        res = x
        x = self.norm1(x)
        x, block_kv_cache = self.attn(x, cos, sin, attention_mask, block_kv_cache)
        x = res + x

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = res + x

        return x, block_kv_cache


class BoundaryPredictor(nn.Module):
    def __init__(self, d_model, d_inner,  activation_function,
                 temp, use_binomial, s_lower_bound, bp_type,   threshold=0.5):
        super().__init__()
        self.temp = temp
        self.use_binomial = use_binomial
        self.s_lower_bound = s_lower_bound
        self.bp_type = bp_type
        self.threshold = threshold

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = nn.GELU()
        elif activation_function == 'silu':
            activation_fn = nn.SiLU()

        self.boundary_predictor = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,

            nn.Linear(d_inner, 1),
        )

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, hidden, prior=None):
        # Hidden is of shape [seq_len x bs x d_model]
        # Boundaries we return are [bs x seq_len]
        
        self.priors = prior
        self.pred_prior = torch.tensor([p[0] for p in self.priors], device=hidden.device)
        boundary_logits = self.boundary_predictor(hidden).squeeze(-1).transpose(0, 1)
        boundary_probs = torch.sigmoid(boundary_logits)
        if self.bp_type == 'gumbel':
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp,
                probs=boundary_probs,
            )
            soft_boundaries = bernoulli.rsample()

            hard_boundaries = (soft_boundaries > self.threshold).float()
            hard_boundaries = (
                hard_boundaries - soft_boundaries.detach() + soft_boundaries
            )
        elif self.bp_type in ['entropy', 'unigram']:
            soft_boundaries = boundary_probs
            hard_boundaries = (soft_boundaries > self.threshold).float()

        self.soft_boundaries = soft_boundaries
        self.hard_boundaries = hard_boundaries

        return soft_boundaries, hard_boundaries

    def infer_boundaries(self, hidden):
        with torch.no_grad():
            boundary_logits = self.boundary_predictor(hidden).squeeze(-1).transpose(0, 1)
            boundary_probs = torch.sigmoid(boundary_logits)
            hard_boundaries = (boundary_probs > self.threshold).float()
        return hard_boundaries

    def calc_loss_without_padding(self, preds, gt, attention_mask=None):
        """

        """
        # B x T
        if self.bp_type in ['entropy', 'unigram']:
            assert preds is not None and gt is not None
            return self.loss(preds, gt.float())

        elif self.bp_type in ['gumbel']:
            if attention_mask is not None and gt is None:

                # create a mask based on attention_mask
                mask = attention_mask.eq(1)  # Mask is True where tokens are present, False for padding

                # apply the mask to predictions
                masked_preds = preds * mask.float()
                sum_preds = masked_preds.sum(dim=-1).unsqueeze(dim=-1)

                # Compute the total count of trials for each example in the batch
                total_count = mask.sum(dim=-1, keepdim=True).float()  # Number of non-padded tokens

            else:
                total_count = preds.size(-1)
                sum_preds = preds.sum(dim=-1)
            
            est_prior = sum_preds / total_count
            prior_std = torch.tensor([p[1] for p in self.priors], device=self.pred_prior.device)
            
            upper_bound = self.pred_prior 
            lower_bound = self.pred_prior - self.s_lower_bound * prior_std
            
            # Calculate losses with smoother transitions
            loss_high = torch.clamp(est_prior - upper_bound, min=0.0)
            loss_low = torch.clamp(lower_bound - est_prior, min=0.0)
            # If both losses are near zero, use simple mean
            loss_boundaries = (loss_high + loss_low).mean()

            return loss_boundaries, self.pred_prior


    


class FxTTransformerLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        for key, value in vars(cfg).items():
            setattr(self, key, value)


        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)

        self.rotary_embd = RotaryEmbedding(cfg)
        
        self.norm = RMSNorm(cfg) 
        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight

        self.crit = torch.nn.CrossEntropyLoss(ignore_index= -100)

            # when loading the pretrained config, the keys become strings instead of int, so we convert to int here
        are_all_script_keys_string = all(isinstance(value, str) for value in self.id_to_script.keys())
        if are_all_script_keys_string:
            self.id_to_script = {int(key): value for key, value in self.id_to_script.items() if key.isdigit()}


        pre_layers, (shortened_layers, ), post_layers, mtp_layers = eval(self.model_config)
        self.is_bp = self.boundaries_type in ['unigram', 'entropy', 'gumbel', 'fixed']

        if post_layers == 0 and shortened_layers == 0:
            assert self.boundaries_type == 'none'
            self.blocks = nn.ModuleList([nn.ModuleList([
                LanguageModelBlock(cfg) for _ in range(pre_layers)
            ])])
        else:
            self.null_group = nn.Parameter(torch.Tensor(1, 1, self.lm_hidden_dim).zero_())
            nn.init.normal_(self.null_group)

            decoder_stacks = [
                self.create_decoder_layers(pre_layers),
                self.create_decoder_layers(shortened_layers),
                self.create_decoder_layers(post_layers),
            ]
            if mtp_layers > 0:
                decoder_stacks.append(self.create_decoder_layers(mtp_layers))

            self.blocks = nn.ModuleList(decoder_stacks)

            # Create boundary predictor layers
            if self.is_bp:
                self.script_to_bp_layers = nn.ModuleDict({script: BoundaryPredictor(
                    d_model=self.lm_hidden_dim,
                    d_inner=self.lm_inter_dim,
                    activation_function=self.activation_function,
                    temp=self.temp,
                    use_binomial=self.use_binomial,
                    s_lower_bound=self.s_lower_bound,
                    bp_type=self.boundaries_type,
                )
                for i, (script, pri) in  itertools.islice(enumerate(zip(self.all_script_ids_dict.keys(), self.all_script_ids_dict.values())), self.num_predictors) # itertools.islice is used to limit the number of predictors
                })

                self.spikes_left = self.spikes_left
        
        self.apply(self._init_weights)

    def create_decoder_layers(self, n_layers):
        layers = nn.ModuleList([
            LanguageModelBlock(self.cfg) for _ in range(n_layers)
        ])
        return layers

    def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Parameter):
                torch.nn.init.normal_(module, mean=0.0, std=0.02)

    def group_causal_mask(self, groups: torch.LongTensor) -> torch.BoolTensor:
        """
        A tokens attends itself, all tokens in previous groups, and the first token in its own group.
        groups: (B, T) group ids (e.g. 0,0,0,1,1,1,2,...), contiguous per group
        returns: allowed mask (B, T, T) where mask[b,i,j]=True iff token i may attend to token j
        """
        B, T = groups.shape
        device = groups.device

        # first token in each group
        first = torch.ones_like(groups, dtype=torch.bool, device=device)
        if T > 1:
            first[:, 1:] = groups[:, 1:] != groups[:, :-1]

        # broadcasted group labels
        g_i = groups.unsqueeze(2)   # (B, T, 1)
        g_j = groups.unsqueeze(1)   # (B, 1, T)
        first_j = first.unsqueeze(1) 

        idx = torch.arange(T, device=device)
        i_idx = idx.view(1, T, 1)
        j_idx = idx.view(1, 1, T)
        self_pos = (i_idx == j_idx)  # (1, T, T) -> will broadcast to (B,T,T)

        if self.attn_type == "previous":
            allowed = (g_j < g_i)
        elif self.attn_type == "self":
            allowed = self_pos
        elif self.attn_type == "previous_self":
            allowed = (g_j < g_i) | self_pos
        elif self.attn_type == "previous_same":
            allowed = (g_j < g_i) | (g_j == g_i)
        elif self.attn_type == "prev_group_self":
            allowed = self_pos | (g_j == g_i - 1) | ((g_j == g_i) & (j_idx < i_idx))
        elif self.attn_type == "old_impl":
            allowed = (g_j < g_i) | ((g_j == g_i) & (first_j | self_pos))
        else:
            raise NotImplementedError(f"Attention type {self.attn_type} not supported.")


        return allowed


    def _forward(self, x, attention_mask, kv_cache, blocks, position_offset=0):
        # get cos and sin from rotary embedding
        position_ids = torch.arange(
            position_offset, position_offset + x.size(1), device=x.device
        ).unsqueeze(0).expand(x.size(0), -1)
        cos, sin = self.rotary_embd(position_ids)

        for i, block in enumerate(blocks):
            x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])
        return x, kv_cache

    def get_spikes(self, vector):
        total = torch.ones_like(vector).bool()

        for i in range(1, self.spikes_left + 1, 1):
            mask = vector[i:] > vector[:-i]
            total[i:] &= mask

        return total

    def freeze_backbone(self):
        freeze_module(self)
        unfreeze_module(self.blocks[3])

    def freeze_decoder(self):
        freeze_module(self.blocks[3])

    def unfreeze_backbone(self):
        unfreeze_module(self)

    def compute_compression_rate(self, hard_boundaries, attention_mask):
        # Create a mask based on attention_mask
        mask = attention_mask.eq(1)  # Mask is True where tokens are present, False for padding

        # Apply the mask to hard_boundaries
        masked_hard_boundaries = hard_boundaries * mask.float()

        # Compute the total number of non-padded positions for each row in the batch
        num_non_padded_positions_per_row = mask.sum(dim=1).float()  # Count the number of non-padded positions for each row

        # Compute the sum of predictions only on non-padded positions for each row in the batch
        sum_hard_boundaries_non_padded_per_row = masked_hard_boundaries.sum(dim=1)  # Sum of hard_boundaries for each row

        # Compute the compression_rate only on non-padded positions for each row in the batch
        zero_mask = sum_hard_boundaries_non_padded_per_row.eq(0)
        if zero_mask.any():
            sum_hard_boundaries_non_padded_per_row[zero_mask] = 1
        compression = (num_non_padded_positions_per_row / sum_hard_boundaries_non_padded_per_row)
        compression_rate = compression.mean()
        compression_variance = compression.var()
        #change the NaN values in compression_variance to 0
        compression_variance = torch.nan_to_num(compression_variance, nan=0.0)
        compression_std = compression.std()

        p_ones = (sum_hard_boundaries_non_padded_per_row / num_non_padded_positions_per_row).mean()
        

        return (compression_rate, compression_variance, compression_std), p_ones


    def compute_boundaries_in_parallel(self, hidden, target, cos, sin, attention_mask, kv_cache, boundary_predictor, priors):

        embeddings = hidden.clone() 
        residual = None
        pre_upsample = None
        shortened_length = None
        soft_boundaries = None
        hard_boundaries = None   
        group_ids = None
        for i in range(len(self.blocks)):
            if i == 0:  
                hidden1, kv_cache[i] = self._forward(
                    hidden, attention_mask, kv_cache[i],
                    blocks=self.blocks[i]
                )
                hidden2 = hidden1.clone() # B x T x C
                residual1 = hidden1.clone() # B x T x C
            if i == 1:  # Downsampling
                bp_input = hidden1.permute(1, 0, 2)  # B x T x C -> T x B x C
                soft_boundaries, hard_boundaries = boundary_predictor(bp_input, prior=priors)                     # B x T
                hidden = downsample(
                    boundaries=hard_boundaries,
                    hidden=hidden1.permute(1, 0, 2),  # T x B x C
                    null_group=self.null_group,
                ).permute(1, 0, 2)  # B x S x C
                
                shortened_length = hidden.size(1) 

                if self.use_group_attn:
                    just_boundaries = hard_boundaries.detach().long()            # (B, T)
                    group_ids = just_boundaries.cumsum(dim=1) - just_boundaries
                  
                
                    residual2 = hidden.clone() # B X S X C
                
                hidden, kv_cache[i] = self._forward(
                    hidden, attention_mask, kv_cache[i],
                    blocks=self.blocks[i]
                )
                hidden = upsample(
                    boundaries=hard_boundaries,
                    shortened_hidden=hidden.permute(1, 0, 2),
                ).permute(1, 0, 2)  # B x T x C
                hidden1, hidden2 = hidden, hidden  # Initialize both outputs

            elif i == 2: 
                hidden1 = hidden1 + residual1

                hidden1, kv_cache[i] = self._forward(
                    hidden1, attention_mask, kv_cache[i],
                    blocks=self.blocks[i]
                )

            if i == 3:  # MTP stage
                residual2 = upsample(
                    boundaries=hard_boundaries,
                    shortened_hidden=residual2.permute(1, 0, 2),
                ).permute(1, 0, 2)
                hidden2 = hidden2 + residual2

                attention_mask = self.group_causal_mask(group_ids).unsqueeze(1)  # (B, 1, T, T)
                hidden2, kv_cache[i] = self._forward(
                    hidden2, attention_mask, kv_cache[i],
                    blocks=self.blocks[i]
                )
        return hidden1, hidden2, target, shortened_length, soft_boundaries, hard_boundaries, group_ids



    def forward(self, batch, task, kv_cache: List[dict]=None, start_pos: int=0, script_id: int=None):

        """
        Data: Batch Size x Sequence length  --> Sequence length x Batch Size
        Attention_mask: Batch Size x Sequence length  --> Batch Size x Sequence length
        """
        self.task = task        
        #note that batch['attention_mask'] has been taken down since I only need it during finetuning
        # In each batch, get all the unique script ids and check that they are contained in script ids
        batch_scripts = [self.id_to_script[script_id]] * batch["input_ids"].size(0)
        batch_priors = [self.all_script_ids_dict[script_id] for script_id in batch_scripts]
        assert all(value in self.id_to_script.keys() for value in [script_id])
        overall_stats = {}


  
        target_ids = batch["input_ids"].clone()
        tgt_len =  target_ids.size(-1)

        hidden = self.token_embedding(batch["input_ids"])  # B x T x C

        B, T_curr, _ = hidden.size()

        # Create position_ids for the current sequence based on start_pos
        current_position_ids = torch.arange(start_pos, start_pos + T_curr, device=batch['input_ids'].device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_embd(current_position_ids) # Get rotary position embeddings for current tokens
        # (Tokenization happens here) Downsample and upsample representations
        if self.is_bp:
            available_bp_id = list(self.script_to_bp_layers.keys())[0]
            boundary_predictor = self.script_to_bp_layers[available_bp_id]
        else:
            boundary_predictor = None
        if kv_cache is None:
            kv_cache = [[None] * len(self.blocks[i]) for i in range(len(self.blocks))]

        hidden1, hidden2, target_ids, shortened_length, soft_boundaries, hard_boundaries, group_ids = self.compute_boundaries_in_parallel(hidden=hidden,
                        cos=cos,
                        sin=sin,
                        attention_mask=None,
                        kv_cache=kv_cache,
                        target=target_ids,
                        boundary_predictor=boundary_predictor,
                        priors=batch_priors
                        )
        hidden1 = self.norm(hidden1)
        hidden2 = self.norm(hidden2)

        loss_boundaries = torch.tensor(0.0, device=batch["input_ids"].device)

        if self.is_bp:
            # Calculate boundary loss here
            soft_boundaries = soft_boundaries[:, -tgt_len:]
            hard_boundaries = hard_boundaries[:, -tgt_len:]
            if task == "LM":
                loss_boundaries, pred_priors = self.script_to_bp_layers[available_bp_id].calc_loss_without_padding(
                                preds=hard_boundaries, gt=None, attention_mask=None
                            )
            else:
                # check the shape of the attention mask
                loss_boundaries, pred_priors = self.script_to_bp_layers[available_bp_id].calc_loss_without_padding(preds=hard_boundaries, gt=None, attention_mask=batch["attention_mask"])

            attention_mask = batch['attention_mask']
            script_compression_rate, script_p_ones = self.compute_compression_rate(hard_boundaries, attention_mask)
            script_indexer = self.id_to_script[script_id].strip("<>")
            overall_stats[f"{script_indexer}_compression_rate"] = script_compression_rate[0].item()
            overall_stats[f"{script_indexer}_compression_var"] = script_compression_rate[1].item()
            overall_stats[f"{script_indexer}_p_ones"] = script_p_ones.item()
            overall_stats['shortened_length'] = shortened_length
            overall_stats['bp_loss'] = loss_boundaries.item()



        logits = self.head(hidden1)
        logits2 = self.head(hidden2)


        loss_boundaries = loss_boundaries.reshape(-1)
        device = hidden1.device
        mtp_loss = torch.tensor(0.0, device=device)
        valid_heads = 0
        attention_mask = batch.get("attention_mask", None)


        if task == "SFT" and 'labels' in batch:
            target_len_no_pad = batch["attention_mask"].sum(dim=1)  
            label_len_no_pad = (batch['labels'] != 0).sum(dim=1)  
            prompt_len = batch['prompt_len']

            # Create mask: [B, T] where True = prompt positions to ignore
            seq_len = target_ids.size(1)  
            seq_indices = torch.arange(seq_len, device=target_ids.device).unsqueeze(0)  # [1, T]
            prompt_mask = seq_indices < prompt_len.unsqueeze(1)  # [B, T]
            
            target_ids = target_ids.clone()  # [B, T]
            target_ids[prompt_mask] = self.ignore_index
            target_ids[batch["attention_mask"] == 0] = self.ignore_index          

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        loss = self.crit(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))   

        if self.use_group_attn:    


            # shift=2 for future token predictionn.
            logits_pred   = logits2[:, :-2, :].contiguous()     # (B, T-1, V)
            targets_next  = target_ids[:, 2:].contiguous() # (B, T-1)
            vocab_dim = logits_pred.size(-1)
            logits_flat  = logits_pred.reshape(-1, vocab_dim)
            targets_flat = targets_next.reshape(-1)


            mtp_loss = self.crit(logits_flat, targets_flat)
            
        if task == "LM" or task == "SFT_eval" or task=="SFT":
            return loss, mtp_loss, overall_stats, loss_boundaries, logits 
        else:
            if task == "tokenization2":
                overall_stats['hard_boundaries'] = hard_boundaries #if I want to the boundaries
                overall_stats['priors'] = pred_priors.tolist()
            return hidden, overall_stats, loss_boundaries
    
    
    @torch.inference_mode()
    def generate(self, input_ids, max_tokens=100, temperature=1.0, top_k=None,
                 top_p=None, stop_token_id=None, repetition_penalty=1.0, threshold=0.9, candidates=1, use_caching=False, speculative=False, drafter=None):  

        # keyword args: generate_verify* order candidates/threshold the opposite
        # way round to generate_group*, so positional passing transposes them.
        common = dict(max_tokens=max_tokens, temperature=temperature, top_k=top_k,
                      top_p=top_p, stop_token_id=stop_token_id,
                      repetition_penalty=repetition_penalty,
                      threshold=threshold, candidates=candidates)

        if speculative:
            if drafter is not None:
                return self.generate_verify_with_fxt(input_ids, fxt_model=drafter, **common)
            else:
                return self.generate_verify(input_ids, **common)

        if use_caching:
            return self.generate_group_cached(input_ids, **common)
        else:
            return self.generate_group(input_ids, **common)



    @torch.inference_mode()
    def generate_group_cached(self, input_ids, max_tokens=100, temperature=1.0, top_k=None,
                        top_p=None, stop_token_id=None, repetition_penalty=1.0, threshold=0.9, candidates=1):
        
        """Use threshold verifier and KV caching"""
        if self.is_bp:
            available_bp_id = list(self.script_to_bp_layers.keys())[0]
            boundary_predictor = self.script_to_bp_layers[available_bp_id]
        out_ids = input_ids.clone()
        hidden_down = None
        b = 0
        confidence_threshold = threshold
        accepted_tokens = 0
        pred_group = 0
        main_hard_boundaries = None
        active_boundaries = None
        acceptance_rates = []

        # --- Stage 0 cache (token-level, fully causal: always safe to extend) ---
        stage0_kv_cache = None
        stage0_cached_hidden1 = None  # (B, T_committed, C)

        # --- Stage 1 cache (group-level, causal over shortened seq) ---
        # Only groups that have closed are committed. The null group (shortened
        # position 0) is always committed once seen.
        stage1_kv_cache = None
        stage1_cached_out = None      # (B, S_committed, C)
        stage1_committed_groups = 0   # number of shortened positions baked in (incl. null)

        # --- Stage 2 cache (token-level over upsampled+residual hidden) ---
        # Only token positions <= last_closed_token_idx are committed.
        stage2_kv_cache = None
        stage2_cached_out = None      # (B, T2_committed, C)
        stage2_committed_tokens = 0

        last_closed_token_idx = -1    # index into out_ids of most recent b==1 token; -1 = none yet

        def _incremental_mask(M, N, device, dtype):
            if N == 0 or M <= 1:
                return None
            allowed = torch.zeros(1, 1, M, N + M, device=device, dtype=torch.bool)
            allowed[..., :N] = True
            allowed[..., N:] = torch.tril(torch.ones(M, M, device=device, dtype=torch.bool))
            mask = torch.zeros(1, 1, M, N + M, device=device, dtype=dtype)
            mask.masked_fill_(~allowed, float('-inf'))
            return mask

        while accepted_tokens < max_tokens:
            # kv_cache only holds the *transient* (uncached) tail per stage; the
            # committed caches above persist across iterations.
            kv_cache = [[None] * len(self.blocks[i]) for i in range(len(self.blocks))]
            attention_mask = None

            # ============ Stage 0: incremental, KV-cached ============
            if stage0_cached_hidden1 is None:
                stage0_input_ids = out_ids
                position_offset = 0
            else:
                committed_len = stage0_cached_hidden1.size(1)
                stage0_input_ids = out_ids[:, committed_len:]
                position_offset = committed_len

            stage0_hidden_in = self.token_embedding(stage0_input_ids)
            stage0_attn_mask = _incremental_mask(
                stage0_input_ids.size(1),
                stage0_cached_hidden1.size(1) if stage0_cached_hidden1 is not None else 0,
                out_ids.device, stage0_hidden_in.dtype,
            )
            kv_cache[0] = stage0_kv_cache if stage0_kv_cache is not None else [None] * len(self.blocks[0])
            stage0_new_hidden, kv_cache[0] = self._forward(
                stage0_hidden_in, stage0_attn_mask, kv_cache[0],
                blocks=self.blocks[0], position_offset=position_offset,
            )
            stage0_kv_cache = kv_cache[0]
            stage0_cached_hidden1 = (
                stage0_new_hidden if stage0_cached_hidden1 is None
                else torch.cat([stage0_cached_hidden1, stage0_new_hidden], dim=1)
            )

            hidden1 = stage0_cached_hidden1
            residual1 = hidden1
            hidden2 = hidden1

            for i in range(1, len(self.blocks)):
                if i == 1:  # Downsampling + stage-1 attn
                    hard_boundaries = boundary_predictor.infer_boundaries(hidden1.transpose(1, 0))
                    b = int(hard_boundaries[0, -1].item())

                    if main_hard_boundaries is None:
                        main_hard_boundaries = hard_boundaries
                        diff = 0
                    else:
                        diff = hard_boundaries.size(1) - main_hard_boundaries.size(1)
                        if diff > 0:
                            main_hard_boundaries = torch.cat(
                                [main_hard_boundaries, hard_boundaries[:, -diff:]], dim=1
                            )

                    if b == 1 or hidden_down is None:

                        active_boundaries = main_hard_boundaries

                        # Full downsample (cheap scatter). Includes null group at index 0.
                        full_down = downsample(
                            boundaries=active_boundaries,
                            hidden=hidden1.permute(1, 0, 2),
                            null_group=self.null_group,
                        ).permute(1, 0, 2)  # (B, S_total, C)
                        S_total = full_down.size(1)

                        if self.use_group_attn:
                            residual_down_full = full_down.clone()

                        # On b==1, all groups (including the just-closed one) are final.
                        # On first step with b==0, only the null group is closed.
                        if b == 1:
                            new_commit_target = S_total
                        else:
                            new_commit_target = 1  # only null group

                        # Extend stage-1 cache incrementally.
                        if stage1_committed_groups < new_commit_target:
                            s1_new_input = full_down[:, stage1_committed_groups:new_commit_target, :]
                            s1_mask = _incremental_mask(
                                s1_new_input.size(1), stage1_committed_groups,
                                full_down.device, full_down.dtype,
                            )
                            kv_cache[1] = (
                                stage1_kv_cache if stage1_kv_cache is not None
                                else [None] * len(self.blocks[1])
                            )
                            s1_new_out, kv_cache[1] = self._forward(
                                s1_new_input, s1_mask, kv_cache[1],
                                blocks=self.blocks[1], position_offset=stage1_committed_groups,
                            )
                            stage1_kv_cache = kv_cache[1]
                            stage1_cached_out = (
                                s1_new_out if stage1_cached_out is None
                                else torch.cat([stage1_cached_out, s1_new_out], dim=1)
                            )
                            stage1_committed_groups = new_commit_target

                        # Tail: still-open group(s). Run on top of committed cache without mutating it.
                        if S_total > stage1_committed_groups:
                            tail_input = full_down[:, stage1_committed_groups:, :]
                            tail_mask = _incremental_mask(
                                tail_input.size(1), stage1_committed_groups,
                                full_down.device, full_down.dtype,
                            )
                            tail_cache = [
                                {'key': blk['key'].clone(), 'value': blk['value'].clone()}
                                for blk in stage1_kv_cache
                            ] if stage1_kv_cache is not None else [None] * len(self.blocks[1])
                            tail_out, _ = self._forward(
                                tail_input, tail_mask, tail_cache,
                                blocks=self.blocks[1], position_offset=stage1_committed_groups,
                            )
                            hidden_down = (
                                tail_out if stage1_cached_out is None
                                else torch.cat([stage1_cached_out, tail_out], dim=1)
                            )
                        else:
                            hidden_down = stage1_cached_out

                        if self.use_group_attn:
                            residual_down = residual_down_full

                        hidden1, hidden2 = hidden_down, hidden_down
                    else:
                        # b == 0 and hidden_down already exists. Reuse; extend
                        # active_boundaries with zeros for newly-appended tokens.
                        if diff > 0:
                            active_boundaries = torch.cat([
                                active_boundaries,
                                torch.zeros((active_boundaries.size(0), diff),
                                            device=active_boundaries.device),
                            ], dim=1)

                elif i == 2:  # Upsample + stage-2 attn
                    pred_group = 0
                    hidden1_full = upsample(
                        boundaries=active_boundaries,
                        shortened_hidden=hidden_down.permute(1, 0, 2),
                    ).permute(1, 0, 2) + residual1
                    T_total = hidden1_full.size(1)

                    # Tokens 0..last_closed_token_idx read only from closed shortened
                    # positions, so their stage-2 input is final.
                    new_commit_target_t2 = last_closed_token_idx + 1

                    if stage2_committed_tokens < new_commit_target_t2:
                        s2_new_input = hidden1_full[:, stage2_committed_tokens:new_commit_target_t2, :]
                        s2_mask = _incremental_mask(
                            s2_new_input.size(1), stage2_committed_tokens,
                            hidden1_full.device, hidden1_full.dtype,
                        )
                        kv_cache[2] = (
                            stage2_kv_cache if stage2_kv_cache is not None
                            else [None] * len(self.blocks[2])
                        )
                        s2_new_out, kv_cache[2] = self._forward(
                            s2_new_input, s2_mask, kv_cache[2],
                            blocks=self.blocks[2], position_offset=stage2_committed_tokens,
                        )
                        stage2_kv_cache = kv_cache[2]
                        stage2_cached_out = (
                            s2_new_out if stage2_cached_out is None
                            else torch.cat([stage2_cached_out, s2_new_out], dim=1)
                        )
                        stage2_committed_tokens = new_commit_target_t2

                    # Tail: tokens in currently-open group.
                    if T_total > stage2_committed_tokens:
                        tail_input = hidden1_full[:, stage2_committed_tokens:, :]
                        tail_mask = _incremental_mask(
                            tail_input.size(1), stage2_committed_tokens,
                            hidden1_full.device, hidden1_full.dtype,
                        )
                        tail_cache = [
                            {'key': blk['key'].clone(), 'value': blk['value'].clone()}
                            for blk in stage2_kv_cache
                        ] if stage2_kv_cache is not None else [None] * len(self.blocks[2])
                        tail_out, _ = self._forward(
                            tail_input, tail_mask, tail_cache,
                            blocks=self.blocks[2], position_offset=stage2_committed_tokens,
                        )
                        hidden1 = (
                            tail_out if stage2_cached_out is None
                            else torch.cat([stage2_cached_out, tail_out], dim=1)
                        )
                    else:
                        hidden1 = stage2_cached_out

                elif i == 3:  # MTP stage (unchanged — small window, recompute each step)
                    pred_group = candidates
                    decoder_hard_boundaries = torch.cat(
                        [active_boundaries,
                        torch.zeros((1, pred_group), device=active_boundaries.device)],
                        dim=1,
                    )
                    hidden_down2 = residual_down + hidden_down
                    hidden2 = upsample(
                        boundaries=decoder_hard_boundaries,
                        shortened_hidden=hidden_down2.permute(1, 0, 2),
                    ).permute(1, 0, 2)
                    ones = decoder_hard_boundaries[0].nonzero(as_tuple=True)[0]
                    prev_seg_start = ones[-2] if len(ones) >= 2 else ones[-1]
                    hidden2_window = hidden2[:, prev_seg_start:, :]
                    new_boundaries = decoder_hard_boundaries[:, prev_seg_start:]
                    group_ids = new_boundaries.cumsum(dim=1) - new_boundaries
                    attention_mask = self.group_causal_mask(group_ids).unsqueeze(1)
                    hidden2, kv_cache[3] = self._forward(
                        hidden2_window, attention_mask, kv_cache[3],
                        blocks=self.blocks[3],
                    )

            pred_group += 1

            # ============ Sampling ============
            if self.use_group_attn:
                all_logits = self.head(
                    self.norm(
                        torch.cat(
                            [hidden1[:, -1:, :], hidden2[:, -pred_group:, :]],
                            dim=1
                        )
                    )
                )
            else:
                all_logits = self.head(
                    self.norm(
                        hidden1[:, -1:, :]
                    )
                )

            if repetition_penalty != 1.0 and accepted_tokens > 0:
                generated = out_ids[0]
                score = all_logits[0, :, generated]
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                all_logits[0, :, generated] = score

            if temperature != 1.0:
                all_logits = all_logits / temperature

            if top_k is not None:
                topk_logits, _ = torch.topk(all_logits, top_k, dim=-1)
                kth_logit = topk_logits[..., -1:]
                all_logits = all_logits.masked_fill(all_logits < kth_logit, float('-inf'))
            elif top_p is not None:
                sorted_logits, sorted_indices = torch.sort(all_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_remove = cumulative_probs > top_p
                sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                sorted_remove[..., 0] = False
                remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_indices, sorted_remove)
                all_logits = all_logits.masked_fill(remove, float('-inf'))

            probs_all = F.softmax(all_logits, dim=-1)
            if temperature != 1.0:
                B_, L_, V_ = probs_all.shape
                sampled = torch.multinomial(probs_all.reshape(-1, V_), 1).reshape(B_, L_)
            else:
                sampled = torch.argmax(all_logits, dim=-1)

            if self.use_group_attn and pred_group > 0:
                spec_probs = probs_all[:, 1:, :]
                spec_ids = sampled[:, 1:]
                pred_probs = spec_probs.gather(-1, spec_ids.unsqueeze(-1)).squeeze(-1)
                below_threshold = pred_probs < confidence_threshold
                if below_threshold.any():
                    first_reject_idx = below_threshold.nonzero(as_tuple=True)[0][0].item()
                    final_next_ids = sampled[:, : 1 + first_reject_idx]
                else:
                    final_next_ids = sampled
            else:
                final_next_ids = sampled[:, :1]

            if accepted_tokens > 9:
                last_tokens = out_ids[0, -9:]
                if (last_tokens[:3] == last_tokens[3:6]).all() and (last_tokens[3:6] == last_tokens[6:9]).all():
                    break

            if stop_token_id is not None and (final_next_ids == stop_token_id).any():
                stop_idx = (final_next_ids == stop_token_id).nonzero(as_tuple=True)[1][0].item()
                final_next_ids = final_next_ids[:, :stop_idx + 1]
                all_logits = all_logits[:, :stop_idx + 1, :]
                end_generation = True
            else:
                end_generation = False

            # Update last_closed_token_idx based on `b` computed BEFORE appending new
            # tokens. The new tokens haven't been through the BP yet; next iter's BP
            # tells us whether any of them close a group.
            if b == 1:
                last_closed_token_idx = out_ids.size(1) - 1

            out_ids = torch.cat([out_ids, final_next_ids], dim=1)
            accepted_tokens += final_next_ids.size(1)
            acceptance_rates.append(final_next_ids.size(1) / all_logits.size(1))

            if end_generation:
                break

        return out_ids.squeeze(0).tolist()[-accepted_tokens:], (
            sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0.0
        )
    
    @torch.inference_mode()
    def generate_group(self, input_ids, max_tokens=100, temperature=1.0, top_k=None,
                        top_p=None, stop_token_id=None, repetition_penalty=1.0, threshold=0.9, candidates=1):
        """Use threshold verifier and no KV caching"""

        if self.is_bp:
            available_bp_id = list(self.script_to_bp_layers.keys())[0]
            boundary_predictor = self.script_to_bp_layers[available_bp_id]
        out_ids = input_ids.clone()
        hidden_up = None
        hidden_down = None
        hard_boundaries = None
        group_ids = None
        b = 0
        confidence_threshold = threshold
        accepted_tokens = 0
        pred_group = 0
        must_accept = True  # since we are in b==1 we must accept the token
        hard_boundaries = None

        main_hard_boundaries = None
        active_boundaries = None  # boundaries consistent with current hidden_down (for upsampling)
        acceptance_rates = []
        count_ones = 1
        while accepted_tokens < max_tokens:
            kv_cache = [[None] * len(self.blocks[i]) for i in range(len(self.blocks))]
            attention_mask = None
            hidden = self.token_embedding(out_ids)

            for i in range(len(self.blocks)):
                if i == 0:
                    embeddings = hidden
                    hidden1, kv_cache[i] = self._forward(
                        hidden, attention_mask, kv_cache[i],
                        blocks=self.blocks[i]
                    )
                    residual1 = hidden1.clone()
                    hidden2 = hidden1.clone() # B x T x C


                elif i == 1:  # Downsampling
                    assert boundary_predictor is not None

                    _, hard_boundaries = boundary_predictor(
                    hidden1.transpose(1, 0), prior=[(0.333, 0.023)]
                    )
                    b = int(hard_boundaries[0, -1].item())
                    count_ones += b
                    if main_hard_boundaries is None:
                        main_hard_boundaries = hard_boundaries
                        diff = 0
                    else:
                        # get difference between main_hard_boundaries and hard_boundaries sum to see if any new boundaries were added
                        diff = hard_boundaries.size(1) - main_hard_boundaries.size(1)
                        if diff > 0:
                            main_hard_boundaries =  torch.cat([main_hard_boundaries, hard_boundaries[:, -diff:]], dim=1) # this is what I used for my NMT task.
                    
                    # cut the current segment
                    if b == 1 or hidden_down is None or b==0:
                        active_boundaries = main_hard_boundaries.clone()

                        hidden_down = downsample(
                            boundaries=active_boundaries,
                            hidden=hidden1.permute(1, 0, 2),  # T x B x C
                            null_group=self.null_group,
                        ).permute(1, 0, 2)  # B x T x C

                        if self.use_group_attn:
                            residual_down = hidden_down.clone()

                        hidden_down, kv_cache[i] = self._forward(
                            hidden_down, attention_mask, kv_cache[i],
                            blocks=self.blocks[i]
                        )  
                        
                        hidden1, hidden2 = hidden_down, hidden_down  # Initialize both outputs
                    else:
                        # b == 0 and hidden_down exists: reuse old hidden_down.
                        # Extend active_boundaries with 0s for the newly added tokens
                        # so that the group count stays consistent with hidden_down.
                        # New tokens are treated as part of the last group.
                        if diff > 0:
                            active_boundaries = torch.cat([
                                active_boundaries,
                                torch.zeros((active_boundaries.size(0), diff), device=active_boundaries.device)
                            ], dim=1)
                          
                elif i == 2:  # Upsampling
                    pred_group = 0
                    hidden1 = upsample(
                            boundaries=active_boundaries,
                            shortened_hidden=hidden_down.permute(1, 0, 2),
                        ).permute(1, 0, 2)  # B x T x C
                    hidden1 = hidden1 + residual1


                    hidden1, kv_cache[i] = self._forward(
                        hidden1, attention_mask, kv_cache[i],
                        blocks=self.blocks[i]
                    )
                elif i == 3:  # MTP stage
                    pred_group = 1 # this in n+1 candidate tokens which == 3
                    decoder_hard_boundaries = active_boundaries.clone()

                    decoder_hard_boundaries = torch.cat([decoder_hard_boundaries, torch.zeros((1,pred_group), device=decoder_hard_boundaries.device)], dim=1)

                    hidden_down2 =   residual_down + hidden_down

                    hidden2 = upsample(
                            boundaries=decoder_hard_boundaries,
                            shortened_hidden=hidden_down2.permute(1, 0, 2),
                        ).permute(1, 0, 2)  # B x T+pred_group x C

                    # Find index where previous segment starts
                    ones = decoder_hard_boundaries[0].nonzero(as_tuple=True)[0]          # [0, 3, 7]
                    prev_seg_start = ones[-2] if len(ones) >= 2 else ones[-1]  # 3
                    hidden2_window = hidden2[:, prev_seg_start:, :]          # B x 7 x C  (positions 3–9)
                    new_boundaries = decoder_hard_boundaries[:, prev_seg_start:]
                    group_ids = new_boundaries.cumsum(dim=1)
                    group_ids -= new_boundaries

                    attention_mask = self.group_causal_mask(group_ids).unsqueeze(1)  # (B, 1, T, T)
                    hidden2, kv_cache[i] = self._forward(
                        hidden2_window, attention_mask, kv_cache[i],
                        blocks=self.blocks[i]
                    )

            pred_group += 1

            
            if self.use_group_attn:
                all_logits = self.head(
                    self.norm(
                        torch.cat(
                            [hidden1[:, -1:, :], hidden2[:, -pred_group:, :]],
                            dim=1
                        )
                    )
                )
            else:
                all_logits = self.head(
                    self.norm(
                        hidden1[:, -1:, :]
                    )
                )

            if repetition_penalty != 1.0 and accepted_tokens > 0:
                generated = out_ids[0]
                score = all_logits[0, :, generated]
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                all_logits[0, :, generated] = score

            if temperature != 1.0:
                all_logits = all_logits / temperature

            if top_k is not None:
                topk_logits, _ = torch.topk(all_logits, top_k, dim=-1)
                kth_logit = topk_logits[..., -1:]
                all_logits = all_logits.masked_fill(all_logits < kth_logit, float('-inf'))
            elif top_p is not None:
                sorted_logits, sorted_indices = torch.sort(all_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_remove = cumulative_probs > top_p
                sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                sorted_remove[..., 0] = False
                remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_indices, sorted_remove)
                all_logits = all_logits.masked_fill(remove, float('-inf'))

            probs_all = F.softmax(all_logits, dim=-1)
            if temperature != 1.0:
                B_, L_, V_ = probs_all.shape
                sampled = torch.multinomial(probs_all.reshape(-1, V_), 1).reshape(B_, L_)
            else:
                sampled = torch.argmax(all_logits, dim=-1)

            if self.use_group_attn and pred_group > 0:
                spec_probs = probs_all[:, 1:, :]
                spec_ids = sampled[:, 1:]
                pred_probs = spec_probs.gather(-1, spec_ids.unsqueeze(-1)).squeeze(-1)
                below_threshold = pred_probs < confidence_threshold
                if below_threshold.any():
                    first_reject_idx = below_threshold.nonzero(as_tuple=True)[0][0].item()
                    final_next_ids = sampled[:, : 1 + first_reject_idx]
                else:
                    final_next_ids = sampled
            else:
                final_next_ids = sampled[:, :1]

            if accepted_tokens > 9:
                last_tokens = out_ids[0, -9:]
                if (last_tokens[:3] == last_tokens[3:6]).all() and (last_tokens[3:6] == last_tokens[6:9]).all():
                    break

            if stop_token_id is not None and (final_next_ids == stop_token_id).any():
                stop_idx = (final_next_ids == stop_token_id).nonzero(as_tuple=True)[1][0].item()
                final_next_ids = final_next_ids[:, :stop_idx + 1]
                all_logits = all_logits[:, :stop_idx + 1, :]
                end_generation = True
            else:
                end_generation = False


            out_ids = torch.cat([out_ids, final_next_ids], dim=1)
            accepted_tokens += final_next_ids.size(1)
            acceptance_rates.append(final_next_ids.size(1) / all_logits.size(1))

            if end_generation:
                break

        return out_ids.squeeze(0).tolist()[-accepted_tokens:], (
            sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0.0
        )


    


    def _generate_full_forward(self, out_ids, candidates, boundary_predictor,
                               need_spec=True):
        """Single uncached full forward over `out_ids` for generation.

        Args:
            need_spec: whether to run the MTP stage (stage 3). The verifier only
                reads `main_logits` (stages 0->1->2), so the verify pass passes
                need_spec=False to skip stage 3 entirely.

        Returns:
            main_logits: (B, T, V) logits from the main (stage-2) head. Position t
                predicts token t+1 — this is the model's exact autoregressive
                next-token distribution and serves as the verifier.
            spec_logits: (B, candidates+1, V) logits from the MTP head over the
                `candidates` appended (open) slots, used to draft future tokens.
                None when need_spec is False.
            active_boundaries: (B, T) hard boundaries used for this pass.
        """

        kv_cache = [[None] * len(self.blocks[i]) for i in range(len(self.blocks))]

        hidden_in = self.token_embedding(out_ids)
        hidden1, _ = self._forward(hidden_in, None, kv_cache[0], blocks=self.blocks[0])
        residual1 = hidden1

    
        active_boundaries = boundary_predictor.infer_boundaries(hidden1.transpose(1, 0))
        full_down = downsample(
            boundaries=active_boundaries,
            hidden=hidden1.permute(1, 0, 2),
            null_group=self.null_group,
        ).permute(1, 0, 2)  # (B, S_total, C), incl. null group at index 0
        residual_down = full_down.clone()
        hidden_down, _ = self._forward(
            full_down, None, kv_cache[1], blocks=self.blocks[1]
        )

        hidden1_full = upsample(
            boundaries=active_boundaries,
            shortened_hidden=hidden_down.permute(1, 0, 2),
        ).permute(1, 0, 2) + residual1
        hidden1, _ = self._forward(
            hidden1_full, None, kv_cache[2], blocks=self.blocks[2]
        )
        main_logits = self.head(self.norm(hidden1))  # (B, T, V)


        if not need_spec:
            return main_logits, None, active_boundaries

       
        decoder_hard_boundaries = torch.cat(
            [active_boundaries,
             torch.zeros((active_boundaries.size(0), candidates),
                         device=active_boundaries.device)],
            dim=1,
        )
        hidden_down2 = residual_down + hidden_down
        hidden2 = upsample(
            boundaries=decoder_hard_boundaries,
            shortened_hidden=hidden_down2.permute(1, 0, 2),
        ).permute(1, 0, 2)  # (B, T + candidates, C)
        group_ids = (decoder_hard_boundaries.cumsum(dim=1)
                     - decoder_hard_boundaries).long()
        attention_mask = self.group_causal_mask(group_ids).unsqueeze(1)
        hidden2, _ = self._forward(
            hidden2, attention_mask, kv_cache[3], blocks=self.blocks[3]
        )
        # The last (candidates+1) MTP positions predict tokens T+1 .. T+candidates+1,
        # i.e. the consecutive continuation after main_logits' next token.
        spec_logits = self.head(self.norm(hidden2[:, -(candidates + 1):, :]))

        return main_logits, spec_logits, active_boundaries

    def _sample_logits(self, logits, context_ids, temperature, top_k, top_p,
                       repetition_penalty):
        """Apply repetition penalty / temperature / top-k / top-p to `logits`
        (B, L, V) and return sampled token ids (B, L). Greedy when temperature==1.

        `context_ids` (B, T) is the already-generated sequence, used only for the
        repetition penalty. Pass None to skip the penalty.
        """
        logits = logits.clone()
        if repetition_penalty != 1.0 and context_ids is not None and context_ids.size(1) > 0:
            generated = context_ids[0]
            score = logits[0, :, generated]
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits[0, :, generated] = score

        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None:
            topk_logits, _ = torch.topk(logits, top_k, dim=-1)
            kth_logit = topk_logits[..., -1:]
            logits = logits.masked_fill(logits < kth_logit, float('-inf'))
        elif top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_indices, sorted_remove)
            logits = logits.masked_fill(remove, float('-inf'))

        if temperature != 1.0:
            probs = F.softmax(logits, dim=-1)
            B_, L_, V_ = probs.shape
            return torch.multinomial(probs.reshape(-1, V_), 1).reshape(B_, L_)
        return torch.argmax(logits, dim=-1)

    def _logits_to_probs(self, logits, context_ids, temperature, top_k, top_p,
                         repetition_penalty):
        """Apply the same repetition-penalty / temperature / top-k / top-p
        transforms as `_sample_logits`, but return the full probability
        distribution (B, L, V) instead of a sampled id.

        Used by speculative sampling, which needs the draft (q) and target (p)
        distributions to compute the accept ratio p(x)/q(x). Mirrors
        `_sample_logits`'s convention that temperature == 1.0 means greedy, so the
        returned distribution is a one-hot point mass at the argmax in that case
        (the post-sampling distribution the accept test compares against).
        """
        logits = logits.clone()
        if repetition_penalty != 1.0 and context_ids is not None and context_ids.size(1) > 0:
            generated = context_ids[0]
            score = logits[0, :, generated]
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits[0, :, generated] = score

        if temperature == 1.0:
            # Greedy convention: collapse to a point mass at the argmax.
            idx = logits.argmax(dim=-1, keepdim=True)
            return torch.zeros_like(logits).scatter_(-1, idx, 1.0)

        logits = logits / temperature

        if top_k is not None:
            topk_logits, _ = torch.topk(logits, top_k, dim=-1)
            kth_logit = topk_logits[..., -1:]
            logits = logits.masked_fill(logits < kth_logit, float('-inf'))
        elif top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_indices, sorted_remove)
            logits = logits.masked_fill(remove, float('-inf'))

        return F.softmax(logits, dim=-1)

    def _speculative_accept(self, draft_ids, q_probs, p_probs):
        """Leviathan et al. (arXiv:2211.17192) speculative sampling.

        Args:
            draft_ids: (B, S) tokens sampled from the draft distribution q.
            q_probs:   (B, S, V) draft distributions used to sample `draft_ids`.
            p_probs:   (B, S+1, V) target distributions; position j verifies
                       `draft_ids[:, j]`, and the trailing position S supplies the
                       bonus token when every draft token is accepted.

        Returns:
            accepted_ids: (1, n+1) accepted draft prefix plus one correction/bonus
                token resampled from the target (residual on rejection).
            n: number of accepted draft tokens.

        Assumes batch size 1, matching the generation loops.
        """
        S = draft_ids.size(1)
        x = draft_ids[0]                                   # (S,)
        qx = q_probs[0].gather(-1, x.unsqueeze(-1)).squeeze(-1)  # q(x)  (S,)
        px = p_probs[0, :S].gather(-1, x.unsqueeze(-1)).squeeze(-1)  # p(x) (S,)

        # Accept token i with prob min(1, p(x_i)/q(x_i)).
        r = torch.rand(S, device=x.device)
        accept = r < torch.clamp(px / qx.clamp_min(1e-12), max=1.0)

        if bool(accept.all()):
            n = S
            # All accepted: bonus token is a clean sample from the target.
            corr = torch.multinomial(p_probs[0, S], 1)
        else:
            n = int((~accept).nonzero(as_tuple=True)[0][0].item())
            # Resample the correction from the normalized residual (p - q)+.
            residual = torch.clamp(p_probs[0, n] - q_probs[0, n], min=0.0)
            total = residual.sum()
            if float(total) <= 1e-12:
                corr = torch.multinomial(p_probs[0, n], 1)
            else:
                corr = torch.multinomial(residual / total, 1)

        accepted_ids = torch.cat([x[:n], corr]).unsqueeze(0)  # (1, n+1)
        return accepted_ids, n


       
    @torch.inference_mode()
    def generate_verify(self, input_ids, max_tokens=100, temperature=1.0, top_k=None,
                        top_p=None, stop_token_id=None, repetition_penalty=1.0, candidates=1, threshold=0.9, drafter=None):
        """Speculative decoding with an explicit verification forward pass.

        Each step does two uncached full forwards (KV caching is intentionally
        skipped here for simplicity):

          1. Draft: the MTP head proposes up to `candidates+1` future tokens on
             top of the main head's immediate next token (K = candidates+2 draft
             tokens, all consecutive).
          2. Verify: the draft tokens are appended and the model is re-run; the
             main (stage-2) head's next-token prediction at each draft position is
             the ground-truth check. We accept the longest matching prefix and add
             one bonus/correction token from the verifier.

        This replaces `generate_group2`'s confidence-threshold acceptance with a
        verifier comparison, so accepted tokens are exactly those the full model
        would have produced autoregressively (exact for greedy / temperature==1).

        Assumes batch size 1, matching `generate_group2`.
        """
        if self.is_bp:
            available_bp_id = list(self.script_to_bp_layers.keys())[0]
            boundary_predictor = self.script_to_bp_layers[available_bp_id]
        out_ids = input_ids.clone()
        hidden_up = None
        hidden_down = None
        hard_boundaries = None
        group_ids = None
        b = 0
        confidence_threshold = threshold
        accepted_tokens = 0
        pred_group = 0
        must_accept = True  # since we are in b==1 we must accept the token
        hard_boundaries = None
        # print(confidence_threshold)

        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        main_hard_boundaries = None
        active_boundaries = None  # boundaries consistent with current hidden_down (for upsampling)
        acceptance_rates = []
        count_ones = 1
        while accepted_tokens < max_tokens:
            # Reset KV cache and attention mask each iteration since we reprocess the full sequence
            kv_cache = [[None] * len(self.blocks[i]) for i in range(len(self.blocks))]
            attention_mask = None
            hidden = self.token_embedding(out_ids)

            for i in range(len(self.blocks)):
                if i == 0:
                    embeddings = hidden
                    hidden1, kv_cache[i] = self._forward(
                        hidden, attention_mask, kv_cache[i],
                        blocks=self.blocks[i]
                    )
                    residual1 = hidden1.clone()
                    hidden2 = hidden1.clone() # B x T x C


                elif i == 1:  # Downsampling
                    assert boundary_predictor is not None
                    # hard_boundaries = boundary_predictor.infer_boundaries(hidden1.transpose(1, 0))
                    # b = int(hard_boundaries[0, -1].item())
                    _, hard_boundaries = boundary_predictor(
                    hidden1.transpose(1, 0), prior=[(0.333, 0.023)]
                    )
                    b = int(hard_boundaries[0, -1].item())
                    count_ones += b
                    if main_hard_boundaries is None:
                        main_hard_boundaries = hard_boundaries
                    else:
                        # get difference between main_hard_boundaries and hard_boundaries sum to see if any new boundaries were added
                        diff = hard_boundaries.size(1) - main_hard_boundaries.size(1)
                        if diff > 0:
                            main_hard_boundaries =  torch.cat([main_hard_boundaries, hard_boundaries[:, -diff:]], dim=1) # this is what I used for my NMT task.
                    
                    # cut the current segment

                    active_boundaries = main_hard_boundaries.clone()

                    hidden_down = downsample(
                        boundaries=active_boundaries,
                        hidden=hidden1.permute(1, 0, 2),  # T x B x C
                        null_group=self.null_group,
                    ).permute(1, 0, 2)  # B x T x C

                    if self.use_group_attn:
                        residual_down = hidden_down.clone()

                    hidden_down, _ = self._forward(
                        hidden_down, attention_mask, kv_cache[i],
                        blocks=self.blocks[i]
                    )  
                    
                    hidden1, hidden2 = hidden_down, hidden_down  # Initialize both outputs
                    

                elif i == 2:  # Upsampling
                    pred_group = 0
                    hidden1 = upsample(
                            boundaries=active_boundaries,
                            shortened_hidden=hidden_down.permute(1, 0, 2),
                        ).permute(1, 0, 2)  # B x T x C
                    hidden1 = hidden1 + residual1


                    hidden1, kv_cache[i] = self._forward(
                        hidden1, None, kv_cache[i],
                        blocks=self.blocks[i]
                    )
                elif i == 3:  # MTP stage
                    pred_group = candidates # this in n+1 candidate tokens which == 3
                    decoder_hard_boundaries = active_boundaries.clone()

                    decoder_hard_boundaries = torch.cat([decoder_hard_boundaries, torch.zeros((1,pred_group), 
                                                            device=decoder_hard_boundaries.device)], dim=1)

                    hidden_down2 =   residual_down + hidden_down

                    hidden2 = upsample(
                            boundaries=decoder_hard_boundaries,
                            shortened_hidden=hidden_down2.permute(1, 0, 2),
                        ).permute(1, 0, 2)  # B x T+pred_group x C

                    # Find index where previous segment starts
                    ones = decoder_hard_boundaries[0].nonzero(as_tuple=True)[0]          # [0, 3, 7]
                    prev_seg_start = ones[-2] if len(ones) >= 2 else ones[-1]  # 3
                    hidden2_window = hidden2[:, prev_seg_start:, :]          # B x 7 x C  (positions 3–9)
                    new_boundaries = decoder_hard_boundaries[:, prev_seg_start:]
                    group_ids = new_boundaries.cumsum(dim=1)
                    group_ids -= new_boundaries

                    attention_mask = self.group_causal_mask(group_ids).unsqueeze(1)  # (B, 1, T, T)
                    hidden2, kv_cache[i] = self._forward(
                        hidden2_window, attention_mask, kv_cache[i],
                        blocks=self.blocks[i]
                    )

            pred_group += 1

            # ============ Sampling ============
            if self.use_group_attn:
                all_logits = self.head(
                    self.norm(
                        torch.cat(
                            [hidden1[:, -1:, :], hidden2[:, -pred_group:, :]],
                            dim=1
                        )
                    )
                )
            else:
                all_logits = self.head(
                    self.norm(
                        hidden1[:, -1:, :]
                    )
                )


            main_logits = all_logits[:, :1, :]
            spec_logits = all_logits[:, 1:, :]
            # The main head's next token is, by construction, a true sample from the
            # target distribution at position T-1 (its context is unchanged in the
            # verify pass), so it is always accepted and never verified. Only the
            # cheaper MTP-drafted tokens need checking.
            first_id = self._sample_logits(
                main_logits[:, -1:, :], out_ids, temperature, top_k, top_p, repetition_penalty
            )  # (B, 1)

            if not self.use_group_attn:
                # No speculation available — one verified token per step.
                accepted_ids = first_id
                acceptance_rates.append(1.0)
            else:
                # Draft distribution q and a sample x ~ q from the MTP head.

                q_probs = self._logits_to_probs(
                    spec_logits, out_ids, temperature, top_k, top_p, repetition_penalty
                )  # (B, S, V), S = candidates+1
                spec_ids = torch.multinomial(
                    q_probs[0], 1
                ).transpose(0, 1)  # (1, S)
                T = out_ids.size(1)
                S = spec_ids.size(1)

                # ============ Verify (speculative tokens only) ============
                cand_seq = torch.cat([out_ids, first_id, spec_ids], dim=1)
                # Verifier only needs the main head — skip the MTP stage.
                verify_main_logits, _, _ = self._generate_full_forward(
                    cand_seq, candidates, boundary_predictor, need_spec=False
                )

                # first_id sits at index T; spec_ids[j] at index T+1+j. The main head
                # at position T+j predicts the token at index T+1+j (= spec_ids[j]),
                # and position T+S predicts one beyond it (the bonus token).
                verify_logits = verify_main_logits[:, T: T + S + 1, :]  # (B, S+1, V)
                p_probs = self._logits_to_probs(
                    verify_logits, out_ids, temperature, top_k, top_p, repetition_penalty
                )  # (B, S+1, V), target distributions

                # Speculative sampling: accept x_j with prob min(1, p(x_j)/q(x_j)),
                # resample the correction from the normalized residual (p-q)+ on the
                # first rejection (or a sclean bonus token when all are accepted).
                spec_accepted, n = self._speculative_accept(spec_ids, q_probs, p_probs)
                accepted_ids = torch.cat([first_id, spec_accepted], dim=1)
                # print(n)
                acceptance_rates.append(n / S)
                # print(f"{n} / {S} speculative tokens (rate {acceptance_rates[-1]:.3f})")


            if stop_token_id is not None and (accepted_ids == stop_token_id).any():
                stop_idx = (accepted_ids == stop_token_id).nonzero(as_tuple=True)[1][0].item()
                accepted_ids = accepted_ids[:, :stop_idx + 1]
                end_generation = True
            else:
                end_generation = False

            out_ids = torch.cat([out_ids, accepted_ids], dim=1)
            accepted_tokens += accepted_ids.size(1)

            if end_generation:
                break
        return out_ids.squeeze(0).tolist()[-accepted_tokens:], (
            sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0.0
        )

    
    
    @torch.inference_mode()
    def generate_verify_with_fxt(self, input_ids, max_tokens=100, temperature=1.0, top_k=None,
                        top_p=None, stop_token_id=None, repetition_penalty=1.0, candidates=1, threshold=0.9, fxt_model=None):
        """Use aan external model to verify the tokens generated by the main model. This is similar to generate_verify, but uses a fixed model for verification.


        """
        assert fxt_model is not None, "generate_verify_with_fxt requires a verifier `fxt_model`"
        if self.is_bp:
            available_bp_id = list(self.script_to_bp_layers.keys())[0]
            boundary_predictor = self.script_to_bp_layers[available_bp_id]
        
        out_ids = input_ids.clone()
        hidden_up = None
        hidden_down = None
        hard_boundaries = None
        group_ids = None
        b = 0
        confidence_threshold = threshold
        accepted_tokens = 0
        pred_group = 0
        must_accept = True  # since we are in b==1 we must accept the token
        hard_boundaries = None
        # print(confidence_threshold)

        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        main_hard_boundaries = None
        active_boundaries = None  # boundaries consistent with current hidden_down (for upsampling)
        acceptance_rates = []
        count_ones = 1
        while accepted_tokens < max_tokens:
            # Reset KV cache and attention mask each iteration since we reprocess the full sequence
            kv_cache = [[None] * len(self.blocks[i]) for i in range(len(self.blocks))]
            attention_mask = None
            hidden = self.token_embedding(out_ids)

            for i in range(len(self.blocks)):
                if i == 0:
                    embeddings = hidden
                    hidden1, kv_cache[i] = self._forward(
                        hidden, attention_mask, kv_cache[i],
                        blocks=self.blocks[i]
                    )
                    residual1 = hidden1.clone()
                    hidden2 = hidden1.clone() # B x T x C


                elif i == 1:  # Downsampling
                    assert boundary_predictor is not None
                    # hard_boundaries = boundary_predictor.infer_boundaries(hidden1.transpose(1, 0))
                    # b = int(hard_boundaries[0, -1].item())
                    _, hard_boundaries = boundary_predictor(
                    hidden1.transpose(1, 0), prior=[(0.333, 0.023)]
                    )
                    b = int(hard_boundaries[0, -1].item())
                    count_ones += b
                    if main_hard_boundaries is None:
                        main_hard_boundaries = hard_boundaries
                    else:
                        # get difference between main_hard_boundaries and hard_boundaries sum to see if any new boundaries were added
                        diff = hard_boundaries.size(1) - main_hard_boundaries.size(1)
                        if diff > 0:
                            main_hard_boundaries =  torch.cat([main_hard_boundaries, hard_boundaries[:, -diff:]], dim=1) # this is what I used for my NMT task.
                    
                    # cut the current segment

                    active_boundaries = main_hard_boundaries.clone()

                    hidden_down = downsample(
                        boundaries=active_boundaries,
                        hidden=hidden1.permute(1, 0, 2),  # T x B x C
                        null_group=self.null_group,
                    ).permute(1, 0, 2)  # B x T x C

                    if self.use_group_attn:
                        residual_down = hidden_down.clone()

                    hidden_down, _ = self._forward(
                        hidden_down, attention_mask, kv_cache[i],
                        blocks=self.blocks[i]
                    )  
                    
                    hidden1, hidden2 = hidden_down, hidden_down  # Initialize both outputs
                    
                          
                elif i == 2:  # Upsampling
                    pred_group = 0
                    hidden1 = upsample(
                            boundaries=active_boundaries,
                            shortened_hidden=hidden_down.permute(1, 0, 2),
                        ).permute(1, 0, 2)  # B x T x C
                    hidden1 = hidden1 + residual1


                    hidden1, kv_cache[i] = self._forward(
                        hidden1, None, kv_cache[i],
                        blocks=self.blocks[i]
                    )
                elif i == 3:  # MTP stage
                    pred_group = candidates # this in n+1 candidate tokens which == 3
                    decoder_hard_boundaries = active_boundaries.clone()

                    decoder_hard_boundaries = torch.cat([decoder_hard_boundaries, torch.zeros((1,pred_group), 
                                                            device=decoder_hard_boundaries.device)], dim=1)

                    hidden_down2 =   residual_down + hidden_down

                    hidden2 = upsample(
                            boundaries=decoder_hard_boundaries,
                            shortened_hidden=hidden_down2.permute(1, 0, 2),
                        ).permute(1, 0, 2)  # B x T+pred_group x C

                    # Find index where previous segment starts
                    ones = decoder_hard_boundaries[0].nonzero(as_tuple=True)[0]          # [0, 3, 7]
                    prev_seg_start = ones[-2] if len(ones) >= 2 else ones[-1]  # 3
                    hidden2_window = hidden2[:, prev_seg_start:, :]          # B x 7 x C  (positions 3–9)
                    new_boundaries = decoder_hard_boundaries[:, prev_seg_start:]
                    group_ids = new_boundaries.cumsum(dim=1)
                    group_ids -= new_boundaries

                    attention_mask = self.group_causal_mask(group_ids).unsqueeze(1)  # (B, 1, T, T)
                    hidden2, kv_cache[i] = self._forward(
                        hidden2_window, attention_mask, kv_cache[i],
                        blocks=self.blocks[i]
                    )

            pred_group += 1

            # ============ Sampling ============
            if self.use_group_attn:
                all_logits = self.head(
                    self.norm(
                        torch.cat(
                            [hidden1[:, -1:, :], hidden2[:, -pred_group:, :]],
                            dim=1
                        )
                    )
                )
            else:
                all_logits = self.head(
                    self.norm(
                        hidden1[:, -1:, :]
                    )
                )

            # breakpoint()

            main_logits = all_logits[:, :1, :]
            # Every drafted token is speculative. The drafter's main head is not the
            # verifier (we verify against fxt_model), so its next-token sample is not
            # a sample from the target distribution and must be checked like the rest.
            spec_logits = all_logits  # (B, S, V), S = 1 + pred_group

            if not self.use_group_attn:
                # No speculation available — one verified token per step.
                accepted_ids = self._sample_logits(
                    main_logits[:, -1:, :], out_ids, temperature, top_k, top_p, repetition_penalty
                )  # (B, 1)
                acceptance_rates.append(1.0)
            else:
                # Draft distribution q and a sample x ~ q from the main + MTP heads.
                q_probs = self._logits_to_probs(
                    spec_logits, out_ids, temperature, top_k, top_p, repetition_penalty
                )  # (B, S, V)
                spec_ids = torch.multinomial(
                    q_probs[0], 1
                ).transpose(0, 1)  # (1, S)
                T = out_ids.size(1)
                S = spec_ids.size(1)

                # ============ Verify (all drafted tokens) ============
                cand_seq = torch.cat([out_ids, spec_ids], dim=1)
                # Verifier only needs the main head — skip the MTP stage.
                verify_bp = fxt_model.script_to_bp_layers[available_bp_id]
                verify_main_logits, _, _ = fxt_model._generate_full_forward(
                    cand_seq, candidates, verify_bp, need_spec=False
                )

                
                verify_logits = verify_main_logits[:, T - 1: T + S, :]  # (B, S+1, V)
                p_probs = self._logits_to_probs(
                    verify_logits, out_ids, temperature, top_k, top_p, repetition_penalty
                )  # (B, S+1, V), target distributions

                
                spec_accepted, n = self._speculative_accept(spec_ids, q_probs, p_probs)
                accepted_ids = spec_accepted
                acceptance_rates.append(n / S)
                # print(f"{n} / {S} speculative tokens (rate {acceptance_rates[-1]:.3f})")

           

            if stop_token_id is not None and (accepted_ids == stop_token_id).any():
                stop_idx = (accepted_ids == stop_token_id).nonzero(as_tuple=True)[1][0].item()
                accepted_ids = accepted_ids[:, :stop_idx + 1]
                end_generation = True
            else:
                end_generation = False

            out_ids = torch.cat([out_ids, accepted_ids], dim=1)
            accepted_tokens += accepted_ids.size(1)

            if end_generation:
                break
        return out_ids.squeeze(0).tolist()[-accepted_tokens:], (
            sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0.0
        )
    

   