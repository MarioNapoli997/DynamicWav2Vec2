import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from einops.layers.torch import Rearrange
from einops import reduce, rearrange, repeat

import transformers
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput, MaskedLMOutput

from typing import Optional, Tuple

from transformers.activations import ACT2FN



class DropPredictor(nn.Module):
    """ Computes the log-probabilities of dropping a token, adapted from PredictorLG here:
    https://github.com/raoyongming/DynamicViT/blob/48ac52643a637ed5a4cf7c7d429dcf17243794cd/models/dyvit.py#L287 """
    def __init__(self, embed_dim):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, policy):
        x = self.in_conv(x)

        ## B = batch size.
        ## N = length of the sequence (the number of tokens).
        ## C = size of the embedding.

        B, N, C = x.size()

        ## Split the input into a local part and a global part based on the policy
        local_x = x[:,:, :C//2]
        global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / (torch.sum(policy, dim=1, keepdim=True)+0.000001)

        ## Combine local and global part

        """The combination of these two parts (local and global) is designed to take into account 
        both token-specific information and the relative importance of the tokens according to policy."""

        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)

        ## self.out_conv produces a two-dimensional tensor, interpreted as a log probability.
        
        return self.out_conv(x)




class Wav2Vec2PositionalConvEmbedding(nn.Module):

    """original class of model wav2vec2, unmodified"""
    
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
        self.padding = Wav2Vec2SamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        ## Transpose input dimensions for 1D convolution
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)

        ## padding to keep the size of the output equal to that of the input
        hidden_states = self.padding(hidden_states)


        hidden_states = self.activation(hidden_states)
        

        ## Restore original size
        hidden_states = hidden_states.transpose(1, 2)


        return hidden_states


class Wav2Vec2SamePadLayer(nn.Module):

    """original class of model wav2vec2, unmodified"""
    
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


    
class Wav2Vec2EncoderLayer(nn.Module):

    """original class of model wav2vec2, unmodified"""
    
    def __init__(self, config):
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Wav2Vec2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        ## get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        ## get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        ## if the attention mask is provided it is used to mask the tokens when calculating the attention
        ## In this case the attention mask provided to the model is the policy calculated by the encoder

        if attention_mask is not None:

          ## Extract attention dimensions

          """H indicates how many heads of attention are involved, while N indicates the maximum length of sequences involved in the attention 
          between queries and keys. These dimensions are important in defining the shape of the tensors used in multi-head attention computation.
          """

          H,N,N = attn_weights.size()

          ## Resize the attention mask to a shape compatible with computation
          mask_policy = attention_mask.reshape(bsz,1,N)
          ## Create a diagonal matrix with 1 on the main diagonal and 0 elsewhere
          eye = torch.eye(N, dtype = mask_policy.dtype, device = mask_policy.device).view(1,1,N,N)
          ## Combine attention mask and diagonal matrix
          """ In this way we add a weight to the masked tokens, which will therefore not be used during the attention calculation """
          mask_policy = mask_policy + (1.0 - mask_policy) * eye

          ## Resize the mask to have the correct size for the attention calculation
          attention_mask = mask_policy.view(bsz,1,N,N)

          ## Check the attention size
          assert attention_mask.size() == (
              bsz,
              1,
              tgt_len,
              src_len,
              ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
          attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
          attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)


        attn_weights = F.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"

            ## Add mask to attention weights
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class Wav2Vec2FeedForward(nn.Module):

     """original class of model wav2vec2, unmodified"""
    
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states




class MyWav2Vec2Encoder(transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Encoder):
  def __init__(self, config, d_model = 1024, n_blocks = 24):
        super().__init__(config)
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.score_predictor = nn.ModuleList([DropPredictor(d_model) for _ in range(n_blocks)])


  def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True, drop_temp = 1,
    ):
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None

    #print(f'[My encoder] input hidden states: {hidden_states.shape}')

    if attention_mask is not None:
      # make sure padded tokens output 0
      hidden_states[~attention_mask] = 0.0

      # extend attention_mask
      attention_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)) * -10000.0
      attention_mask = attention_mask.expand(
          attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
          )

    position_embeddings = self.pos_conv_embed(hidden_states)
    hidden_states = hidden_states + position_embeddings
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)


    # Initialize drop decisions
    B, P, _ = hidden_states.shape
    prev_decision = torch.ones(B, P, 1, dtype=hidden_states.dtype, device=hidden_states.device)

    out_pred_prob = []
    pred_distr = [[],[],[],[]]


    for i,layer in enumerate(self.layers):

      """ in the 5th, 12th, 17th layer we calculate the policy which will then be passed to the "Wav2Vec2Attention" class as attention mask. """

      if i in [5,12,17]:
        policy = torch.ones(B, P, 1, dtype=hidden_states.dtype, device=hidden_states.device)
        # Current drop score
        pred_score = self.score_predictor[i](hidden_states, prev_decision)#.reshape(B, -1, 2)
        keepall = torch.cat((torch.zeros_like(pred_score[:,:,0:1]), torch.ones_like(pred_score[:,:,1:2])),2)
        pred_score = pred_score*drop_temp + keepall*(1-drop_temp)

        if True: #self.training:

          # Convert to log-prob
          pred_score = torch.log(pred_score + 1e-8)

          # Sample mask and update previous one
          hard_keep_decision = F.gumbel_softmax(pred_score, hard = True)[:, :, 1:2]*prev_decision

        else:

          # Treshold mask and update previous one
          hard_keep_decision = (pred_score[:, :, 1:2] > 0.9).float() * prev_decision

        policy = hard_keep_decision
        prev_decision = hard_keep_decision

    """In the other layers the policy will be set as None so as not to mask any token during the attention calculation"""

      else:
        policy = None

      if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

      # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
      dropout_probability = np.random.uniform(0, 1)

      if self.training and (dropout_probability < self.config.layerdrop):  # skip the layer
        layer_outputs = (None, None)
      else:
        if getattr(self.config, "gradient_checkpointing", False) and self.training:
          # create gradient checkpointing function
          def create_custom_forward(module):
            def custom_forward(*inputs):
              return module(*inputs, output_attentions)

            return custom_forward

          layer_outputs = torch.utils.checkpoint.checkpoint(
              create_custom_forward(layer),
              hidden_states,
              attention_mask = policy,
                      )
        else:
          layer_outputs = layer(
              hidden_states, attention_mask = policy, output_attentions=output_attentions
                    )
          hidden_states = layer_outputs[0]

      if output_attentions:
        all_self_attentions = all_self_attentions + (layer_outputs[1],)

      #print(f'[My encoder] layers hidden states: {hidden_states.shape}')

    if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)


    if not return_dict:
      return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
    return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
