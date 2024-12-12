from typing import Optional, Tuple
import torch
from torch import nn
from .config import BaseConfig, VisionConfig, TextConfig


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class SiglipMHAPoolingHead(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.norm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SiglipMLP(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU(approximate='tanh')

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipAttention(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = self.embed_dim // self.num_heads
        self.scale = self.head_size ** -0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(p=config.attention_dropout)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        # (batch, seq_len, embed_dim)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # (batch, seq_len, num_heads, head_size)
        query_states = query_states.reshape(batch, seq_len, self.num_heads, self.head_size)
        key_states = key_states.reshape(batch, seq_len, self.num_heads, self.head_size)
        value_states = value_states.reshape(batch, seq_len, self.num_heads, self.head_size)

        # (batch, num_heads, seq_len, head_size)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # (batch, num_heads, seq_len, seq_len)
        attention_scores = (self.scale * query_states) @ key_states.transpose(-1, -2)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # (batch, num_heads, seq_len, seq_len)
        attention_weights = self.dropout(attention_scores.softmax(dim=-1, dtype=torch.float32))
        # (batch, num_heads, seq_len, head_size)
        attentions = attention_weights @ value_states
        # (batch, seq_len, num_heads, head_size)
        attentions = attentions.permute(0, 2, 1, 3).contiguous()
        attentions = attentions.reshape(batch, seq_len, self.embed_dim)

        return self.out_proj(attentions)


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = SiglipAttention(config=config)

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = SiglipMLP(config=config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states=hidden_states, attention_mask=attention_mask)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()

        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()

        self.use_head = config.use_head

        self.embedding = SiglipVisionEmbedding(config=config)
        self.encoder = SiglipEncoder(config=config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        if self.use_head:
            self.head = SiglipMHAPoolingHead(config)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # (batch, grid*grid, hidden_size)
        hidden_states = self.embedding(pixel_values)
        hidden_states = self.encoder(hidden_states)
        # (batch, grid*grid, hidden_size)
        hidden_states = self.norm(hidden_states)

        # (batch, hidden_size)
        pooler_output = self.head(hidden_states) if self.use_head else None

        return hidden_states, pooler_output


class SiglipTextTransformer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.embedding = SiglipTextEmbedding(config)
        self.encoder = SiglipEncoder(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.head = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        hidden_states = self.embedding(input_ids=input_ids, position_ids=position_ids)

        # note: SigLIP's text model does not use a causal mask, unlike the original CLIP model.
        # expand attention_mask
        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, dtype=hidden_states.dtype)

        hidden_states = self.encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )

        # (batch, seq_len, hidden_size)
        hidden_states = self.norm(hidden_states)

        # Assuming "sticky" EOS tokenization, last token is always EOS.
        # (batch, hidden_size)
        pooled_output = hidden_states[:, -1, :]
        # (batch, hidden_size)
        pooled_output = self.head(pooled_output)

        # hidden_states, pooled (EOS token) states
        return hidden_states, pooled_output


class SiglipVisionEmbedding(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding='valid'
        )

        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, config.hidden_size)
        self.register_buffer('position_ids', torch.arange(self.num_patches).unsqueeze(0), persistent=False)

    def forward(
            self,
            pixel_values: torch.FloatTensor
    ) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        pixel_values = pixel_values.to(dtype=self.patch_embedding.weight.dtype)
        # (batch, hidden_size, grid, grid)
        patch_embeds = self.patch_embedding(pixel_values)
        # (batch, hidden_size, grid*grid)
        patch_embeds = torch.flatten(patch_embeds, 2)
        # (batch, grid*grid, hidden_size)
        patch_embeds = torch.transpose(patch_embeds, 1, 2)

        # if interpolate_pos_encoding:
        #     embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        # else:
        #     embeddings = patch_embeds + self.position_embedding(self.position_ids)

        embeddings = patch_embeds + self.position_embedding(self.position_ids)
        return embeddings


class SiglipTextEmbedding(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()

        self.tokens_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_embedding_positions, config.hidden_size)
        self.register_buffer('position_ids', torch.arange(config.max_embedding_positions).unsqueeze(0), persistent=False)

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_len = input_ids.shape[-1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]

        input_embeds = self.tokens_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)

        return input_embeds + position_embeds


class SiglipVisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()

        self.vision_model = SiglipVisionTransformer(config=config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)


class SiglipTextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.text_model = SiglipTextTransformer(config=config)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )


class SiglipModel(nn.Module):
    def __init__(self, vision_config: VisionConfig, text_config: TextConfig):
        super().__init__()

        self.vision_model = SiglipVisionModel(config=vision_config)
        self.text_model = SiglipTextModel(config=text_config)

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

    def get_text_features(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        pooled_output = text_outputs[1]
        return pooled_output

    def get_image_features(
            self,
            pixel_values: torch.Tensor
    ) -> torch.Tensor:
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs[1]
        return pooled_output

    def forward(
            self,
            input_ids: torch.Tensor,
            pixel_values: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            return_loss: Optional[bool] = None
    ) -> Tuple:
        """
        :param input_ids: (batch, seq_len)
        :param pixel_values: (batch, channel, height, width)
        :param attention_mask:
        :param position_ids:
        :param return_loss:
        :return:
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]

        # normalized features
        image_embeds: torch.Tensor = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds: torch.Tensor = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = (
                text_embeds @ image_embeds.t().to(text_embeds.device) * self.logit_scale.exp()
                + self.logit_bias
        )

        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            # (2, 2)
            # [[1, 0], [0, 1]]
            eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
            # [[-1, -1,], [-1, -1]] + [[2, 0], [2, 0]]
            # = [[1, -1], [-1, 1]]
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()

        output = (logits_per_text, logits_per_image, text_embeds, image_embeds)
        return output + (loss, ) if loss is not None else output
