import numpy as np
import torch
import torch.nn as nn
import transformers
from minGPT.transformer import minGPT
from feedforward.mlp import AdaInMLP
from gpt2 import GPT2Model

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size, context_length, max_ep_len, action_tanh, device, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        transformer_config = transformers.GPT2Config(
            n_embd=hidden_size,
            block_size=context_length * 3,
            **kwargs,
        )
        # transformer_config = transformers.GPT2Config(
        #     vocab_size=1,
        #     n_embd=hidden_size,
        #     n_ctx=context_length * 3,
        #     device=device,
        #     **kwargs
        # )

        self.state_dim = state_dim
        self.act_dim = 1
        self.c_len = context_length
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.transformer = minGPT(transformer_config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_action = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_action = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, states, actions, ret_to_go, timesteps, attn_mask=None):
        batch_size, seq_len = states.shape[0], states.shape[1]
        if attn_mask is None:
            attn_mask = torch.ones((batch_size, seq_len), dype=torch.long)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        return_embeddings = self.embed_return(ret_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # dims batch x seq_len x hidden state
        state_embeddings += time_embeddings
        action_embeddings += time_embeddings
        return_embeddings += time_embeddings

        stacked_inputs = torch.stack((
            return_embeddings, state_embeddings, action_embeddings
        ), dim=1).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_len, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attn_mask = torch.stack(
            (attn_mask, attn_mask, attn_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 1, 3*seq_len).to(device=self.device)
        # stacked_attention_mask = torch.stack(
        #     (attn_mask, attn_mask, attn_mask), dim=1
        # ).permute(0, 2, 1).reshape(batch_size, 3*seq_len).to(device=self.device)

        logits = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attn_mask,
        )

        # transformer_outputs = self.transformer(
        #     inputs_embeds=stacked_inputs,
        #     attention_mask=stacked_attention_mask,
        # )
        # logits = transformer_outputs['last_hidden_state']

        logits = logits.reshape(batch_size, seq_len, 3, self.hidden_size).permute(0, 2, 1, 3)
        action_hidden = logits[:, 1, :, :]

        action_preds = self.predict_action(action_hidden)
        # action_preds = self.predict_action(logits[:,1])
        return action_preds

    def get_action(self, states, actions, returns_to_go, timesteps, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        states = states[:,-self.c_len:]
        actions = actions[:,-self.c_len:]
        returns_to_go = returns_to_go[:,-self.c_len:]
        timesteps = timesteps[:,-self.c_len:]

        # pad all tokens to sequence length
        seq_len = states.shape[1]
        attention_mask = torch.cat([torch.ones(seq_len), torch.zeros(self.c_len-seq_len)])
        attention_mask = attention_mask.to(device=states.device).reshape(1, -1)
        states = torch.cat(
            [states, torch.zeros((states.shape[0], self.c_len-seq_len, self.state_dim), device=states.device)],
            dim=1).to(dtype=torch.float32)
        actions = torch.cat(
            [actions, torch.zeros((actions.shape[0], self.c_len - actions.shape[1], self.act_dim),
                            device=actions.device)],
            dim=1).to(dtype=torch.float32)
        returns_to_go = torch.cat(
            [returns_to_go, torch.zeros((returns_to_go.shape[0], self.c_len-returns_to_go.shape[1], 1), device=returns_to_go.device)],
            dim=1).to(dtype=torch.float32)
        timesteps = torch.cat(
            [timesteps, torch.zeros((timesteps.shape[0], self.c_len-timesteps.shape[1]), device=timesteps.device)],
            dim=1
        ).to(dtype=torch.long)
        # print(attention_mask)
        # print(states)
        # print(actions)
        # print(returns_to_go)
        # print(timesteps)

        action_preds = self.forward(
            states, actions, returns_to_go, timesteps, attn_mask=attention_mask, **kwargs)

        return action_preds[0, seq_len - 1]

    def configure_optimizers(self, train_config):
        return self.transformer.configure_optimizers(train_config)