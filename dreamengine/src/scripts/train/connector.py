import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipModel, SiglipProcessor
from PIL import Image
import numpy as np

import pdb


class PerceiverResampler(nn.Module):
    def __init__(self, dim, num_latents=64, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        
        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        
        # Cross-attention projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
        self.ln_k = nn.LayerNorm(dim)
        self.ln_q = nn.LayerNorm(dim)

    def forward(self, x):
        # x shape: (batch, num_patches, dim)
        b, m, d = x.shape
        
        # Initialize latents for the batch
        latents = self.latents.unsqueeze(0).repeat(b, 1, 1) # (b, num_latents, dim)
        
        q_input = self.ln_q(latents)
        kv_input = self.ln_k(x)
        
        # Concatenate latents to KV pool just like Flamingo Perceiver Resampler
        kv_input = torch.cat((kv_input, q_input), dim=1)
        
        # Project
        q = self.to_q(q_input)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        # Reshape for multi-head attention
        # (b, heads, seq_len, dim_head)
        q = q.view(b, -1, self.heads, q.shape[-1] // self.heads).transpose(1, 2)
        k = k.view(b, -1, self.heads, k.shape[-1] // self.heads).transpose(1, 2)
        v = v.view(b, -1, self.heads, v.shape[-1] // self.heads).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)

        b, h, n, dh = out.shape
        out = out.transpose(1, 2).contiguous().view(b, -1, h * dh)
        
        return self.to_out(out)


class DeepPerceiverResampler(nn.Module):
    def __init__(self, dim, num_latents=128, depth=6, heads=8):
        """
        depth (int): The number of Perceiver layers. DeepMind's Flamingo used depth=6.
        """
        super().__init__()
        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        
        # Stack of Perceiver Layers
        self.layers = nn.ModuleList([
            PerceiverResampler(dim=dim, num_latents=num_latents, heads=heads, dim_head=64) for _ in range(depth)
        ])
        
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, visual_features):       
        # Pass through the deep stack
        for layer in self.layers:
            latents = layer(visual_features)
            
        return self.final_norm(latents)


class MLPProjection(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.layer_norm(self.net(x))


# ==========================================
# 2. MAIN MULTIMODAL MODEL WORKSPACE
# ==========================================

class MultimodalFusionModel(nn.Module):
    def __init__(self, siglip_model_name="google/siglip-base-patch16-224", num_latents=128):
        super().__init__()
        # Load Siglip
        self.siglip = SiglipModel.from_pretrained(siglip_model_name, torch_dtype=torch.float16)
        dim = self.siglip.config.vision_config.hidden_size
        
        # Freeze SIGLIP completely
        for param in self.siglip.parameters():
            param.requires_grad = False
            
        # Initialize Trainable Components
        # self.perceiver = PerceiverResampler(dim=dim, num_latents=num_latents)
        self.big_perceiver = DeepPerceiverResampler(dim=dim, num_latents=num_latents)
        self.ln_self = nn.LayerNorm(dim)
        
        # 2 Layer MLP Projection
        self.mlp = MLPProjection(dim=dim, hidden_dim=dim * 4)

    def forward(self, input_ids, attention_mask, pixel_values):
        # Extract Text Embeddings from SIGLIP Text Encoder
        text_outputs = self.siglip.text_model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        text_embeds = text_outputs.last_hidden_state # (batch, text_seq_len, dim)
        
        # Extract Patch Embeddings from SIGLIP Vision Encoder
        vision_outputs = self.siglip.vision_model(pixel_values=pixel_values)
        patch_embeds = vision_outputs.last_hidden_state # (batch, vision_patches, dim)

        # 2. Pass through 1 layer Gated Cross-Attention
        x = self.big_perceiver(torch.cat((text_embeds, patch_embeds, patch_embeds), dim=1))
        
        # 3. Pass through 1 layer Self Attention
        x_norm = self.ln_self(x)
        
        # 4. Pass through 2-layer MLP Projection
        output_tokens = self.mlp(x_norm)
        
        return output_tokens


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 1. Initialize Processor and Model
#     processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
#     model = MultimodalFusionModel().to(device, dtype=torch.float16)
    
#     # Verify parameter frozen/trainable states
#     trainable_params = [p for p in model.parameters() if p.requires_grad]
#     param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     frozen_params = [p for p in model.siglip.parameters()]
#     print(f"Total trainable parameter tensors: {param_count}")
#     print(f"Are Siglip parameters frozen? {all(p.requires_grad == False for p in frozen_params)}")

#     # 2. Setup Optimizer for only the trainable parameters
#     optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
    
#     # 3. Simulated Data Pipeline
#     # Create dummy image and text inputs
#     dummy_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
#     dummy_text = ["A photo of an asset being processed via machine learning."]
    
#     inputs = processor(text=dummy_text, images=dummy_image, return_tensors="pt", padding="max_length", truncation=True)
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     for k, v in inputs.items():
#         if torch.is_floating_point(v):
#             inputs[k] = v.half()

#     if 'attention_mask' not in inputs:
#         pad_token_id = processor.tokenizer.pad_token_id or 1
#         inputs['attention_mask'] = (inputs['input_ids'] != pad_token_id).long()
    
#     # Target values (Assuming we are trying to align to some downstream target token space)
#     # Target shape matches output shape: (Batch, Text_Seq_Len, Dimension)
#     batch_size, text_seq_len = inputs['input_ids'].shape
#     dim = model.siglip.config.vision_config.hidden_size
#     dummy_targets = torch.randn(batch_size, text_seq_len, dim).to(device)

#     # 4. Dummy Training Step Execution
#     model.train()
#     optimizer.zero_grad()
    
#     # Forward Pass
#     output_tokens = model(
#         input_ids=inputs['input_ids'],
#         attention_mask=inputs['attention_mask'],
#         pixel_values=inputs['pixel_values']
#     )
    
#     # Smooth L1 Loss for exact coordinate/value matching
#     criterion = nn.SmoothL1Loss()
#     loss = criterion(output_tokens, dummy_targets)
    
#     # Backward Pass & Update
#     loss.backward()
#     optimizer.step()
    
#     print(f"Output Matrix Shape: {output_tokens.shape}")
#     print(f"Step completed successfully. Loss: {loss.item():.4f}")