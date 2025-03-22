import torch
import torch.nn as nn
import torch.nn.functional as F

txt_tok_len = 50
img_tok_len = 60

# define the masks
def get_mask(txt_tok, img_tok, typ, alignm=None):
    if typ=="causal":
        mask = torch.tril(torch.ones(txt_tok+img_tok, txt_tok+img_tok)).unsqueeze(0).unsqueeze(0)
    # elif typ=="transfusion":
    #     full_mask = torch.zeros(txt_tok+img_tok, txt_tok+img_tok)
    #     mask_txt_txt = torch.tril(torch.ones(txt_tok, txt_tok))
    #     mask_img_img = torch.diag(torch.ones(img_tok, img_tok))
    #     mask_img_txt = torch.zeros(img_tok, txt_tok)
    #     mask_txt_img =  
    # elif typ=="trinity":
    #     full_mask = torch.zeros(txt_tok+img_tok, txt_tok+img_tok)
    #     part_mask_txt = torch.tril(torch.ones(txt_tok, txt_tok))
    else:
        mask = None
        print("Incorrect mask")
    return mask

# do cross attentions
class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim_q // num_heads) ** -0.5

        # cross attention Q, K and V
        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.v_proj = nn.Linear(dim_kv, dim_q)

        # projection layer
        self.out_proj = nn.Linear(dim_q, dim_q)
        
        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, typ):
        B, N, C = q.shape
        _, N1, _ = kv.shape

        q = self.q_proj(q).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k_proj(kv).view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v_proj(kv).view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale

        # add causal mask
        causal_mask = get_mask(txt_tok_len, img_tok_len, typ)
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))

        # continue processing
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)

        output = self.out_proj(attn_output)
        
        return output
    
class LinearLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(LinearLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        linear_output = self.linear(x)
        x = self.norm(x + linear_output)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads, hidden_dim):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attention = CrossAttention(dim_q, dim_kv, num_heads)
        self.linear_block = LinearLayer(dim_q, hidden_dim)

    def forward(self, q, kv, typ):
        x = self.cross_attention(q, kv, typ)
        x = self.linear_block(x)
        return x
    
class EncoderModel(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8, num_blocks=4):
        super(EncoderModel, self).__init__()
        hidden_dim=dim_q//4
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(dim_q, dim_kv, num_heads, hidden_dim) for _ in range(num_blocks)
        ])

    def forward(self, q, kv, typ):
        for block in self.blocks:
            q = block(q, kv, typ)
        return q

# Example usage
if __name__ == "__main__":
    q = torch.randn(2, 110, 128)
    kv = torch.randn(2, 110, 128)
    
    cross_attn = EncoderModel(dim_q=128, dim_kv=128, num_heads=8, num_blocks=8)
    output = cross_attn(q, kv, typ="causal")
    print(output.shape) 