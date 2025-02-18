import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size//heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, key_len, self.heads, self.head_dim)

        # shape - squeeze in head and head dimension
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy*mask

        attention = torch.softmax(energy/(self.embed_size ** (0.5), dim=3)) 
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)

        out = self.fc_out(out)
        return out  

class AttentionBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(AttentionBlock, self).__init__()
        self.attention = CrossAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.ffn = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
            )
        
        self.dropout = nn.Dropout(Dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        xf = self.ffn(x)
        out = self.dropout(self.norm2(xf + x))
        return out

class ProjectionBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(ProjectionBlock, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
            )
        self.dropout = nn.Dropout(Dropout)
        self.norm1 = nn.LayerNorm(embed_size)
    
    def forward(self, x):
        xf = self.ffn(x)
        out = self.dropout(self.norm1(xf+x))
        return out

class Transformer(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout):
        super(Transformer, self).__init__()
        self.device = "cuda"
        self.heads = heads
        self.layers = nn.ModuleList(
            [AttentionBlock(embed_size, heads, forward_expansion, dropout)
            for _ in range(num_layers)]
        )        

    def create_mask(self, txt_len, img_len, idx):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to("cuda")

    def forward(self, x_txt, x_img):
        src_len1 = x_txt.shape[0]
        src_len2 = x_img.shape[0]
        mask = self.create_mask(src_len1, src_len2, idx)
        return 


        

        




        

        
