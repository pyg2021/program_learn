import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), 
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):              
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)           # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)          # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        image_height=100
        image_width=10000
        out_channels=32
        patch_height, patch_width = pair(patch_size)

        assert  image_height % patch_height ==0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}
        self.cnn_block1=nn.Sequential(
            nn.Conv2d(channels,out_channels,3,1,1),
            nn.ReLU(True),
            nn.Conv2d(out_channels,out_channels,3,1,1),
            nn.ReLU(True),
            nn.Conv2d(out_channels,channels,3,1,1))
        
        self.cnn_block2=nn.Sequential(
            nn.Conv2d(channels,1,3,1,1),
            nn.ReLU(True),
            )
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))					# nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.inverse_embedding = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b d (p c) -> b c p d', p=patch_height*patch_width),
            
        )
        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm((num_patches+1)),
            nn.Linear((num_patches+1), num_patches)
        )

    def forward(self, img):
        x=self.cnn_block1(img)
        x = self.to_patch_embedding(img)        # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        b, n, _ = x.shape           # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)  
        x = torch.cat((cls_tokens, x), dim=1)               # 将cls_token拼接到patch token中去       (b, 65, dim)
        x += self.pos_embedding[:, :(n+1)]                  # 加位置嵌入（直接加）      (b, 65, dim)
        x = self.dropout(x)

        x = self.transformer(x)                                                 # (b, 65, dim)


        x=self.inverse_embedding(x)

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]                   # (b, dim)

        x = self.to_latent(x)                                                   # Identity (b, dim)
        x=self.mlp_head2(x)
        x=self.cnn_block2(x)
        return x.reshape(-1,1,100,10000)
        # return self.mlp_head(x)                                                 #  (b, num_classes)

# x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]                   # (b, dim)



# model_vit = ViT(
#         image_size = 1000,
#         patch_size = 50,
#         num_classes = 1000,
#         dim = 1024,
#         depth = 6,
#         heads = 16,
#         mlp_dim = 2048,
#         dropout = 0.1,
#         emb_dropout = 0.1
#     )
# n=1000
# device="cuda"
# model_vit.to(device)
# img = torch.ones(32, 3, n, n).to(device)
# img_target=torch.ones(32,3,n,n)*3
# img_target=img_target.to(device)
# optimizer=torch.optim.Adam(model_vit.parameters(),1e-3)
# loss=torch.nn.L1Loss()

# for i in range(1000):
#     model_vit.train()
#     optimizer.zero_grad()
#     y=model_vit(img)
#     loss_0=loss(y.reshape(-1,3,n,n),img_target)
#     loss_0.backward()
#     optimizer.step()
#     print(loss_0)

# img_test=torch.ones(1, 3, 256, 256).to(device)
# img_target=torch.ones(1,3,256,256)*3
# img_target=img_target.to(device)
# y=model_vit(img_test)
# print("loss:",loss(y.reshape(-1,3,256,256),img_target))
# print(y.reshape(-1,3,256,256))