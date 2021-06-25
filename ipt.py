import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter, Softmax, Dropout, LayerNorm, ReLU, Embedding



class MultiheadAttention(nn.Module):
    """
    Apply multi-headed attention from "from_tensor" to "to_tensor".

    Args:
        batch_size (int): Batch size of input datasets.
        from_tensor_width (int): Size of last dim of from_tensor.
        to_tensor_width (int): Size of last dim of to_tensor.
        from_seq_length (int): Length of from_tensor sequence.
        to_seq_length (int): Length of to_tensor sequence.
        num_attention_heads (int): Number of attention heads. Default: 1.
        size_per_head (int): Size of each attention head. Default: 512.
        query_act (str): Activation function for the query transform. Default: None.
        key_act (str): Activation function for the key transform. Default: None.
        value_act (str): Activation function for the value transform. Default: None.
        has_attention_mask (bool): Specifies whether to use attention mask. Default: False.
        attention_probs_dropout_prob (float): The dropout probability for
                                      MultiheadAttention. Default: 0.0.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        do_return_2d_tensor (bool): True for return 2d tensor. False for return 3d
                             tensor. Default: False.
    """

    def __init__(self,
                 q_tensor_width,
                 k_tensor_width,
                 v_tensor_width,
                 hidden_width,
                 out_tensor_width,
                 num_attention_heads=1,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 out_act=None,
                 has_attention_mask=True,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 do_return_2d_tensor=False,
                 same_dim=True):
        super(MultiheadAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.size_per_head = int(hidden_width / num_attention_heads)
        self.has_attention_mask = has_attention_mask             # unused
        self.use_one_hot_embeddings = use_one_hot_embeddings     # unused
        self.initializer_range = initializer_range               # unused
        self.do_return_2d_tensor = do_return_2d_tensor
        self.same_dim = same_dim
        self.device = torch.device('cuda')

        self.scores_mul = 1.0 / math.sqrt(float(self.size_per_head))  
        self.shape_q_2d = (-1, q_tensor_width)
        self.shape_k_2d = (-1, k_tensor_width)
        self.shape_v_2d = (-1, v_tensor_width)
        self.hidden_width = int(hidden_width)
        if self.same_dim:                      
            self.in_proj_layer = Parameter(torch.FloatTensor(np.random.rand(hidden_width * 3, q_tensor_width))).to(self.device)
        
        else:
            self.query_layer = nn.Linear(q_tensor_width, hidden_width, bias=False)

            self.key_layer = nn.Linear(k_tensor_width, hidden_width, bias=False)

            self.value_layer = nn.Linear(v_tensor_width, hidden_width, bias=False)
         
        self.out_proj = nn.Linear(hidden_width, out_tensor_width, bias=False).to(self.device)

        self.trans_shape = (0, 2, 1, 3)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p = 1. - attention_probs_dropout_prob)
        self.use_dropout = attention_probs_dropout_prob > 0

        '''if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.add = P.TensorAdd()
            self.cast = P.Cast()
            self.get_dtype = P.DType()'''

    def forward(self, tensor_q, tensor_k, tensor_v):
        """
        Apply multihead attention.
        """
        batch_size, seq_length, _ = tensor_q.size()
        shape_qkv = (batch_size, -1, self.num_attention_heads, self.size_per_head)
        shape_linear = (batch_size * seq_length, self.num_attention_heads * self.size_per_head)

        if self.do_return_2d_tensor is True:
            shape_return = (batch_size * seq_length, self.num_attention_heads * self.size_per_head)
            if seq_length == -1:
                shape_return = (-1, self.num_attention_heads * self.size_per_head)
        else:
            shape_return = (batch_size, seq_length, self.num_attention_heads * self.size_per_head)

        tensor_q_2d = torch.reshape(tensor_q, self.shape_q_2d)
        tensor_k_2d = torch.reshape(tensor_k, self.shape_k_2d)
        tensor_v_2d = torch.reshape(tensor_v, self.shape_v_2d)

        if torch.eq(tensor_q_2d, tensor_v_2d) is True:
            x = torch.mm(self.in_proj_layer, tensor_q_2d)
            query_out, key_out, value_out = torch.split(x, 3, dim=0)
        
        elif self.same_dim is True:
            _start = 0
            _end = self.hidden_width
            _w = self.in_proj_layer[_start:_end, :]
            query_out = torch.mm(_w, tensor_q_2d.permute(1, 0))
            _start = self.hidden_width
            _end = self.hidden_width * 2
            _w = self.in_proj_layer[_start:_end, :]
            key_out = torch.mm(_w, tensor_k_2d.permute(1, 0))
            _start = self.hidden_width * 2
            _end = None
            _w = self.in_proj_layer[_start:]
            value_out = torch.mm(_w, tensor_v_2d.permute(1, 0))
        else:
            query_out = self.query_layer(tensor_q_2d)
            key_out = self.key_layer(tensor_k_2d)
            value_out = self.value_layer(tensor_v_2d)

        query_out = torch.transpose(query_out, 0, 1)
        key_out = torch.transpose(key_out, 0, 1)
        value_out = torch.transpose(value_out, 0, 1)
        

        query_layer = torch.reshape(query_out, shape_qkv)
        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = torch.reshape(key_out, shape_qkv)
        key_layer = key_layer.permute(0, 2, 1, 3)

        
        attention_scores = torch.matmul(query_layer, key_layer.permute(0, 1, 3, 2))
        attention_scores = torch.multiply(attention_scores, self.scores_mul)   # divide sqrt(d)

        attention_probs = self.softmax(attention_scores)
        
        if self.use_dropout:
            attention_probs = self.dropout(attention_probs)
        
        value_layer = torch.reshape(value_out, shape_qkv)
        value_layer = value_layer.permute(0, 2, 1, 3)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        context_layer = torch.reshape(context_layer, shape_linear)

        context_layer = self.out_proj(context_layer)
        context_layer = torch.reshape(context_layer, shape_return)

        return context_layer


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(q_tensor_width=d_model,
                                            k_tensor_width=d_model,
                                            v_tensor_width=d_model,
                                            hidden_width=d_model,
                                            out_tensor_width=d_model,
                                            num_attention_heads=nhead,
                                            attention_probs_dropout_prob=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(p=1. - dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm((d_model, ), eps=1e-7)
        self.norm2 = LayerNorm((d_model, ), eps=1e-7)
        self.dropout1 = nn.Dropout(p=1.- dropout)
        self.dropout2 = nn.Dropout(p=1.- dropout)

        self.activation = ReLU()
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos=None):
        b, n, d = src.size()
        permute_linear = (b * n, d)
        permute_recover = (b, n, d)
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = torch.reshape(src2, permute_linear)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = torch.reshape(src2, permute_recover)
        src = src + self.dropout2(src2)
        
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(q_tensor_width=d_model,
                                            k_tensor_width=d_model,
                                            v_tensor_width=d_model,
                                            hidden_width=d_model,
                                            out_tensor_width=d_model,
                                            num_attention_heads=nhead,
                                            attention_probs_dropout_prob=dropout)
        self.multihead_attn = MultiheadAttention(q_tensor_width=d_model,
                                                k_tensor_width=d_model,
                                                v_tensor_width=d_model,
                                                hidden_width=d_model,
                                                out_tensor_width=d_model,
                                                num_attention_heads=nhead,
                                                attention_probs_dropout_prob=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(p=1. - dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm((d_model, ), eps=1e-7)
        self.norm2 = LayerNorm((d_model, ), eps=1e-7)
        self.norm3 = LayerNorm((d_model, ), eps=1e-7)
        self.dropout1 = nn.Dropout(p=1.- dropout)
        self.dropout2 = nn.Dropout(p=1.- dropout)
        self.dropout3 = nn.Dropout(p=1.- dropout)

        self.activation = ReLU()

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, tgt, memory, pos=None, query_pos=None):
        b, n, d = tgt.size()
        permute_linear = (b * n, d)
        permute_recover = (b, n, d)
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tensor_v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(tensor_q=self.with_pos_embed(tgt2, query_pos),
                                   tensor_k=self.with_pos_embed(memory, pos),
                                   tensor_v=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = torch.reshape(tgt2, permute_linear)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt2 = torch.reshape(tgt2, permute_recover)
        tgt = tgt + self.dropout3(tgt2)

        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([encoder_layer for i in range(num_layers)])    
    
    def forward(self, src, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, pos=pos)
        
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([decoder_layer for i in range(num_layers)])

    def forward(self, tgt, memory, pos=None, query_pos=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)
        
        return output


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length
        self.device = torch.device('cuda')

        self.position_ids = torch.LongTensor(np.arange(self.seq_length)).to(self.device)      
        self.position_ids = torch.reshape(self.position_ids, (1, self.seq_length))
    
    def forward(self, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, :self.seq_length]
        
        position_embeddings = self.pe(position_ids)

        return position_embeddings


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_dim,
                 patch_dim,
                 num_channels,
                 embedding_dim,
                 num_heads,
                 num_layers,
                 hidden_dim,
                 num_queries,
                 dropout_rate=0,
                 norm=False,
                 mlp=False,
                 pos_every=False,
                 no_pos=False,
                 con_loss=False):
        super(VisionTransformer, self).__init__()
        self.norm = norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim[0]*img_dim[1]) // (patch_dim **2))
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels
        self.out_dim = patch_dim * patch_dim * num_channels
        self.no_pos = no_pos
        self.unf = _unfold_(patch_dim)
        self.fold = _fold_(patch_dim, output_shape=(img_dim[0], img_dim[1]))

        if self.mlp is not True:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                          nn.Dropout(p=1. - dropout_rate),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, self.out_dim),
                                          nn.Dropout(p=1. - dropout_rate))
        
        if num_queries > 0:
            self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)
        else:
            self.query_embed = None

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(self.seq_length, self.embedding_dim, self.seq_length)
        
        self.dropout_layer1 = nn.Dropout(p=1. - dropout_rate)
        self.con_loss = con_loss

    def forward(self, x, query_idx_tensor):
        x = self.unf(x)
        b, n, _ = x.size()
        
        if self.mlp is not True:
            x = torch.reshape(x, (b * n, -1))
            x = self.dropout_layer1(self.linear_encoding(x)) + x
            x = torch.reshape(x, (b, n, -1))
        
        if query_idx_tensor is not None:
            query_embed = torch.tile(torch.reshape(self.query_embed(query_idx_tensor), (1, self.seq_length, self.embedding_dim)), (b, 1, 1))
        else:
            query_embed = None

        if not self.no_pos:
            pos = self.position_encoding()
            x = self.encoder(x + pos)
        else:
            x = self.encoder(x)

        if query_embed is None:
            x = self.decoder(x, x, query_pos=pos)
        else:
            x = self.decoder(x, x, query_pos=query_embed)   # different from original

        if self.mlp is not True:
            x = torch.reshape(x, (b * n, -1))
            x = self.mlp_head(x) + x
            x = torch.reshape(x, (b, n, -1))
        if self.con_loss:
            con_x = x
            x = self.fold(x)
            return x, con_x
        
        x = self.fold(x)

        return x

class IPT_core(nn.Module):
    def __init__(self, img_dim, n_feats, patch_dim, num_heads, num_layers, num_queries, dropout_rate, mlp, pos_every, no_pos):
        super(IPT_core, self).__init__()

        self.body = VisionTransformer(img_dim=img_dim,
                                      patch_dim=3,
                                      num_channels=64,
                                      embedding_dim=n_feats * patch_dim * patch_dim,
                                      num_heads=num_heads,
                                      num_layers=num_layers,
                                      hidden_dim=n_feats * patch_dim * patch_dim * 4,
                                      num_queries=0,
                                      dropout_rate=1,
                                      mlp=False,
                                      pos_every=False,
                                      no_pos=False,
                                      con_loss=False)
    
    def forward(self, x):
        
        res = self.body(x, None)
        res += x

        return x



class _unfold_(nn.Module):
    def __init__(self, kernel_size, stride=-1):
        super(_unfold_, self).__init__()
        if stride == -1:
            self.stride = kernel_size
        self.kernel_size = kernel_size
    
    def forward(self, x):
        N, C, H, W = x.size()   # [N, C, H, W]
        numH = H // self.kernel_size
        numW = W // self.kernel_size
        if numH * self.kernel_size != H or numW * self.kernel_size != W:
            x = x[:, :, :numH * self.kernel_size, :numW * self.kernel_size]
        output_img = torch.reshape(x, (N, C, numH, self.kernel_size, W))   # [N, C, numH, kernel_size, W]

        output_img = output_img.permute(0, 1, 2, 4, 3)    # [N, C, numH, W, kernel_size]
        output_img = torch.reshape(output_img, (N*C, numH, numW, self.kernel_size, self.kernel_size))     # [N*C, numH, numW, kernel_size, kernel_size]
        output_img = output_img.permute(0, 1, 2, 4, 3)    # [N, C, numH, W, kernel_size, kernel_size]
        output_img = torch.reshape(output_img, (N, C, numH * numW, self.kernel_size * self.kernel_size))  # [N, C, numH * numW, kernel_size * kernel_size]
        output_img = output_img.permute(0, 2, 1, 3)       # [N, numH * numW, C, kernel_size * kernel_size]
        output_img = torch.reshape(output_img, (N, numH * numW, -1))   # [N, numH * numW, C * kernel_size * kernel_size]

        return output_img


class _fold_(nn.Module):
    def __init__(self, kernel_size, output_shape=(-1, -1), stride=-1):
        super(_fold_, self).__init__()

        if isinstance(kernel_size, (list, tuple)):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = [kernel_size, kernel_size]
        
        if stride == -1:
            self.stride = self.kernel_size[0]
        self.output_shape = output_shape

    def forward(self, x):
        N, C, L = x.size()   # [N, C, L]
        org_C = L // (self.kernel_size[0] * self.kernel_size[1])
        org_H = self.output_shape[0]
        org_W = self.output_shape[1]
        numH = org_H // self.kernel_size[0]
        numW = org_W // self.kernel_size[1]
        output_img = torch.reshape(x, (N, C, org_C, self.kernel_size[0], self.kernel_size[1]))   # [N, C, orgC, kernel_size, kernel_size]
        output_img = output_img.permute(0, 2, 3, 1, 4)   # [N, orgC, kernel_size, C, kernel_size]
        output_img = torch.reshape(x, (N*org_C, self.kernel_size[0], numH, numW, self.kernel_size[1])) # [N*orgC, kernel_size, numH, numW, kernel_size]
        output_img = output_img.permute(0, 2, 3, 1, 4)   # [N*orgC, numH, numW, kernel_size, kernel_size]

        output_img = torch.reshape(output_img, (N, org_C, org_H, org_W))   # [N, orgC, orgH, orgW]

        return output_img




'''
temp = torch.randn((16, 100, 576)).cuda()
temp_2D = torch.randn((16, 64, 96, 48)).cuda()
WSA = MultiheadAttention(q_tensor_width=576, k_tensor_width=576, v_tensor_width=576, hidden_width=576, out_tensor_width=576).cuda()
Encoder_layer = TransformerEncoderLayer(d_model=576, nhead=1, dim_feedforward=2304).cuda()
Decoder_layer = TransformerDecoderLayer(d_model=576, nhead=1, dim_feedforward=2304).cuda()
Encoder = TransformerEncoder(Encoder_layer, num_layers=2).cuda()
Decoder = TransformerDecoder(Decoder_layer, num_layers=2).cuda()
IPT = IPT_core(img_dim=(temp_2D.size(2), temp_2D.size(3)), n_feats=64, patch_dim=3, num_heads=12, num_layers=1, num_queries=0, dropout_rate=1, mlp=False, pos_every=False, no_pos=False).cuda()
#print(WSA)
#print(WSA(temp, temp, temp).size())
#print(Encoder)
#print(Encoder(temp).size())
#print(Decoder)
#print(Decoder(temp, temp).size())
print(IPT)
print(IPT(temp_2D).size())
'''

