import torch
import math
import torch.nn as nn
import torch.nn.functional as F
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.1, activation='relu'):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, x):
        b, t, l, h = x.shape
        output = x.view(b*t, l, h)
        output = output.transpose(1, 2)
        output = self.w2(self.activation(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        return output.view(b, t, l, h)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        if x.dim() == 3:
            x = x + self.pe[:, :x.size(1)]
        else:
            new_pe = self.pe.unsqueeze(1)
            x = x + new_pe[:,:,:x.size(2)]
        return self.dropout(x)

class SNet(nn.Module):
    def __init__(self, word_embeddings=None):
        super(SNet, self).__init__()
        self.max_turn_num = 1
        self.max_len = 32
        self.vocab_size = 21128
        self.embed_dim = 200
        self.hidden_size = 50
        self.candidates_set_size = 14
        self.out_channels = 8
        self.fusion_method = "last"
        self.k = 50
        self.keep_ref_num = 10
        self.pad_token_id = 0

        self.pseudo_loss_fuc = nn.KLDivLoss()
        self.loss_fuc = nn.NLLLoss()

        self.dropout_rate = 0.1
        self.query_layer = nn.Linear(self.embed_dim, self.hidden_size)
        self.key_layer = nn.Linear(self.embed_dim, self.hidden_size)
        self.value_layer = nn.Linear(self.embed_dim, self.hidden_size)
        self.fc = PositionalWiseFeedForward(self.hidden_size, self.embed_dim, self.dropout_rate, 'gelu')

        if word_embeddings:
            self.embedding = nn.Embedding(num_embeddings=len(word_embeddings), embedding_dim=self.embed_dim, padding_idx=0,
                                           _weight=torch.FloatTensor(word_embeddings))
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.position_layer = PositionalEncoding(self.embed_dim, self.dropout_rate, self.max_len)
        self.atten_layer_norm = nn.LayerNorm(self.embed_dim)
        self.con_layer_norm = nn.LayerNorm(self.hidden_size)
        self.attn_drop = nn.Dropout(self.dropout_rate)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.candidates_set_size * self.hidden_size, 4 * self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(4 * self.hidden_size, self.candidates_set_size),
            nn.Softmax()
        )

    def padding_mask(self, seq_k, seq_q, pad_token=0):
        # seq_k [B, T, L_k], seq_q [B, L_q]
        len_k = seq_k.size(2)
        len_q = seq_q.size(1)
        pad_mask = seq_k.eq(pad_token)
        pad_mask = pad_mask.unsqueeze(2).expand(-1, -1, len_q, len_k)  # shape [B, T, L_q, L_k]
        return pad_mask
    
    def cross_attention(self, q, k, v, mask=None, return_att=False):
        q = self.query_layer(q)
        k = self.key_layer(k)
        v = self.value_layer(v)

        att = torch.einsum('blh, btsh -> btls', q, k)
        att = att * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            att = att.masked_fill(mask == True, -1e10)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        if return_att:
            return att
        else:
            y = att @ v # (B, T, L, S) x (B, T, S, hs) -> (B, T, L, hs)
            y = self.proj(y)
            return y

    def forward(self, post, refs, label, pseudo=False):
        '''
        post: [batch_size, max_len]
        refs: [batch_size, candidate_num, max_len]
        label: [batch_size, candidate_num]
        '''
        batch_size = post.size(0)
        post_embedd = self.embedding(post)
        refs_embedd = self.embedding(refs)
        refs_mask = self.padding_mask(refs, post, self.pad_token_id)

        post_embedd = self.dropout(post_embedd)
        refs_embedd = self.dropout(refs_embedd)

        post_embedd = self.position_layer(post_embedd)
        refs_embedd = self.position_layer(refs_embedd)

        post = self.atten_layer_norm(post_embedd)
        y = self.cross_attention(post, refs_embedd, refs_embedd, refs_mask)
        y = self.con_layer_norm(y)
        fcout = self.fc(y)
        y = self.dropout(y + fcout)
        # y = self.dropout(y + refs_embedd) #(B, T, L, hs)
        y = torch.sum(y, dim=2)
        logits = self.classifier(y.contiguous().view(batch_size, -1))
        logits = logits.log()
        if pseudo:
            loss = self.pseudo_loss_fuc(logits, label)
        else:
            loss = self.loss_fuc(logits, label)
        return loss

    def extract_M(self, post, refs):
        '''
            post: [batch_size, turn_num, seq_length]
            refs: [batch_size, candidates_set_size, seq_length]
            M: [batch_size * candidates_set_size, turn_num, seq_length, seq_length]
        '''
        batch_size = contexts_indices.size(0)
        contexts_indices = contexts_indices.unsqueeze(1)

        post_embedd = self.embedding(post)
        refs_embedd = self.embedding(refs)
        refs_mask = self.padding_mask(refs, post, self.pad_token_id)

        post_embedd = self.dropout(post_embedd)
        refs_embedd = self.dropout(refs_embedd)

        post_embedd = self.position_layer(post_embedd)
        refs_embedd = self.position_layer(refs_embedd)

        post = self.atten_layer_norm(post_embedd)
        att = self.cross_attention(post, refs_embedd, refs_embedd, refs_mask, True) #[b, t, l, s]
        att = torch.sum(att, dim=2)

        att = att.view(batch_size, -1)
        candidates_indices = refs.view(batch_size, -1)
        kvalue, _ = att.topk(self.k)
        mask = att.gt(kvalue[:,-1].unsqueeze(1))
        pad_mask = candidates_indices.ne(0)
        cls_mask = candidates_indices.ne(101)
        eos_mask = candidates_indices.ne(102)
        mask = mask * pad_mask * cls_mask * eos_mask
        # keep_words = candidates_indices.masked_fill(mask==True, 0) #pad_id
        # keep_words = candidates_indices.masked_scatter(mask) #pad_id
        for idx in range(batch_size):
            keep_word = torch.masked_select(candidates_indices[idx,:], mask[idx,:])[:self.keep_ref_num]
            if keep_word.size(0) < self.keep_ref_num:
                keep_word = torch.cat((torch.zeros(self.keep_ref_num-keep_word.size(0)).type_as(keep_word), keep_word), dim=-1)
            keep_word = keep_word.unsqueeze(0)
            if idx == 0:
                keep_words = keep_word
            else:
                keep_words = torch.cat((keep_words,keep_word), dim=0)
        return keep_words

    def load_weight(self, path):
        self.load_state_dict(state_dict=torch.load(path))
        if torch.cuda.is_available(): self.cuda()
