import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(1024)
torch.cuda.manual_seed(1024)

class GRUModel(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=1, batch_first=True, directions=1, dropout=0.0):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=in_dim,
                        hidden_size=out_dim,
                        num_layers=num_layers,
                        batch_first=batch_first,
                        bidirectional=(directions==2),
                        dropout=dropout)
    
    def forward(self, x, lengths):
        # x: [batch_size*turn_num, seq_len, embed_dim]
        # lengths: [batch_size*turn_num]
        # x: [batch_size*candidates_set_size, seq_len, embed_dim]
        # lengths: # [batch_size*candidates_set_size]
        # x: [batch_size * candidates_set_size, turn_num, out_channels *16 * 16]
        # lengths: # [batch_size * candidates_set_size]
    
        seq_len = x.size(1) 
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        x_sort = x.index_select(0, idx_sort)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths_sort = lengths_sort.to("cpu") # pack_padded_sequence的lengths参数必须是cpu上的参数

        #有的batch中，对话的轮数可能都小于max_turn_num，前面用全padding将其进行了填充，需要将这部分的有效长度置为1
        for i in range(len(lengths_sort)):
            if lengths_sort[i] == 0:
                lengths_sort[i] = 1

        x_pack = nn.utils.rnn.pack_padded_sequence(x_sort, lengths_sort, batch_first=True)
        out_pack, h_n =self.gru(x_pack)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=True) 

        out_unsort = out.index_select(0, idx_unsort)

        # out的文本长度为该batch中最长文本的有效长度
        # 某个batch中最长文本的有效长度都可能小于预先设置的max_seq_len，为了后续计算，需要将其填充
        if out_unsort.size(1) < seq_len:
            pad_tensor = Variable(torch.zeros(out_unsort.size(0), seq_len - out_unsort.size(1), out_unsort.size(2))).to(out_unsort)
            out_unsort = torch.cat([out_unsort, pad_tensor], 1)

        return out_unsort


class CNNModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=8):
        super(CNNModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), bias=False)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
    
    def forward(self, x):
        # x: [batch_size * candidates_set_size * turn_num, 2, seq_length, seq_length]
        out = self.conv(x) # [batch_size * candidates_set_size, turn_num, out_channels, W_conv_out, H_conv_out]
        out = self.max_pool(out) # [batch_size * candidates_set_size, turn_num, out_channels, W_pool_out, H_pool_out]
        return out

class FeatureFusion(nn.Module):
    def __init__(self, method = "last", turn_num=None):
        super(FeatureFusion, self).__init__()
        self.method = method
        if method == "static":
            self.weight = nn.Parameter(torch.FloatTensor(np.random.randn(turn_num)), requires_grad=True)
    
    def forward(self, x, lengths, att_status=None):
        # x: [batch_size * candidates_set_size, turn_num, hidden_size]
        # lengths: [batch_size * candidates_set_size]
        
        if self.method == "last":
            for b_id, length in enumerate(lengths):
                x[b_id, -1, :] = x[b_id, length-1, :]
            return x[:, -1, :] # [batch_size * candidates_set_size, hidden_size]
        elif self.method == "static":
            for b_id, length in enumerate(lengths):
                x[b_id, length:, :] = 0.0
            return torch.matmul(self.weight, x) # [batch_size * candidates_set_size, hidden_size]
        elif self.method == "dynamic":
            # TODO
            pass
        else:
            raise ValueError(f"Feature-fusion method {self.method} not supported")



class SMN(nn.Module):
    def __init__(self, word_embeddings=None):
        super(SMN, self).__init__()
        self.max_turn_num = 1
        self.max_seq_len = 32
        self.vocab_size = 21128
        self.embed_dim = 200
        self.hidden_size = 50
        self.candidates_set_size = 14
        self.out_channels = 8
        self.fusion_method = "last"
        self.k = 50
        self.keep_ref_num = 10
        self.pad_token_id = 0
        # self.id2word = {}
        self.loss_fuc = nn.KLDivLoss()
        # for k, v in vocab.items():
        #     self.id2word[v] = k
        # self.device = torch.device('cuda:0,1')
        self.dropout_rate = 0.1
        if word_embeddings:
            self.embedding = nn.Embedding(num_embeddings=len(word_embeddings), embedding_dim=self.embed_dim, padding_idx=0,
                                           _weight=torch.FloatTensor(word_embeddings))
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        
        # self.gru_context = GRUModel(self.embed_dim, self.hidden_size)
        # self.gru_response = GRUModel(self.embed_dim, self.hidden_size)
        self.gru_context = nn.GRU(self.embed_dim, self.hidden_size, batch_first=True)
        self.gru_response = nn.GRU(self.embed_dim, self.hidden_size, batch_first=True)
        self.A = nn.Parameter(torch.randn((self.hidden_size, self.hidden_size)))
        
        self.cnn_layer = CNNModel(2, self.out_channels)

        # self.gru2 = GRUModel(self.out_channels * 10 * 10, self.hidden_size)
        self.gru2 = nn.GRU(self.out_channels * 10 * 10, self.hidden_size, batch_first=True)

        self.feature_fusion = FeatureFusion(self.fusion_method)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.classifier = nn.Sequential(
            nn.Linear(self.candidates_set_size * self.hidden_size, 100),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(100, self.candidates_set_size),
            nn.Softmax()
        )

    def extract_M(self, contexts_indices, candidates_indices):
        '''
            contexts_indices: [batch_size, turn_num, seq_length]
            candidates_indices: [batch_size, candidates_set_size, seq_length]
            M: [batch_size * candidates_set_size, turn_num, seq_length, seq_length]
        '''
        batch_size = contexts_indices.size(0)
        contexts_indices = contexts_indices.unsqueeze(1)
        # print(contexts_indices.shape)
        contexts_seq_len = (contexts_indices != 0).sum(dim=-1).long() # [batch_size, turn_num]
        # contexts_turns_num = (contexts_seq_len != 0).sum(dim=-1).long() # [batch_size]
        candidates_seq_len = (candidates_indices != 0).sum(dim=-1).long() # [batch_size, candidates_set_size]
        # candidates_set_size = (candidates_seq_len != 0).sum(dim=-1).long() # [batch_size]

        contexts_embed = self.embedding(contexts_indices) # [batch_size, turn_num, seq_length, embed_dim]
        contexts_embed = self.dropout(contexts_embed)

        candidates_embed = self.embedding(candidates_indices) # [batch_size, candidates_set_size, seq_length, embed_dim]
        candidates_embed = self.dropout(candidates_embed)

        contexts_all_inputs_len = contexts_seq_len.view(-1) # [batch_size*turn_num]
        candidates_all_inputs_len = candidates_seq_len.view(-1) # [batch_size*candidates_set_size]

        contexts_inputs = contexts_embed.view(-1, self.max_seq_len, self.embed_dim) # [batch_size*turn_num, seq_len, embed_dim]
        candidates_inputs = candidates_embed.view(-1, self.max_seq_len, self.embed_dim) # [batch_size*candidates_set_size, seq_len, embed_dim]

        # contexts_hiddens = self.gru_context(contexts_inputs, contexts_all_inputs_len) # [batch_size*turn_num, seq_len, hidden_size]
        contexts_hiddens,_ = self.gru_context(contexts_inputs)
        candidates_hiddens,_ = self.gru_response(candidates_inputs) # [batch_size*candidates_set_size, seq_len, hidden_size]

        contexts_hiddens = contexts_hiddens.view(-1, self.max_turn_num, self.max_seq_len, self.hidden_size) # [batch_size, turn_num, seq_length, hidden_size]
        candidates_hiddens = candidates_hiddens.view(-1, self.candidates_set_size, self.max_seq_len, self.hidden_size) # [batch_size, candidates_set_size, seq_length, hidden_size]

        M1 = torch.einsum("btph, bcqh -> btcpq", contexts_embed, candidates_embed) # [batch_size, turn_num, candidates_set_size, seq_length, seq_length]
        # M1 = M1.permute(0, 2, 1, 3, 4).contiguous() # [batch_size, candidates_set_size, turn_num, seq_length, seq_length]
        # M1 = M1.view(-1, self.candidates_set_size, self.max_seq_len, self.max_seq_len) # [batch_size * candidates_set_size, turn_num, seq_length, seq_length]

        contexts_hiddens = torch.einsum("btij, jk -> btik", contexts_hiddens, self.A)
        # print(contexts_hiddens.shape, candidates_hiddens.shape)
        M2 = torch.einsum("btph, bcqh -> btcpq", contexts_hiddens, candidates_hiddens) # [batch_size, turn_num, candidates_set_size, seq_length, seq_length]
        # M2 = M2.permute(0, 2, 1, 3, 4).contiguous() # [batch_size, candidates_set_size, turn_num, seq_length, seq_length]
        # M2 = M2.view(-1, self.candidates_set_size, self.max_seq_len, self.max_seq_len) # [batch_size * candidates_set_size, turn_num, seq_length, seq_length]
        M = M1.squeeze() + M2.squeeze()   #[batch_size, candidates_set_size, seq_length, seq_length]
        candidates_indices = candidates_indices.squeeze() #[batch_size, candidates_set_size, seq_length]
        
        if candidates_indices.dim() == 2:
            M = M.unsqueeze(0)
            candidates_indices = candidates_indices.unsqueeze(0)
        M = torch.sum(M, dim=3)
        M = M.view(batch_size, -1)
        candidates_indices = candidates_indices.view(batch_size, -1)
        kvalue, _ = M.topk(self.k)
        mask = M.gt(kvalue[:,-1].unsqueeze(1))
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
        # M = M.cpu().numpy().tolist()
        # contexts_indices = contexts_indices.cpu().numpy().tolist()
        # candidates_indices = candidates_indices.cpu().numpy().tolist()
        # keep_words = []
        # other_words = []
        # for context, scores in zip(contexts_indices[1:], M):
        #     # keep_word = []
        #     # other_word = []
        #     context_word = ''.join(list(map(lambda x: self.id2word[x], context))).replace('<PAD>', '')


            # for word, score in zip(context, scores):
            #     if score > 0:
            #         if word != 0:
            #             keep_word.append(self.id2word[word])
            #     else:
            #         if word != 0:
            #             other_word.append(self.id2word[word])
            # keep_words.append(' '.join(keep_word))
        
        # print('keep_word:{}, other_word: {}, context: {}, post: {}, resp: {}'.format(' '.join(keep_word), ' '.join(other_word), context_word, post, resp))

    def forward(self, contexts_indices, candidates_indices, y_dev):
        # contexts_indices: [batch_size, turn_num, seq_length] post
        # candidates_indices: [batch_size, candidates_set_size, seq_length] ref
        batch_size = contexts_indices.size(0)
        contexts_indices = contexts_indices.unsqueeze(1)
        contexts_seq_len = (contexts_indices != 0).sum(dim=-1).long() # [batch_size, turn_num]
        contexts_turns_num = (contexts_seq_len != 0).sum(dim=-1).long() # [batch_size]
        candidates_seq_len = (candidates_indices != 0).sum(dim=-1).long() # [batch_size, candidates_set_size]
        # candidates_set_size = (candidates_seq_len != 0).sum(dim=-1).long() # [batch_size]

        contexts_embed = self.embedding(contexts_indices) # [batch_size, turn_num, seq_length, embed_dim]
        contexts_embed = self.dropout(contexts_embed)

        candidates_embed = self.embedding(candidates_indices) # [batch_size, candidates_set_size, seq_length, embed_dim]
        candidates_embed = self.dropout(candidates_embed)

        contexts_all_inputs_len = contexts_seq_len.view(-1) # [batch_size*turn_num]
        candidates_all_inputs_len = candidates_seq_len.view(-1) # [batch_size*candidates_set_size]

        contexts_inputs = contexts_embed.view(-1, self.max_seq_len, self.embed_dim) # [batch_size*turn_num, seq_len, embed_dim]
        candidates_inputs = candidates_embed.view(-1, self.max_seq_len, self.embed_dim) # [batch_size*candidates_set_size, seq_len, embed_dim]

        contexts_hiddens,_ = self.gru_context(contexts_inputs) # [batch_size*turn_num, seq_len, hidden_size]

        candidates_hiddens,_ = self.gru_response(candidates_inputs) # [batch_size*candidates_set_size, seq_len, hidden_size]

        contexts_hiddens = contexts_hiddens.view(-1, self.max_turn_num, self.max_seq_len, self.hidden_size) # [batch_size, turn_num, seq_length, hidden_size]
        candidates_hiddens = candidates_hiddens.view(-1, self.candidates_set_size, self.max_seq_len, self.hidden_size) # [batch_size, candidates_set_size, seq_length, hidden_size]

        M1 = torch.einsum("btph, bcqh -> btcpq", contexts_embed, candidates_embed) # [batch_size, turn_num, candidates_set_size, seq_length, seq_length]
        # M1 = M1.permute(0, 2, 1, 3, 4).contiguous() # [batch_size, candidates_set_size, turn_num, seq_length, seq_length]
        # M1 = M1.view(-1, self.candidates_set_size, self.max_seq_len, self.max_seq_len) # [batch_size * candidates_set_size, turn_num, seq_length, seq_length]

        contexts_hiddens = torch.einsum("btij, jk -> btik", contexts_hiddens, self.A)
        # print(contexts_hiddens.shape, candidates_hiddens.shape)
        M2 = torch.einsum("btph, bcqh -> btcpq", contexts_hiddens, candidates_hiddens) # [batch_size, turn_num, candidates_set_size, seq_length, seq_length]
        # M2 = M2.permute(0, 2, 1, 3, 4).contiguous() # [batch_size, candidates_set_size, turn_num, seq_length, seq_length]
        # M2 = M2.view(-1, self.candidates_set_size, self.max_seq_len, self.max_seq_len) # [batch_size * candidates_set_size, turn_num, seq_length, seq_length]
        # self.extract_M(M1, contexts_indices, candidates_indices)
        # self.extract_M(M2, contexts_indices, candidates_indices)
        M = [M1, M2]
        M = torch.stack(M, dim=2).contiguous() # [batch_size * candidates_set_size, turn_num, 2, seq_length, seq_length]
        M = M.view(-1, 2, self.max_seq_len, self.max_seq_len) # [batch_size * candidates_set_size * turn_num, 2, seq_length, seq_length]
        M = self.cnn_layer(M)
        # print("#" * 20)
        # print("M shape: ", M.shape)
        # print("#" * 20)
        # 16与设置的文本最大长度和cnn的实现方式相关，可计算得到
        M = M.view(-1, self.candidates_set_size, self.out_channels, 10, 10) # [batch_size * turn_num, candidates_set_size, out_channels, 16, 16]
        size_0 = M.size(0)
        M = M.view(size_0, self.candidates_set_size, -1) # [batch_size * turn_num, candidates_set_size, out_channels *16 * 16]

        # contexts_turns_num_extend = [copy.deepcopy(contexts_turns_num) for _ in range(self.candidates_set_size)]
        # contexts_turns_num_extend = torch.stack(contexts_turns_num_extend, dim=1) # [batch_size, candidates_set_size]
        # contexts_turns_num_extend = contexts_turns_num_extend.view(-1) # [batch_size * candidates_set_size]
        H,_ = self.gru2(M) # [batch_size * turn_num, candidates_set_size , hidden_size]
        # if self.fusion_method != "dynamic":
        #     F = self.feature_fusion(H, contexts_turns_num_extend).contiguous() # [batch_size * candidates_set_size, hidden_size]
        # else:
        #     F = self.feature_fusion(H, contexts_turns_num_extend, att_status = contexts_hiddens).contiguous()

        flatten = H.contiguous().view(batch_size, -1) #[batch_size, candidates_set_size*hidden_size]
        logits = self.classifier(flatten)
        logits = logits.log()
        loss = self.loss_fuc(logits, y_dev)

        # probs = torch.nn.functional.sigmoid(logits)

        return loss

    def load_weight(self, path):
        self.load_state_dict(state_dict=torch.load(path))
        if torch.cuda.is_available(): self.cuda()
