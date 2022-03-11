import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv
from transformers import RobertaModel
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class FeedForward2Layer(nn.Module):
    '''Extra feedforward network between graph network udpdates'''
    def __init__(self, input_dim, ff_dim, dropout):
        super(FeedForward2Layer, self).__init__()
        self.input_dim = input_dim
        self.ff_dim = ff_dim
        self.linear1 = nn.Linear(input_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        ss = self.norm1(src)
        ss2 = self.linear2(self.dropout1(F.relu(self.linear1(ss))))
        ss = ss + self.dropout2(ss2)
        ss = self.norm2(ss)
        return ss

class GNNWrapper(nn.Module):
    '''Wrapper for graph neural network. Loops over multiple layers and adds feedforward layer in between.
       Nodes are embedded utterances; edges embedded common sense features.
    '''
    def __init__(self, in_channels, ff_dim, out_channels, heads, edge_dim, num_layers, edge_starting_dim=768, dropout=0.1, beta=True, bias=True, root_weight=True):
        super(GNNWrapper, self).__init__()
        assert in_channels == heads * out_channels, 'in_channels must equal heads * out_channels' # preserces dimensionality of x
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_layers = num_layers

        # linear map for edge embedding (e.g. COMET's 768dim --> 200dim)
        self.mapping = nn.Linear(edge_starting_dim, edge_dim)
        self.edge_dim = edge_dim

        # single layer of the graph transformer
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=TransformerConv#torch_geometric.nn.conv.TransformerConv
        # each node is represented by a 200dim vector, which is updated by
        # (i) a linear mapping from itself (in the previous layer)
        # (ii) weighted sum of its neighbors representations
        #   the weights are calculated using an attention mechanism (with multiple heads),
        #   where the attention is given by the similarity between the node's and the neighbor's representations (after linear transform)
        # the representations passed between neighbors are also modified adding an edge representation (also dim 200)
        # Note: can be batched with disconnected graphs, as specified by edge index during the forward call
        conv = TransformerConv(in_channels=in_channels,
                               out_channels=out_channels,
                               heads=heads, concat=True,
                               beta=beta, dropout=dropout,
                               edge_dim=edge_dim, bias=bias,
                               root_weight=root_weight)
        # additional 2 layer net to update each node's representation
        ff = FeedForward2Layer(in_channels, ff_dim, dropout)
        self.convs = _get_clones(conv, num_layers)
        self.ffnet = _get_clones(ff, num_layers)

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.mapping(edge_attr)
        # update the graph network N times, where N=num_layers
        for i in range(self.num_layers):
            x = self.convs[i](x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.ffnet[i](x)
        return x

class UtteranceEncoder(nn.Module):
    '''Embedds utterances using RoBERTa'''
    def __init__(self, model_type, out_dim):
        super(UtteranceEncoder, self).__init__()
        self.model_type = model_type
        self.out_dim = out_dim
        self.encoder = RobertaModel.from_pretrained('roberta-large')
        self.hidden_dim = 1024
        self.mapping = nn.Linear(self.hidden_dim, out_dim)

    def forward(self, conversations, mask, use_gpu=True):
        if isinstance(conversations, torch.Tensor):
            output = self.encoder(conversations, mask)
            hidden_state = output['last_hidden_state']
            sent_emb = self.mapping(torch.max(hidden_state, dim=1)[0]) #max pooling
        elif isinstance(conversations, list): # batch mode
            # concatenate conversations before feeding to roberta
            max_utt_len = max([int(c.size(1)) for c in conversations])
            conversation = list()
            msk = list()
            for c, m in zip(conversations, mask):
                conversation.append(torch.cat((c, torch.zeros((c.size(0), max_utt_len-c.size(1)), dtype=torch.long)), dim=1))
                msk.append(torch.cat((m, torch.zeros((m.size(0), max_utt_len-m.size(1)))), dim=1))

            conversation = torch.cat(conversation, dim=0) # total number utterances x max utterance length
            msk = torch.cat(msk, dim=0)
            if use_gpu:
                conversation = conversation.cuda()
                msk = msk.cuda()

            output = self.encoder(conversation, msk)
            hidden_state = output['last_hidden_state'] # total number utterances x max utterance length x hidden_state_dim
            sent_emb = self.mapping(torch.max(hidden_state, dim=1)[0])
            # max pooling --> total number of utterances x  hidden_state_dim --> lower dimensional embedding (e.g. 200)

        return sent_emb

    def __repr__(self):
        return '{}({}, mode={}, out_dim={})'.format(self.__class__.__name__,
                                                    self.encoder.__class__.__name__,
                                                    self.mode,
                                                    self.out_dim)

class CombinedModel(nn.Module):
    '''Combines the RoBERTa-based utterance encoder with the graph neural network for taking into account common sense knowledge'''
    def __init__(self, encoder_type='roberta-large', sent_dim=200, ff_dim=200, nhead=4,
                 edge_dim=200, num_layer=2, num_class=7, edge_starting_dim=768):
        super(CombinedModel, self).__init__()
        self.utterenc = UtteranceEncoder(encoder_type, sent_dim)
        self.gnn = GNNWrapper(sent_dim,   # in_channels (node dimensionality)
                         ff_dim,     # ff_dim (final linear mapping)
                         sent_dim // nhead, # out_channels (size of attention head)
                         nhead, # attention heads for edge weights
                         edge_dim, # edge embedding dimensionality
                         num_layer, # number of times to iterate over graph
                         edge_starting_dim=edge_starting_dim)
        self.classifier = nn.Linear(sent_dim, num_class) # final classifier; linear layer because logits used for loss

    def forward(self, conversations, masks, conv_len=None, edge_indices=None, edge_attrs=None, use_gpu=True):
        sent_emb = self.utterenc(conversations, masks, use_gpu)
        edge_attr = torch.cat(edge_attrs, dim=0)
        batch = []
        cumbatch = []
        count = 0
        for i, l in enumerate(conv_len):
            num_edge = int(edge_indices[i].size(1))
            batch += [i] * num_edge
            cumbatch += [count] * num_edge
            count += l
        batch = torch.tensor(batch, dtype=torch.long)
        cumbatch = torch.tensor([cumbatch, cumbatch], dtype=torch.long)
        edge_index = torch.cat(edge_indices, dim=1)
        edge_index = edge_index + cumbatch
        if use_gpu:
            edge_index = edge_index.cuda()
            edge_attr = edge_attr.cuda()
        mental_emb = self.gnn(sent_emb, edge_index, edge_attr)
        logits = self.classifier(mental_emb)
        return logits
