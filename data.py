from torch.utils.data import Dataset
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence

class BaseDataset(Dataset):
    def __init__(self, dataset_name, dataset_type, tokenizer, hip=2, comet_type='old'):
        '''Load and preprocess data'''
        super(BaseDataset, self).__init__()
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.hip = hip # determines how far on either side of the current utterance to connect with edges

        # load the utterances, speaker labels, and emotion labels
        dataset_path = 'features/' + dataset_name + '_graph_hip' + str(self.hip) + '_new.pkl'
        data = pickle.load(open(dataset_path, 'rb'), encoding='utf-8')
        data = data[dataset_type] # select train, test, dev

        self.utt = data[0] # 1038 conversations; each conversation contains several utterances. Text format.
        # self.utt[0][0:3]
        #["also I was the point person on my company's transition from the KL-5 to GR-6 system.",
        # "You must've had your hands full.",
        # 'That I did. That I did.',
        # ... ] length=14

        self.label = data[1] # self.label[0] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 1, 0]
        self.spk = data[2] # self.spk[0] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

        graph = data[3]
        self.edge_index = graph['edge_index'] # these are the connects for each utterance to other speakers in the future
                                              # or self in the past present and future
        # for instance, utterance #0 connects to the first 4 utterances ahead of it
        # self.edge_index[0][:,self.edge_index[0][0,:]==0] =
        #array([[0, 0, 0, 0, 0],
        #       [0, 1, 2, 3, 4]])

        self.edge_type = graph['edge_type'] # these are oWant, xWant etc.
        # for utterance #0, these edges are xEffect for self connection, oWant for connection to other speaker
        # self.edge_type[0][0:5] = ['xEffect', 'oWant', 'xWant', 'oWant', 'xWant']

        self.utt_id = []
        self.wmask = []

        # tokenize the utterances
        for conv in self.utt:
            input_ids = []
            attention_mask = []
            for u in conv:
                encoded_inputs = tokenizer(u, truncation=True, max_length=52)
                input_ids.append(torch.tensor(encoded_inputs['input_ids'], dtype=torch.long))
                attention_mask.append(torch.tensor(encoded_inputs['attention_mask'], dtype=torch.float))
            self.utt_id.append(input_ids)
            self.wmask.append(attention_mask)

        # load common-sense edge features from COMET
        if comet_type=='old':
            edge_attr_path = 'features/' + dataset_name + '_edge_attr_' + dataset_type + '_1.pkl'
        else:
            edge_attr_path = 'features/' + dataset_name + '_edge_attr_' + dataset_type + '_chris.pkl'
        self.cmsk = pickle.load(open(edge_attr_path, 'rb'), encoding='utf-8') # this contains 4 embeddings per utterance; xWant, oWant, etc. some are not used
        self.length = len(self.label)

    def __getitem__(self, item):
        '''Used by PyTorch's data loader to retrieve a single entry in the dataset'''
        selected_utt = self.utt_id[item]
        selected_label = self.label[item]
        selected_mask = self.wmask[item]
        selected_uttm = [1] * len(selected_label)
        selected_spk = self.spk[item]
        selected_edge_index = self.edge_index[item]
        selected_edge_type = self.edge_type[item]
        selected_cmsk = self.cmsk[item]

        # process edges for the a particular utterance
        # select the relevant commonsense embedding (xNeed, xWant etc.) for each edge
        selected_edge_attr = []
        selected_edge_relation_binary = []
        selected_edge_relation = []
        for i in range(selected_edge_index.shape[1]):
            edge_i = selected_edge_index[0, i]
            eg_tp = selected_edge_type[i]
            selected_edge_attr.append(torch.tensor(selected_cmsk[edge_i][eg_tp]))
            edge_j = selected_edge_index[1, i]
            selected_edge_relation_binary.append(1 if eg_tp == 'oWant' else 0)
            if edge_j <= edge_i:
                selected_edge_relation.append(2)
            else:
                if eg_tp == 'xWant':
                    selected_edge_relation.append(0)
                else:
                    selected_edge_relation.append(1)

        # pad the utterance and mask
        selected_utt = pad_sequence(selected_utt, batch_first=True, padding_value=0) # utterance tokens, batch size x max length
        selected_mask = pad_sequence(selected_mask, batch_first=True, padding_value=0) # utterance mask

        # convert to tensors
        selected_label = torch.tensor(selected_label, dtype=torch.long) # emotional labels
        selected_uttm = torch.tensor(selected_uttm, dtype=torch.float) # used for counting utterance
        selected_spk = torch.tensor(selected_spk, dtype=torch.float) # speaker label
        selected_edge_index = torch.tensor(selected_edge_index, dtype=torch.long) #  2 x number of edges (for the head, tail)
        selected_edge_attr = torch.stack(selected_edge_attr, dim=0)  # embeddings for those edges
        selected_edge_relation_binary = torch.tensor(selected_edge_relation_binary, dtype=torch.long) # not used for MELD
        selected_edge_relation = torch.tensor(selected_edge_relation, dtype=torch.long) # type of relation (also not used)

        return selected_utt, selected_mask, selected_label, selected_uttm, selected_spk, selected_edge_index, selected_edge_attr, selected_edge_relation_binary, selected_edge_relation

    def __len__(self):
        return self.length

def collate_fn_batch(batch):
    '''Unpacks batch and reorganizes. Used by PyTorch's DataLoader to process a batch. '''
    utt, mask, label, uttm, spk, edge_index, edge_attr, edge_rel_b, edge_rel = [], [], [], [], [], [], [], [], []
    for d in batch:
        utt.append(d[0])
        mask.append(d[1])
        label.append(d[2])
        uttm.append(d[3])
        spk.append(d[4])
        edge_index.append(d[5])
        edge_attr.append(d[6])
        edge_rel_b.append(d[7])
        edge_rel.append(d[8])
    return utt, mask, label, uttm, spk, edge_index, edge_attr, edge_rel_b, edge_rel
