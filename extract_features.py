import torch
from comet.comet import Comet
import pickle
from tqdm import tqdm

if __name__ == '__main__':

    # load comet
    comet = Comet('/home/cgagne/SKAIG-ERC_Repro/pretrained/comet-atomic_2020_BART')
    comet.model.zero_grad()

    # get data
    dataset_path = '/home/cgagne/SKAIG-ERC_Repro/features/MELD_graph_hip2_new.pkl'
    data_all = pickle.load(open(dataset_path, 'rb'), encoding='utf-8')

    for split in ['train','dev','test']:
        print(split)

        data = data_all[split]
        conversations = data[0]
        relations = ['xWant', 'oWant', 'xIntent', 'xEffect']

        features_all = []
        for c in tqdm(range(len(conversations))):
            features_for_conv=[]
            utterances = conversations[c]

            for utt in utterances:
                features_for_rel = {}

                for rel in relations:
                    queries = ["{} {} [GEN]".format(utt, rel)]
                    hidden_states = comet.get_hidden(queries, rel)
                    features_for_rel[rel] = hidden_states

                features_for_conv.append(features_for_rel)
            features_all.append(features_for_conv)

        # save
        pickle.dump(features_all, open( f"features/MELD_edge_attr_{split}_chris.pkl", "wb" ) )
