import random
import argparse
import torch
import pickle
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time
from tqdm import tqdm

from data import BaseDataset, collate_fn_batch
from model import CombinedModel

def get_loaders(dataset_name, batch_size=4, pretrained_model='roberta-large', valid=0.1, shuffle=True, comet_type='old'):

    tokenizer = AutoTokenizer.from_pretrained('pretrained/'+pretrained_model)

    # preprocessing conversations (e.g. tokenizer), loading edge features, providing data entry getters
    trainset = BaseDataset(dataset_name, 'train', tokenizer, comet_type=comet_type)
    testset = BaseDataset(dataset_name, 'test', tokenizer, comet_type=comet_type)
    devset = BaseDataset(dataset_name, 'dev', tokenizer, comet_type=comet_type)

    # batch data, shuffle etc.
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn_batch)
    dev_loader = DataLoader(devset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn_batch)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn_batch)

    return train_loader, dev_loader, test_loader

def train(model, loss_func, trainloader, devloader, testloader, n_epochs, optimizer,
          scheduler, training_step, model_path, log_path, metric_path, use_gpu):
    model.train()

    f = open(log_path, 'a+', encoding='utf-8')

    train_loss_list = []
    train_f1_list = []
    dev_f1_list = []
    test_f1_list = []

    best_fscore = 0
    best_mf1 = 0
    best_accuracy = 0
    best_report = None

    best_test_fscore = 0
    best_test_mf1 = 0
    best_test_accuracy = 0
    best_test_report = None

    step = 0
    early_stopping_step = 0

    for epoch in range(n_epochs):
        losses = []
        preds = []
        labels = []
        masks = []
        num_utt = 0
        print('Epoch {} start: '.format(epoch + 1))
        if step > training_step:
            break
        start_time = time.time()
        for data in tqdm(trainloader):

            # unpack batch data
            textf, wrdm, label, uttm, spkm, edge_index, edge_attr, _ , _ = data
            # textf: list of conversations, each with one to up to several utterances
                # the list is length=(batch_size)
                # each conversation is a tensor of tokens with size=(number of utterances, length longest utterance)
            # wrdm: list of mask tokens tensors for utterances padded to longest utterance in conversation
                # each mask tensor is size=(number of utterances, length longest utterance)
            # label: list of emotion labels tensors
                # each tensor is size=(number of utterances)
            # uttm: list of 1's tensors; used for counting the conversation length
                # each tensor is size=(number of utterances)
            # spkm: list of speaker indicators
                # each tensor is size=(number of utterances)
            # edge_idx: specifies which utterances are connected to one another
                # each tensor is size=(2, number of edges)
            # edge_attr: commonsense feature embeddings for each edge
                # each tensor is size=(number of edge, edge dimensionality)

            conv_len = [int(torch.sum(um).item()) for um in uttm]

            # get predictions
            logits = model(textf, wrdm, conv_len, edge_index, edge_attr, use_gpu)

            # calculate loss
            label = torch.cat(label, dim=0)
            if use_gpu:
                label = label.cuda()
            loss = loss_func(logits, label)

            # update parameters
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # store predictions
            pred_ = torch.argmax(torch.softmax(logits, dim=-1), dim=1) # convert logits to probs
            preds.append(pred_.cpu().numpy())
            labels.append(label.data.cpu().numpy())
            losses.append(loss.item() * label.size(0))
            num_utt += label.size(0)
            step += 1
            if step > training_step:
                break

        ## end of epoch processing ##
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        # calculate train stats
        avg_loss = np.round(np.sum(losses) / num_utt, 4)
        avg_accuracy = round(accuracy_score(labels, preds) * 100, 5)
        avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 5)
        train_mf1 = round(f1_score(labels, preds, average='macro') * 100, 5)

        # run model on evaluation dataset
        # calculate validation/test stats
        dev_accuracy, dev_fscore, dev_mf, dev_reports = evaluate(model, devloader, use_gpu)
        test_accuracy, test_fscore, test_mf, test_reports = evaluate(model, testloader, use_gpu)

        # early stopping based on validation F1-score
        if dev_fscore > best_fscore:
            best_fscore = dev_fscore
            best_mf1 = dev_mf
            best_accuracy = dev_accuracy
            best_report = dev_reports

            best_test_fscore = test_fscore
            best_test_mf1 = test_mf
            best_test_accuracy = test_accuracy
            best_test_report = test_reports

            early_stopping_step = 0
        else:
            early_stopping_step += 1

        train_loss_list.append(avg_loss)
        train_f1_list.append(avg_fscore)
        dev_f1_list.append(dev_fscore)
        test_f1_list.append(test_fscore)

        # Logging
        log = 'Train: Epoch {} Loss {}, ACC {}, F1 {}, mF {}'.format(epoch + 1, avg_loss,
                                                                     avg_accuracy, avg_fscore, train_mf1)
        print(log)
        f.write(log + '\n')
        log = 'Validation: ACC {}, F1 {}, mF {}'.format(dev_accuracy, dev_fscore, dev_mf)
        print(log)
        f.write(log + '\n')
        log = 'Test: ACC {}, F1 {}, mF {}'.format(test_accuracy, test_fscore, test_mf)
        print(log)
        f.write(log + '\n')
        print('Epoch {} finished. Elapse {}'.format(epoch + 1, round(time.time() - start_time, 4)))
        if early_stopping_step == 10:
            break

    ## end of training ##
    print('----------------------------------------------')
    f.write('----------------------------------------------')
    log = '\n\n[DEV] best ACC {}, F1 {}, mF {}'.format(best_accuracy, best_fscore, best_mf1)
    f.write(log + '\n')
    print(log)
    f.write(best_report)
    log = '[TEST] best ACC {}, F1 {}, mF {}'.format(best_test_accuracy, best_test_fscore, best_test_mf1)
    f.write(log + '\n')
    print(log)
    f.write(best_test_report)
    f.write('----------------------------------------------\n')
    f.close()
    dump_data = [train_loss_list, train_f1_list, dev_f1_list, test_f1_list]
    pickle.dump(dump_data, open(metric_path, 'wb'))

def evaluate(model, dataloader, use_gpu):
    model.eval()
    preds = []
    labels = []
    for data in dataloader:
        textf, wrdm, label, uttm, spkm, edge_index, edge_attr, _, _ = data
        conv_len = [int(torch.sum(um).item()) for um in uttm]
        with torch.no_grad():
            logits = model(textf, wrdm, conv_len, edge_index, edge_attr, use_gpu)
        label = torch.cat(label, dim=0)
        pred_ = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        preds.append(pred_.cpu().numpy())
        labels.append(label.data.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 5)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 5) # weighted mean of F1 across (multi-)classes, uses number of true instances
    report_classes = classification_report(labels, preds, digits=4)
    mf1 = round(f1_score(labels, preds, average='macro') * 100, 5) # unweighted mean of F1 across (multi-)classes
    model.train()
    return avg_accuracy, avg_fscore, mf1, report_classes

def set_up_saving(args):
    model_index = str(args.index)
    model_path = 'results/model' + model_index + '.pkl'
    log_path = 'results/log' + model_index + '.txt'
    metric_path = 'results/metric' + model_index + '.pkl'
    f = open(log_path, 'a+', encoding='utf-8')
    f.write(str(args) + '\n\n')
    f.close()
    return(model_path, log_path, metric_path)

def set_seeds(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_name', type=str, default='MELD')
    parser.add_argument('-hip', type=int, default=1) # how many steps forward/backward to look for each utterance

    # fine tuning parameters
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-lr', type=float, default=8e-6)
    parser.add_argument('-n_epochs', type=int, default=50)
    parser.add_argument('-warmup_step', type=int, default=1000) # steps to increase learning rate to max
    parser.add_argument('-training_step', type=int, default=10000) # total number of training steps
    parser.add_argument('-use_gpu', type=bool, default=False)
    parser.add_argument('-schedule', type=str, default='linear')
    parser.add_argument('-seed', type=int, default=7)

    # utterance encoder parameters
    parser.add_argument('-pretrain', type=str, default='roberta-large')
    parser.add_argument('-sent_dim', type=int, default=200) # lower dim mapping; Roberta's 768d -> 200; this will be the size of the GNNs nodes

    # graph network parameters
    parser.add_argument('-comet_type', type=str, default='old')
    parser.add_argument('-gnn_nhead', type=int, default=4) # how many attention heads to use in the GNN
    parser.add_argument('-gnn_ff_dim', type=int, default=200) # dimensionality of additional feedforward network between layers of GNN
    parser.add_argument('-gnn_dropout', type=float, default=0.1)
    parser.add_argument('-edge_dim', type=int, default=200) # dimensionality of graph edges
    parser.add_argument('-gnn_num_layer', type=int, default=2) # how many updates to graph to do (i.e. layers)

    parser.add_argument('-index', type=int, default=1) # for saving different models
    args = parser.parse_args()
    print(args)
    return(args)

def main():

    args = parse_args()
    set_seeds(args)
    model_path, log_path, metric_path = set_up_saving(args)

    # whether to use their features or mine
    if args.comet_type=='old':
        edge_starting_dim=768
    elif args.comet_type=='new':
        edge_starting_dim=1024

    # instantiate the combined model; consists of a utterance encoder (Roberta),
    # a graph neural network, and classifier (for emotions)
    model = CombinedModel(args.pretrain, args.sent_dim,
                  args.gnn_ff_dim, args.gnn_nhead,
                  args.edge_dim, args.gnn_num_layer,
                  edge_starting_dim=edge_starting_dim)

    if args.use_gpu:
        model = model.cuda()

    # data loaders, which process the data (load features, embed utterances etc.) and then deal with batching
    train_loader, dev_loader, test_loader = get_loaders(args.dataset_name, args.batch_size, args.pretrain, comet_type=args.comet_type)
    loss_func = nn.CrossEntropyLoss(reduction='mean')

    # set up optimization
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_step, args.training_step)

    # main training loop
    train(model, loss_func, train_loader, dev_loader, test_loader, args.n_epochs, optimizer, scheduler,
          args.training_step, model_path, log_path, metric_path, args.use_gpu)

if __name__ == '__main__':
    '''Calls:
        python train_model.py -use_gpu True
        python train_model.py -use_gpu True -index 2 -comet_type new
        python train_model.py -use_gpu True -index 3 -comet_type new

    '''
    main()
