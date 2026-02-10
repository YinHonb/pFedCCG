import torch
import numpy as np
import torch.optim as optim
import copy
from evaluate.evaluate_classification import compute_local_test_accuracy
from Data import cifar_dataset, yahoo_dataset
from utils import optimize_W_l2
from model.cifar_model import cifar_cnn
from model.text_model import textcnn

class pFedCCG_classification(object):
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">> Using device: {self.device}")
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed) if hasattr(args, 'seed') else torch.cuda.manual_seed(42)
        self.node_num = args.n_parties
        if args.dataset == "cifar10":
            model = cifar_cnn
            dataset = cifar_dataset
        elif args.dataset == "yahoo":
            model = textcnn
            dataset = yahoo_dataset

        self.init_model = model(cfg['classes_size']).to(self.device)
        self.init_parameters = self.init_model.state_dict()

        self.model = []
        for _ in range(self.node_num):
            client_model = model(cfg['classes_size']).to(self.device)
            self.model.append(client_model)
        for m in self.model:
            m.load_state_dict(self.init_parameters)

        self.delta_weights = []
        for i in range(self.node_num):
            self.delta_weights.append(
                {key: torch.zeros_like(value).to(self.device) for key, value in self.model[i].named_parameters()})

        self.W = None

        self.train_dataloaders, self.val_dataloaders, self.test_loader, self.net_dataidx_map, self.traindata_cls_counts, self.data_distributions, self.js_matrix = dataset.dataset_read(
            args.dataset, args.datadir, args.batch_size, args.n_parties, args.partition, args.beta, args.skew_class)

        total_data_points = sum([len(self.net_dataidx_map[k]) for k in range(self.node_num)])
        self.p_vector = np.array([len(self.net_dataidx_map[k]) / total_data_points for k in range(self.node_num)])
        self.best_val_acc_list = [0 for _ in range(self.node_num)]
        self.best_test_acc_list = [0 for _ in range(self.node_num)]
        self.best_test_acc_list_generalized = [0 for _ in range(self.node_num)]


    def step(self, now_time):
        self.model_evaluate(now_time)
        self.local_train()
        self.model_evaluate(now_time)
        self.W = optimize_W_l2(self.p_vector, self.data_distributions, 0.2 + 0.2 / (1.0 + np.exp((25.0 - float(now_time)) / 7.0)))
        self.aggregation_model()

    def local_train(self):
        for node in range(self.node_num):
            net = self.model[node]
            subset_data = self.train_dataloaders[node]
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.args.lr, momentum=0.9, weight_decay=1e-5)
            criterion = torch.nn.CrossEntropyLoss()
            net.train()
            for epoch in range(self.args.epochs):
                for x, y in subset_data:
                    x = x.to(self.device)
                    y = y.to(self.device).long()
                    optimizer.zero_grad()
                    y = y.long()
                    outputs = net(x)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

    def aggregation_model(self):
        model = {k: self.model[k] for k in range(self.node_num)}
        tmp_client_state_dict = {}
        for client_id in model.keys():
            tmp_client_state_dict[client_id] = copy.deepcopy(self.init_model.state_dict())
            for key in tmp_client_state_dict[client_id]:
                tmp_client_state_dict[client_id][key] = torch.zeros_like(tmp_client_state_dict[client_id][key])

        for client_id in model.keys():
            tmp_client_state = tmp_client_state_dict[client_id]
            aggregation_weight_vector = copy.deepcopy(self.W[client_id])

            for neighbor_id in model.keys():
                net_para = model[neighbor_id].state_dict()
                for key in tmp_client_state:
                    tmp_client_state[key] += net_para[key] * aggregation_weight_vector[neighbor_id]

        for client_id in model.keys():
            model[client_id].load_state_dict(tmp_client_state_dict[client_id])


    def model_evaluate(self, step_num):
        for node in range(self.node_num):
            net = self.model[node]
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net,
                                                                                      self.test_loader,
                                                                                      self.data_distributions[node])
            if personalized_test_acc > self.best_test_acc_list[node]:
                self.best_test_acc_list[node] = personalized_test_acc
            print('>> Client {} test2 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(
                node, personalized_test_acc, generalized_test_acc))

            if generalized_test_acc > self.best_test_acc_list_generalized[node]:
                self.best_test_acc_list_generalized[node] = generalized_test_acc

        print('>> (Current) Round {} | Local Per: {:.5f} ({:.5f}), generalized Per: {:.5f}'.format(step_num, np.mean(self.best_test_acc_list), np.std(self.best_test_acc_list), np.mean(self.best_test_acc_list_generalized)))
        return np.mean(self.best_test_acc_list)

