import numpy as np
import torch

def compute_local_test_accuracy(model, dataloader, data_distribution, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    generalized_total, generalized_correct = 0, 0

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x = x.to(device)
            target = target.to(device)

            out = model(x)
            _, pred_label = torch.max(out.data, 1)
            correct_filter = (pred_label == target.data)

            generalized_total += x.data.size()[0]
            generalized_correct += correct_filter.sum().item()

            target_cpu = target.data.cpu().numpy()
            correct_filter_cpu = correct_filter.cpu().numpy()

            for i, true_label in enumerate(target_cpu):
                total_label_num[true_label] += 1
                if correct_filter_cpu[i]:
                    correct_label_num[true_label] += 1

    personalized_correct = (correct_label_num * data_distribution).sum()
    personalized_total = (total_label_num * data_distribution).sum()
    return personalized_correct / personalized_total, generalized_correct / generalized_total


def compute_acc(net, test_data_loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x = x.to(device)
            target = target.to(device)

            out = net(x)
            _, pred_label = torch.max(out.data, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

    return correct / float(total)


def evaluate_global_model(args, nets_this_round, global_model, val_local_dls, test_dl, data_distributions,
                          best_val_acc_list, best_test_acc_list, benign_client_list, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = global_model.to(device)

    for net_id, _ in nets_this_round.items():
        if net_id in benign_client_list:
            val_local_dl = val_local_dls[net_id]
            data_distribution = data_distributions[net_id]

            val_acc = compute_acc(global_model, val_local_dl, device=device)
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(
                global_model, test_dl, data_distribution, device=device
            )

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} | Personalized Test Acc: {:.5f} | Generalized Test Acc: {:.5f}'.format(
                net_id, personalized_test_acc, generalized_test_acc
            ))

    return np.array(best_test_acc_list)[np.array(benign_client_list)].mean()