import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models import CnnModel, CrnnModel, CrnnLongModel, RnnModel

import multiprocessing.dummy as multiprocessing


class Data:
    def __init__(self, audio_paths, sec=30, fps=25):
        self.audio_paths = audio_paths
        self.sec = sec
        self.fps = fps
        self.classes = {'blues': 0, 'classical': 1, 'country': 2,
                        'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6,
                        'pop': 7, 'reggae': 8, 'rock': 9}

    def __getitem__(self, index):
        elem = self.audio_paths[index]
        genre = elem.split('.')[0][4:]

        df = pd.read_csv("genres/genres/" + genre + "/" + self.audio_paths[index], sep=',', header=None)
        features_x = df.values.transpose(1, 0)[:, 1:]
        features_x, s = frame_feature_extractor(features_x)
        y = self.classes[genre]
        y = [y for _ in range(s)]
        return features_x, y

    def __len__(self):
        return len(self.audio_paths)


class MusicalClassifier:
    def __init__(self, epoch, batch_size, train, val, classifier_model,
                 p_name="", sec=30, fps=25, dropp_lr_epoch=[None], device='cpu'):
        """
        MusicalClassifier
        :param epoch: number of epoch
        :param train: train data
        :param train: validation data
        :param dropp_lr_epoch: numbers of epoch when lr is dropped by 10
        """
        self.data_train = Data(train, fps=fps)
        self.trainloader = DataLoader(self.data_train, batch_size=batch_size, shuffle=True)

        self.data_val = Data(val, sec=sec, fps=fps)
        self.valloader = DataLoader(self.data_val, batch_size=1, shuffle=True)
        self.device = device
        self.p_name = p_name
        self.epoch = epoch
        self.classifier = classifier_model
        self.classifier = self.classifier.to(self.device)
        self.classifier.apply(self.init_weights)
        self.optimizer = optim.Adam(self.classifier.parameters(), 1e-3)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()
        self.dropp_lr_epoch = dropp_lr_epoch

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv1d:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def train(self):
        best_acc = 0
        best_loss = 1000
        list_loss_v = []
        list_acc_v = []

        list_loss_t = []
        list_acc_t = []
        for ep in range(self.epoch):
            self.classifier.train()

            if ep in self.dropp_lr_epoch:
                self.scheduler.step()
                print("lr: ", [p['lr'] for p in self.optimizer.param_groups])

            total_loss = 0.0
            total_correct = 0.0
            step = 0
            for x, y in self.trainloader:
                step += 1
                x = torch.cat(x, 0)
                y = torch.cat(y, 0)
                x = x.to(self.device).float()
                y = y.to(self.device)

                model_prob_labels = self.classifier(x)
                _, predict_labels = torch.max(F.softmax(model_prob_labels, -1), 1)

                loss = self.criterion(model_prob_labels, y)
                print(loss)
                total_loss += loss.detach().item()
                total_correct += torch.sum(predict_labels == y.data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_acc = total_correct / len(self.data_train)
            avg_loss = total_loss / step
            list_loss_t.append(avg_loss)
            list_acc_t.append(avg_acc)
            print("device:{}, Epoch: {}, epoch_loss: {:.5}, accuracy: {:.5}".format(self.p_name, (ep + 1), avg_loss,
                                                                                    avg_acc))
            self.save(f'{p}/{self.p_name}/{ep}_classifier.pkl')

            # validation
            loss_val, acc_val = self.test(self.data_val, self.valloader)
            if len(list_acc_v) > 0:
                if acc_val >= best_acc and loss_val < best_loss:
                    print("SAVED")
                    self.save(f'{p}/{self.p_name}/classifier.pkl')
                    best_acc = acc_val
                    best_loss = loss_val

            list_loss_v.append(loss_val)
            list_acc_v.append(acc_val)
            print("VALIDATION! device: {}, Epoch: {}, loss: {:.5}, accuracy: {:.5}".format(self.p_name, (ep + 1),
                                                                                           loss_val, acc_val))

        self.save(f'{p}/{self.p_name}/end_classifier.pkl')
        return list_loss_t, list_acc_t, list_loss_v, list_acc_v

    def save(self, path):
        torch.save(self.classifier.state_dict(), path)

    def test(self, data, loader):
        self.classifier.eval()
        acc = 0.0
        loss = 0.0
        for x, y in loader:
            x = torch.cat(x, 0)
            y = torch.cat(y, 0)
            x = x.to(self.device).float()
            y = y.to(self.device)

            model_prob_labels = self.classifier(x)
            l = self.criterion(model_prob_labels, y)
            loss += l.detach().item()

            _, predict_labels = torch.max(F.softmax(model_prob_labels, -1), 1)
            acc += torch.sum(predict_labels == y.data)

        return loss / len(data), acc / len(data)


def frame_feature_extractor(S):
    if not S.shape[0] % 128 == 0:
        S = S[:-1 * (S.shape[0] % 128)]  # divide the mel spectrogram
    chunk_num = int(S.shape[0] / 128)
    mel_chunks = np.split(S, chunk_num)  # create 128 * 128 data frames
    return mel_chunks, chunk_num

def train_net(arg_list):
    data_train, data_val, device_name, p_name, batch_size, classifier_model, dropp_lr_epoch = arg_list
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    classifier = classifier_model()
    mus_class = MusicalClassifier(500, batch_size, data_train, data_val, classifier, p_name, sec=30, fps=25,
                                  dropp_lr_epoch=dropp_lr_epoch)
    print(mus_class.classifier)
    list_loss_t, list_acc_t, list_loss_v, list_acc_v = mus_class.train()
    val = pd.DataFrame({"loss": list_loss_v, "acc": list_acc_v})
    val.to_csv(f'{p}/validation_{p_name}.csv', sep=',', index=False)

    tr = pd.DataFrame({"loss": list_loss_t, "acc": list_acc_t})
    tr.to_csv(f'{p}/train_{p_name}.csv', sep=',', index=False)

    test_loss, test_acc = test_net(f"{p}/{p_name}/classifier.pkl", data_test, 25, device, classifier_model, net_args)
    print(f"{device_name}, classifier.pkl: ", test_loss, test_acc)
    test_loss, test_acc = test_net(f"{p}/{p_name}/end_classifier.pkl", data_test, 25, device, classifier_model,
                                   net_args)
    print(f"{device_name}, end_classifier.pkl: ", test_loss, test_acc)


def test_net(path, test_data, fps, device, classifier_model, net_args):
    data = Data(test_data, fps=fps)
    loader = DataLoader(data, batch_size=1, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    classifier = classifier_model()
    classifier = classifier.to(device)
    classifier.load_state_dict(torch.load(path))
    classifier.eval()

    acc = 0.0
    loss = 0.0
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device)

        model_prob_labels = classifier(x)
        l = criterion(model_prob_labels, y)
        loss += l.detach().item()

        _, predict_labels = torch.max(F.softmax(model_prob_labels, -1), 1)
        acc += torch.sum(predict_labels == y.data)

    return loss / len(data), acc / len(data)


if __name__ == "__main__":
    data = []
    p = "exp1"
    classes = ('blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock')
    for c in classes:
        names = os.listdir("genres/genres/" + c)
        data.extend(names)
    data = [d for d in data if d.split('.')[-1] == 'csv' and d[:3] == "mel"]

    data_train, data_test = train_test_split(data, test_size=0.2)
    data_val, data_test = train_test_split(data_test, test_size=0.5)

    arg_list = [(data_train, data_val, 'cuda:0', '0', 128, CnnModel, [None]),
                (data_train, data_val, 'cuda:1', '1', 128, CrnnModel, [None]),
                (data_train, data_val, 'cuda:0', '2', 128, CrnnLongModel, [None])]

    # mp = multiprocessing.Pool(processes=22)
    # mp.map(train_net, arg_list)
    #
    # mp.close()
    # mp.join()
    for i in range(1):
        train_net(arg_list[2])
