import numpy as np
import torch
import copy

from models import SplitLSTMAutoEncoder, DCCLSTMAutoEncoder, MLP
from torch import nn, optim
from utils import make_seq_batch
from sklearn.metrics import f1_score

EVAL_WIN = 2000

class Server:
    def __init__(self, train_A, train_B, config):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.train_A = train_A
        self.train_B = train_B
        self.input_size_A = train_A["A"].shape[1]
        self.input_size_B = train_B["B"].shape[1]
        self.rep_size = int(config["FL"]["rep_size"])
        self.n_classes = len(set(train_A["y"]))
        self.label_modality = config["SERVER"]["label_modality"]
        self.test_modality = config["SERVER"]["test_modality"]
        self.frac = float(config["SERVER"]["frac"])
        self.n_epochs = int(config["SERVER"]["num_epochs"])
        self.lr = float(config["SERVER"]["lr"])
        self.criterion = config["SERVER"]["criterion"]
        self.optimizer = config["SERVER"]["optimizer"]
        self.model_ae = config["SIMULATION"]["model_ae"]
        self.model_sv = config["SIMULATION"]["model_sv"]
        self.mlp_dropout = 0.5 if config["SIMULATION"]["data"] == "mhealth" else 0.0

        if config["SIMULATION"]["data"] == "ur_fall":
            self.batch_min = 16
            self.batch_max = 32
        else:
            self.batch_min = 128
            self.batch_max = 256

    def init_models(self):
        if self.model_ae == "split_LSTM":
            self.global_ae = SplitLSTMAutoEncoder(
                self.input_size_A, self.input_size_B, self.rep_size).double().to(self.device)

        if self.model_ae == "DCCAE_LSTM":
            self.global_ae = DCCLSTMAutoEncoder(
                self.input_size_A, self.input_size_B, self.rep_size).double().to(self.device)

        if self.model_sv == "MLP":
            self.global_sv = MLP(
                self.rep_size, self.n_classes, self.mlp_dropout).double().to(self.device)

    def select_clients(self, clients):
        """Selects clients to communicate with.

        Args:
            clients: a list of Client objects

        Returns:
            A list of selected Client objects
        """
        n_selected_clients = int(len(clients) * self.frac)
        selected_clients = np.random.choice(
            clients, n_selected_clients, replace=False)
        return selected_clients

    def average_models(self, local_models):
        """Averages local models into a new global model.

        Args:
            local_models: a list of tuples containing models, client modalities, and client weights

        Returns:
            A new global model.
        """
        w_avg_A = None
        w_avg_B = None
        n_A = 0
        n_B = 0
        for model in local_models:
            if model[1] == "A" or model[1] == "AB":
                n_A += model[2]
                if not w_avg_A:
                    w_avg_A = copy.deepcopy(model[0].state_dict())
                    for key in w_avg_A.keys():
                        if "A" in key:
                            w_avg_A[key] = w_avg_A[key] * model[2]
                else:
                    for key in w_avg_A.keys():
                        if "A" in key:
                            # multiply client weight
                            w_avg_A[key] += model[0].state_dict()[key] * \
                                model[2]

            if model[1] == "B" or model[1] == "AB":
                n_B += model[2]
                if not w_avg_B:
                    w_avg_B = copy.deepcopy(model[0].state_dict())
                    for key in w_avg_B.keys():
                        if "B" in key:
                            w_avg_B[key] = w_avg_B[key] * model[2]
                else:
                    for key in w_avg_B.keys():
                        if "B" in key:
                            w_avg_B[key] += model[0].state_dict()[key] * \
                                model[2]
        w_avg = w_avg_A if w_avg_A else w_avg_B

        if w_avg_A:
            for key in w_avg.keys():
                if "A" in key:
                    w_avg[key] = w_avg_A[key] / n_A
        if w_avg_B:
            for key in w_avg.keys():
                if "B" in key:
                    w_avg[key] = w_avg_B[key] / n_B

        return w_avg

    def train_classifier(self, label_modality, optimizer, criterion, x_train, y_train, idx_start, idx_end):
        """Trains the global classifier with labelled data on the server"""

        x = x_train[:, idx_start:idx_end, :]
        seq = torch.from_numpy(x).double().to(self.device)
        y = y_train[:, idx_start:idx_end]

        with torch.no_grad():
            rpts = self.global_ae.encode(seq, label_modality)
        targets = torch.from_numpy(y.flatten()).to(self.device)

        optimizer.zero_grad()
        output = self.global_sv(rpts)
        loss = criterion(output, targets.long())
        top_p, top_class = output.topk(1, dim=1)
        equals = top_class == targets.view(*top_class.shape).long()
        accuracy = torch.mean(equals.type(torch.FloatTensor))

        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss.item(), accuracy

    def update(self, local_models):
        """Updates the global model using received local models.

        Args:
            local_models: a list of local models

        Returns:
            A tuple containing loss and accuracy values
        """
        # Average all local models and update the global ae
        global_weights = self.average_models(local_models)
        self.global_ae.load_state_dict(global_weights)
        self.global_ae.eval()
        self.global_sv.train()

        if self.criterion == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss().to(self.device)
        if self.optimizer == "Adam":
            optimizer = optim.Adam(self.global_sv.parameters(), lr=self.lr)

        round_loss = []
        round_accuracy = []
        for epoch in range(self.n_epochs):
            epoch_loss = []
            epoch_accuracy = []
            batch_size = np.random.randint(
                low=self.batch_min, high=self.batch_max)
            x_A_train, _, y_A_train = make_seq_batch(
                self.train_A, [0], len(self.train_A["A"]), batch_size)
            _, x_B_train, y_B_train = make_seq_batch(
                self.train_B, [0], len(self.train_B["B"]), batch_size)
            # A_train and B_train both are in the shape of (batch_size, seq_len, input_size), i.e., batch first

            if "A" in self.label_modality:
                seq_len = x_A_train.shape[1]
                idx_start = 0
                idx_end = 0
                while idx_end < seq_len:
                    win_len = np.random.randint(low=16, high=32)
                    idx_start = idx_end
                    idx_end += win_len
                    idx_end = min(idx_end, seq_len)
                    loss, accuracy = self.train_classifier(
                        "A", optimizer, criterion, x_A_train, y_A_train, idx_start, idx_end)
                    epoch_loss.append(loss)
                    epoch_accuracy.append(accuracy)

            if "B" in self.label_modality:
                seq_len = x_B_train.shape[1]
                idx_start = 0
                idx_end = 0
                while idx_end < seq_len:
                    win_len = np.random.randint(low=16, high=32)
                    idx_start = idx_end
                    idx_end += win_len
                    idx_end = min(idx_end, seq_len)
                    loss, accuracy = self.train_classifier(
                        "B", optimizer, criterion, x_B_train, y_B_train, idx_start, idx_end)
                    epoch_loss.append(loss)
                    epoch_accuracy.append(accuracy)

            round_loss.append(np.mean(epoch_loss))
            round_accuracy.append(np.mean(epoch_accuracy))

        return np.mean(round_loss), np.mean(round_accuracy)

    def eval(self, data_test):
        """Evaluates global models against testing data on the server.

        Args:
            data_test: a dictionary containing testing data of modalities A&B and labels y.

        Returns:
            A tuple containing loss and accuracy values
        """
        self.global_ae.eval()
        self.global_sv.eval()
        if self.criterion == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss().to(self.device)

        if self.test_modality == "A":
            x_samples = np.expand_dims(data_test["A"], axis=0)
        elif self.test_modality == "B":
            x_samples = np.expand_dims(data_test["B"], axis=0)
        y_samples = np.expand_dims(data_test["y"], axis=0)

        win_loss = []
        win_accuracy = []
        win_f1 = []
        n_samples = x_samples.shape[1]
        n_eval_process = n_samples // EVAL_WIN + 1

        # np_gt is a numpy.ndarray of labels (classes)
        # equals is a tensor of booleans representing if the model correctly guessed 
        class_occurances_total = [0] * 25
        class_occurances_correct = [0] * 25

        for i in range(n_eval_process):
            idx_start = i * EVAL_WIN
            idx_end = np.min((n_samples, idx_start+EVAL_WIN))
            x = x_samples[:, idx_start:idx_end, :]
            y = y_samples[:, idx_start:idx_end]
        
            inputs = torch.from_numpy(x).double().to(self.device)
            targets = torch.from_numpy(y.flatten()).to(self.device)
            rpts = self.global_ae.encode(inputs, self.test_modality)
            output = self.global_sv(rpts)

            loss = criterion(output, targets.long())
            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets.view(*top_class.shape).long()
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            np_gt = y.flatten()
            np_pred = top_class.squeeze().cpu().detach().numpy()
            weighted_f1 = f1_score(np_gt, np_pred, average="weighted")

            win_loss.append(loss.item())
            win_accuracy.append(accuracy)
            win_f1.append(weighted_f1)

            # count the number of times each class was predicted correctly
            # loop through each prediction
            for i in range(len(np_gt)): 
                class_index = np_gt[i]
                #print("print(class_index)", class_index)
                # increment total occurances for that class
                class_occurances_total[int(class_index)] += 1
                # if the prediction was correct, increment correct occurances for that class
                if equals[i] == True:
                    class_occurances_correct[int(class_index)] += 1

            # print("class_occurances_correct len", len(class_occurances_correct))
            # print(class_occurances_correct)

            # print("\nclass_occurances_total len", len(class_occurances_total))
            # print(class_occurances_total)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
       
        return np.mean(win_loss), np.mean(win_accuracy), np.mean(weighted_f1), class_occurances_correct, class_occurances_total
