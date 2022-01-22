import os
from typing import Text, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn.modules import activation
from qlib.contrib.model.pytorch_utils import count_parameters
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.utils import get_or_create_path, save_multiple_parts_file, unpack_archive_with_buffer
from sklearn.metrics import roc_auc_score, mean_squared_error
from qlib.workflow import R

# reference to qlib/qlib/contrib/pytorch_nn.py
class CNNModelPytorch(Model):
    def __init__(
        self,
        input_dim=360,
        output_dim=1,
        layer_num=1,
        kernel_size=3,
        lr=0.001,
        max_steps=300,
        batch_size=2000,
        early_stop_rounds=50,
        eval_steps=20,
        lr_decay=0.96,
        lr_decay_steps=100,
        optimizer="gd",
        loss="mse",
        GPU=0,
        seed=None,
        weight_decay=0.0,
        **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("CNNModelPytorch")
        self.logger.info("CNN pytorch version...")

        # set hyper-parameters.
        self.layer_num = layer_num
        self.kernel_size = kernel_size
        self.lr = lr
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.early_stop_rounds = early_stop_rounds
        self.eval_steps = eval_steps
        self.lr_decay = lr_decay
        self.lr_decay_steps = lr_decay_steps
        self.optimizer = optimizer.lower()
        self.loss_type = loss
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.weight_decay = weight_decay

        self.logger.info(
            "CNN parameters setting:"
            "\nlayer_num : {}"
            "\nkernel_size: {}"
            "\nlr : {}"
            "\nmax_steps : {}"
            "\nbatch_size : {}"
            "\nearly_stop_rounds : {}"
            "\neval_steps : {}"
            "\nlr_decay : {}"
            "\nlr_decay_steps : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\neval_steps : {}"
            "\nseed : {}"
            "\ndevice : {}"
            "\nuse_GPU : {}"
            "\nweight_decay : {}".format(
                layer_num,
                kernel_size,
                lr,
                max_steps,
                batch_size,
                early_stop_rounds,
                eval_steps,
                lr_decay,
                lr_decay_steps,
                optimizer,
                loss,
                eval_steps,
                seed,
                self.device,
                self.use_gpu,
                weight_decay,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        if loss not in {"mse", "binary"}:
            raise NotImplementedError("loss {} is not supported!".format(loss))
        self._scorer = mean_squared_error if loss == "mse" else roc_auc_score

        self.cnn_model = Net(input_dim, output_dim, kernel_size, layer_num)
        self.logger.info("model:\n{:}".format(self.cnn_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.cnn_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.cnn_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.cnn_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        # Reduce learning rate when loss has stopped decrease
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.train_optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            verbose=True,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0.00001,
            eps=1e-08,
        )

        self.fitted = False
        self.cnn_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        verbose=True,
        save_path=None,
    ):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")
        x_train, y_train = df_train['feature'], df_train['label']
        x_valid, y_valid = df_valid['feature'], df_valid['label']
        try:
            wdf_train, wdf_valid = dataset.prepare(["train", "valid"], col_set=["weight"], data_key=DataHandlerLP.DK_L)
            w_train, w_valid = wdf_train["weight"], wdf_valid["weight"]
        except KeyError as e:
            w_train = pd.DataFrame(np.ones_like(y_train.values), index=y_train.index)
            w_valid = pd.DataFrame(np.ones_like(y_valid.values), index=y_valid.index)

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_loss = np.inf
        evals_result["train"] = []
        evals_result["valid"] = []
        # train
        self.logger.info("training...")
        self.fitted = True
        # return
        # prepare training data
        x_train_values = torch.from_numpy(x_train.values).float()
        y_train_values = torch.from_numpy(y_train.values).float()
        w_train_values = torch.from_numpy(w_train.values).float()
        train_num = y_train_values.shape[0]
        # prepare validation data
        x_val_auto = torch.from_numpy(x_valid.values).float().to(self.device)
        y_val_auto = torch.from_numpy(y_valid.values).float().to(self.device)
        w_val_auto = torch.from_numpy(w_valid.values).float().to(self.device)

        for step in range(self.max_steps):
            if stop_steps >= self.early_stop_rounds:
                if verbose:
                    self.logger.info("\tearly stop")
                break
            loss = AverageMeter()
            self.cnn_model.train()
            self.train_optimizer.zero_grad()
            choice = np.random.choice(train_num, self.batch_size)
            x_batch_auto = x_train_values[choice].to(self.device)
            y_batch_auto = y_train_values[choice].to(self.device)
            w_batch_auto = w_train_values[choice].to(self.device)

            # forward
            preds = self.cnn_model(x_batch_auto)
            cur_loss = self.get_loss(preds, w_batch_auto, y_batch_auto, self.loss_type)
            cur_loss.backward()
            self.train_optimizer.step()
            loss.update(cur_loss.item())
            R.log_metrics(train_loss=loss.avg, step=step)

            # validation
            train_loss += loss.val
            # for evert `eval_steps` steps or at the last steps, we will evaluate the model.
            if step % self.eval_steps == 0 or step + 1 == self.max_steps:
                stop_steps += 1
                train_loss /= self.eval_steps

                with torch.no_grad():
                    self.cnn_model.eval()
                    loss_val = AverageMeter()

                    # forward
                    preds = self.cnn_model(x_val_auto)
                    cur_loss_val = self.get_loss(preds, w_val_auto, y_val_auto, self.loss_type)
                    loss_val.update(cur_loss_val.item())
                R.log_metrics(val_loss=loss_val.val, step=step)
                if verbose:
                    self.logger.info(
                        "[Epoch {}]: train_loss {:.6f}, valid_loss {:.6f}".format(step, train_loss, loss_val.val)
                    )
                evals_result["train"].append(train_loss)
                evals_result["valid"].append(loss_val.val)
                if loss_val.val < best_loss:
                    if verbose:
                        self.logger.info(
                            "\tvalid loss update from {:.6f} to {:.6f}, save checkpoint.".format(
                                best_loss, loss_val.val
                            )
                        )
                    best_loss = loss_val.val
                    stop_steps = 0
                    torch.save(self.cnn_model.state_dict(), save_path)
                train_loss = 0
                # update learning rate
                self.scheduler.step(cur_loss_val)

        # restore the optimal parameters after training
        self.cnn_model.load_state_dict(torch.load(save_path))
        if self.use_gpu:
            torch.cuda.empty_cache()

    def get_loss(self, pred, w, target, loss_type):
        if loss_type == "mse":
            delta = pred - target
            sqr_loss = delta ** 2
            loss = torch.mul(sqr_loss, w).mean()
            return loss
        elif loss_type == "binary":
            loss = nn.BCELoss(weight=w)
            return loss(pred, target)
        else:
            raise NotImplementedError("loss {} is not supported!".format(loss_type))

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x_test_pd = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        x_test = torch.from_numpy(x_test_pd.values).float().to(self.device)
        self.cnn_model.eval()
        with torch.no_grad():
            preds = self.cnn_model(x_test).detach().cpu().numpy()
        return pd.Series(np.squeeze(preds), index=x_test_pd.index)

    def save(self, filename, **kwargs):
        with save_multiple_parts_file(filename) as model_dir:
            model_path = os.path.join(model_dir, os.path.split(model_dir)[-1])
            # Save model
            torch.save(self.cnn_model.state_dict(), model_path)

    def load(self, buffer, **kwargs):
        with unpack_archive_with_buffer(buffer) as model_dir:
            # Get model name
            _model_name = os.path.splitext(list(filter(lambda x: x.startswith("model.bin"), os.listdir(model_dir)))[0])[
                0
            ]
            _model_path = os.path.join(model_dir, _model_name)
            # Load model
            self.cnn_model.load_state_dict(torch.load(_model_path))
        self.fitted = True


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, layer_num):
        super(Net, self).__init__()
        cnn_layers = []
        for i in range(layer_num):
            conv = nn.Conv1d(input_dim, output_dim, kernel_size, 1, int((kernel_size-1)/2))
            cnn_layers.append(conv)
            relu = nn.ReLU()
            cnn_layers.append(relu)
        drop_input = nn.Dropout(0.05)
        cnn_layers.append(drop_input)
        softmax = nn.Softmax(output_dim)
        cnn_layers.append(softmax)
        self.cnn_layers = nn.ModuleList(cnn_layers)

    def forward(self, x):
        # [2000,20] -> batch_size, in_channels, length = [1,20,2000]
        cur_output = x.transpose(0,1).unsqueeze(0)
        for i, now_layer in enumerate(self.cnn_layers):
            cur_output = now_layer(cur_output)
        cur_output = cur_output.squeeze().unsqueeze(1)
        return cur_output