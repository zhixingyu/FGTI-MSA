import logging
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from ...utils import MetricsTop, dict_to_str, contralLoss

logger = logging.getLogger('MMSA')
class MISA():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.MSELoss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.loss_cmd = CMD()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.discrib = args.dataset_name + args.discription
        # tensorboard
        self.train_step = 0
        self.val_step = 0
        self.test_step = 0

    def do_train(self, model, dataloader, return_epoch_results=False):
        self.model = model
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=self.args.learning_rate)
        epochs, best_epoch = 0, 0

        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        while True:
            epochs += 1
            # train
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    # using accumulated gradients
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    text = batch_data['text'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    vision = batch_data['vision'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    # forward
                    outputs = model(text, audio, vision)['M']

                    # compute loss
                    cls_loss = self.criterion(outputs, labels)
                    contrastive_loss = self.get_contrastive_loss(labels)
                    cls3 = self.get_3cls_loss(labels)
                    cmd_loss = self.get_cmd_loss()

                    loss = cls_loss + \
                           self.args.cls3_weight * (cls3 + cmd_loss)+\
                           self.args.contralLoss_weight * contrastive_loss
                    # backward
                    loss.backward()
                    self.train_step += 1

                    if self.args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad],
                                                       max_norm=20, norm_type=2)
                    # store results
                    train_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()

                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            # lr_scheduler.step()
            train_loss = train_loss / len(dataloader['train'])

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >> loss: "
                f"{round(train_loss, 4)} {dict_to_str(train_results)} >>step_lr={optimizer.param_groups[0]['lr']}"
            )

            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL", return_sample_results=False)
            # test_results = self.do_test(model, dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    outputs = model(text, audio, vision)

                    cls_loss = self.criterion(outputs['M'], labels)
                    contrastive_loss = self.get_contrastive_loss(labels)
                    cls3 = self.get_3cls_loss(labels)
                    cmd_loss = self.get_cmd_loss()
                    loss = cls_loss + \
                           self.args.cls3_weight * (cls3 + cmd_loss) + \
                           self.args.contralLoss_weight * contrastive_loss
                    eval_loss += loss.item()

                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        return eval_results

    def get_contrastive_loss(self, label):
        # losses between shared states
        loss = contralLoss(self.model.Model.utt_shared_t, label)
        loss += contralLoss(self.model.Model.utt_shared_v, label)
        loss += contralLoss(self.model.Model.utt_shared_a, label)
        return loss / 3.0

    def get_cmd_loss(self, ):

        if not self.args.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = self.loss_cmd(self.model.Model.utt_shared_t, self.model.Model.utt_shared_v, 5)
        loss += self.loss_cmd(self.model.Model.utt_shared_t, self.model.Model.utt_shared_a, 5)
        loss += self.loss_cmd(self.model.Model.utt_shared_a, self.model.Model.utt_shared_v, 5)
        loss = loss / 3.0

        return loss

    def get_3cls_loss(self, labels):

         # .cpu().data.numpy()
        x1 = self.model.Model.conA  # .cpu().data.numpy()
        x2 = self.model.Model.conV  # .cpu().data.numpy()
        x3 = self.model.Model.conT  # .cpu().data.numpy()
        criterion = nn.MSELoss()

        p1 = criterion(x1, labels)
        p2 = criterion(x2, labels)
        p3 = criterion(x3, labels)
        res = 0.33 * (p1 + p2 + p3)
        return res * 0.5

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """
    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

