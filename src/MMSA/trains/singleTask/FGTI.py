import logging
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from ...utils import MetricsTop, dict_to_str,contralLoss
from scipy.stats import gamma
import pickle
logger = logging.getLogger('MMSA')

class MISA():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.MSELoss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()
        self.loss_hsic = HSIC()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.discrib='my_modelV4_' + args.dataset_name + args.discription
         # tensorboard
        self.train_step = 0
        self.val_step = 0
        self.test_step = 0

    def do_train(self, model, dataloader, return_epoch_results=False):
        self.model = model
        # print('###########################11111111111111111111111111111111111')
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.learning_rate)#, weight_decay=self.args.weight_decay)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
            features_results = {
                'shared_a': [],
                'shared_v': [],
                'shared_t': [],
                'shared1_a': [],
                'shared1_v': [],
                'shared1_t': [],
                'private_a': [],
                'private_v': [],
                'private_t': [],
                'p_a': [],
                'p_v': [],
                'p_t': [],
            }

        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        while True: 
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
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
                    hsic_loss = self.get_hsic_loss()
                    contrastive_loss = self.get_contrastive_loss(labels)
                    cls3 = self.get_3cls_loss(labels)
                    if self.args.use_cmd_sim:
                        similarity_loss = cmd_loss
                    else:
                        similarity_loss = domain_loss
                    
                    loss = cls_loss + \
                           self.args.hsic_weight * hsic_loss + \
                           self.args.cls3_weight * cls3+\
                            self.args.contralLoss_weight * contrastive_loss
                    # backward
                    loss.backward()
                    self.train_step += 1

                    if self.args.grad_clip != -1.0:
                        # torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], self.args.grad_clip)
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
            # epoch results
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
                features_results['shared_a'].append(self.model.Model.utt_shared_a.cpu().detach().numpy())
                features_results['shared_v'].append(self.model.Model.utt_shared_v.cpu().detach().numpy())
                features_results['shared_t'].append(self.model.Model.utt_shared_t.cpu().detach().numpy())
                features_results['shared1_a'].append(self.model.Model.utt_shared_a.cpu().detach().numpy())
                features_results['shared1_v'].append(self.model.Model.utt_shared_v.cpu().detach().numpy())
                features_results['shared1_t'].append(self.model.Model.utt_shared_t.cpu().detach().numpy())
                features_results['private_a'].append(self.model.Model.utt_private_a.cpu().detach().numpy())
                features_results['private_v'].append(self.model.Model.utt_private_v.cpu().detach().numpy())
                features_results['private_t'].append(self.model.Model.utt_private_t.cpu().detach().numpy())
                features_results['p_a'].append(self.model.Model.p_a.cpu().detach().numpy())
                features_results['p_v'].append(self.model.Model.p_v.cpu().detach().numpy())
                features_results['p_t'].append(self.model.Model.p_t.cpu().detach().numpy())

            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
            features_results = {
                'shared_a': [],
                'shared_v': [],
                'shared_t': [],
                'shared1_a': [],
                'shared1_v': [],
                'shared1_t': [],
                'private_a': [],
                'private_v': [],
                'private_t': [],
                'p_a': [],
                'p_v': [],
                'p_t': [],
            }
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

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        # TODO: add features
                        # for item in features.keys():
                        #     features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels.cpu().detach().tolist())
                        preds = outputs["M"].cpu().detach().numpy()
                        # test_preds_i = np.argmax(preds, axis=1)
                        sample_results.extend(preds.squeeze())

                        features_results['shared_a'].append(self.model.Model.utt_shared_a.cpu().detach().numpy())
                        features_results['shared_v'].append(self.model.Model.utt_shared_v.cpu().detach().numpy())
                        features_results['shared_t'].append(self.model.Model.utt_shared_t.cpu().detach().numpy())
                        features_results['shared1_a'].append(self.model.Model.utt_shared_a.cpu().detach().numpy())
                        features_results['shared1_v'].append(self.model.Model.utt_shared_v.cpu().detach().numpy())
                        features_results['shared1_t'].append(self.model.Model.utt_shared_t.cpu().detach().numpy())
                        features_results['private_a'].append(self.model.Model.utt_private_a.cpu().detach().numpy())
                        features_results['private_v'].append(self.model.Model.utt_private_v.cpu().detach().numpy())
                        features_results['private_t'].append(self.model.Model.utt_private_t.cpu().detach().numpy())
                        features_results['p_a'].append(self.model.Model.p_a.cpu().detach().numpy())
                        features_results['p_v'].append(self.model.Model.p_v.cpu().detach().numpy())
                        features_results['p_t'].append(self.model.Model.p_t.cpu().detach().numpy())
                    
                    loss = self.criterion(outputs['M'], labels)

                    contrastive_loss = self.get_contrastive_loss(labels)
                    hsic_loss = self.get_hsic_loss()
                    cls3 = self.get_3cls_loss(labels)
                    if self.args.use_cmd_sim:
                        similarity_loss = cmd_loss
                    else:
                        similarity_loss = domain_loss
                    eval_loss += loss.item()
                    whole_loss = loss + \
                                 self.args.hsic_weight * hsic_loss + \
                                 self.args.cls3_weight * cls3 + \
                                 self.args.contralLoss_weight * contrastive_loss
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())


        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features_results.keys():
                features_results[k] = np.concatenate(features_results[k], axis=0)
            eval_results['Features'] = features_results
            eval_results['Labels'] = all_labels

            csv_filename = 'D:/Speech/codes\\MMSA-master_3\MMSA\\results\\normal\\'+ mode + self.discrib + '.pkl'
            f = open(csv_filename, 'wb')
            pickle.dump(eval_results, f)
            f.close()
            print('-'*50 + 'saved in '+ str(csv_filename) + '-'*50)



        return eval_results

    def get_contrastive_loss(self,label):
        # losses between shared states
        loss = contralLoss(self.model.Model.utt_shared_t,label )
        loss += contralLoss(self.model.Model.utt_shared_v, label)
        loss += contralLoss(self.model.Model.utt_shared_a, label)
        return loss/3.0

    def get_hsic_loss(self,):

        x1 = self.model.Model.utt_private_a.cpu().data.numpy()
        y1 = self.model.Model.utt_shared1_a.cpu().data.numpy()
        x2 = self.model.Model.utt_private_v.cpu().data.numpy()
        y2 = self.model.Model.utt_shared1_v.cpu().data.numpy()
        x3 = self.model.Model.utt_private_t.cpu().data.numpy()
        y3 = self.model.Model.utt_shared1_t.cpu().data.numpy()

        z1 = self.model.Model.utt_shared_a.cpu().data.numpy()
        z2 = self.model.Model.utt_shared_v.cpu().data.numpy()
        z3 = self.model.Model.utt_shared_t.cpu().data.numpy()
        b = x1.shape[0]
        pc1, pc2, pc3 = [], [], []
        pc4, pc5, pc6 = [], [], []
        pc7, pc8, pc9 = [], [], []
        for i in range(b):
            h1, h2 = self.loss_hsic(y1[i].reshape(len(y1[i]), 1), y2[i].reshape(len(y2[i]), 1))
            h3, h4 = self.loss_hsic(y2[i].reshape(len(y2[i]), 1), y3[i].reshape(len(y3[i]), 1))
            h5, h6 = self.loss_hsic(y3[i].reshape(len(y3[i]), 1), y1[i].reshape(len(y1[i]), 1))

            hz1, hz2 = self.loss_hsic(z1[i].reshape(len(z1[i]), 1), z2[i].reshape(len(z2[i]), 1))
            hz3, hz4 = self.loss_hsic(z2[i].reshape(len(z2[i]), 1), z3[i].reshape(len(z3[i]), 1))
            hz5, hz6 = self.loss_hsic(z3[i].reshape(len(z3[i]), 1), z1[i].reshape(len(z1[i]), 1))

            # hh1, hh2 = self.loss_hsic(y1[i].reshape(len(z1[i]), 1), z1[i].reshape(len(y1[i]), 1))
            # hh3, hh4 = self.loss_hsic(y2[i].reshape(len(z2[i]), 1), z2[i].reshape(len(y2[i]), 1))
            # hh5, hh6 = self.loss_hsic(y3[i].reshape(len(z3[i]), 1), z3[i].reshape(len(y3[i]), 1))

            pc1.append(h1 / h2)
            pc2.append(h3 / h4)
            pc3.append(h5 / h6)
            pc4.append(hz1 / hz2)
            pc5.append(hz3 / hz4)
            pc6.append(hz5 / hz6)
            # pc7.append(hh1 / hh2)
            # pc8.append(hh3 / hh4)
            # pc9.append(hh5 / hh6)
        p1 = np.array(np.abs(pc1))
        p2 = np.array(np.abs(pc2))
        p3 = np.array(np.abs(pc3))
        p4 = np.array(np.abs(pc4))
        p5 = np.array(np.abs(pc5))
        p6 = np.array(np.abs(pc6))
        # p7 = np.array(np.abs(pc7))
        # p8 = np.array(np.abs(pc8))
        # p9 = np.array(np.abs(pc9))

        p = 1/np.mean(p1)+1/np.mean(p2)+1/np.mean(p3)+1/np.mean(p4)+1/np.mean(p5)+1/np.mean(p6)#+ 1/np.mean(p7)+1/np.mean(p8)+1/np.mean(p9)
        res = p
        return res/6.0
    def get_3cls_loss(self, labels):

        xc = self.model.Model.conC#.cpu().data.numpy()
        x1 = self.model.Model.conA#.cpu().data.numpy()
        x2 = self.model.Model.conV#.cpu().data.numpy()
        x3 = self.model.Model.conT#.cpu().data.numpy()
        criterion = nn.MSELoss()

        p1 = criterion(x1, labels)
        p2 = criterion(x2, labels)
        p3 = criterion(x3, labels)
        pc = criterion(xc, labels)
        res = 0.33*(p1+p2+p3)+pc
        return res*0.5

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

class HSIC(nn.Module):
    def __init__(self):
        super(HSIC, self).__init__()

    def forward(self, X, Y, alph = 0.8):
        """
        X, Y are numpy vectors with row - sample, col - dim
        alph is the significance level
        auto choose median to be the kernel width
        """
        n = X.shape[0]

        # ----- width of X -----
        Xmed = X

        G = np.sum(Xmed*Xmed, 1)
        G = G.reshape(n, 1)
        Q = np.tile(G, (1, n) )
        R = np.tile(G.T, (n, 1) )

        dists = Q + R - 2* np.dot(Xmed, Xmed.T)
        dists = dists - np.tril(dists)
        dists = dists.reshape(n**2, 1)

        width_x = np.sqrt( 0.5 * np.median(dists[dists>0]) )
        # ----- -----

        # ----- width of X -----
        Ymed = Y

        G = np.sum(Ymed*Ymed, 1).reshape(n,1)
        Q = np.tile(G, (1, n) )
        R = np.tile(G.T, (n, 1) )

        dists = Q + R - 2* np.dot(Ymed, Ymed.T)
        dists = dists - np.tril(dists)
        dists = dists.reshape(n**2, 1)

        width_y = np.sqrt( 0.5 * np.median(dists[dists>0]) )
        # ----- -----

        bone = np.ones((n, 1), dtype = float)
        H = np.identity(n) - np.ones((n,n), dtype = float) / n

        K = self.rbf_dot(X, X, width_x)
        L = self.rbf_dot(Y, Y, width_y)

        Kc = np.dot(np.dot(H, K), H)
        Lc = np.dot(np.dot(H, L), H)

        testStat = np.sum(Kc.T * Lc) / n

        varHSIC = (Kc * Lc / 6)**2

        varHSIC = ( np.sum(varHSIC) - np.trace(varHSIC) ) / n / (n-1)

        varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

        K = K - np.diag(np.diag(K))
        L = L - np.diag(np.diag(L))

        muX = np.dot(np.dot(bone.T, K), bone) / n / (n-1)
        muY = np.dot(np.dot(bone.T, L), bone) / n / (n-1)

        mHSIC = (1 + muX * muY - muX - muY) / n

        al = mHSIC**2 / varHSIC
        bet = varHSIC*n / mHSIC

        thresh = gamma.ppf(1-alph, al, scale=bet)[0][0]

        return (testStat, thresh)

    def rbf_dot(self, pattern1, pattern2, deg):
        size1 = pattern1.shape
        size2 = pattern2.shape

        G = np.sum(pattern1 * pattern1, 1).reshape(size1[0], 1)
        H = np.sum(pattern2 * pattern2, 1).reshape(size2[0], 1)

        Q = np.tile(G, (1, size2[0]))
        R = np.tile(H.T, (size1[0], 1))

        H = Q + R - 2 * np.dot(pattern1, pattern2.T)

        H = np.exp(-H / 2 / (deg ** 2))

        return H