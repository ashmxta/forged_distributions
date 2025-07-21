#!/usr/bin/env python3
import os
import copy
import shutil
import inspect
import warnings
import numpy as np
import torch
import torchvision
from torch.serialization import add_safe_globals
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
import utils  # renamed module
import model
import opacus.accountants.prv

# allow safe serialization of the PRV accountant
add_safe_globals([opacus.accountants.prv.PRVAccountant])

torch.backends.cudnn.benchmark = True

class train_fn:
    def __init__(
        self,
        lr=0.01,
        batch_size=128,
        dataset='SVHN',
        architecture="resnet20",
        exp_id=None,
        model_dir=None,
        save_freq=None,
        dec_lr=None,
        trainset=None,
        save_name=None,
        num_class=10,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        seed=0,
        optimizer="sgd",
        gamma=0.1,
        overwrite=0,
        epochs=10,
        dp=0,
        sigma=None,
        cn=1,
        delta=1e-5,
        eps=1,
        norm_type='gn',
        sample_data=1,
        poisson=False,
        remove_points=None,
        reduction="sum"
    ):
        # bind all args to self
        for name, param in inspect.signature(train_fn).parameters.items():
            setattr(self, name, eval(name))

        # reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # checkpoint directory
        if self.save_freq:
            self.save_dir = utils.get_save_dir(self.save_name or f"ckpt_{self.dataset}_{architecture}_{self.eps}_{exp_id}")
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
            elif overwrite:
                shutil.rmtree(self.save_dir)
                os.mkdir(self.save_dir)
        else:
            self.save_dir = None

        # load datasets
        self.trainset = utils.load_dataset(self.dataset, train=True, download=True) if trainset is None else trainset
        self.testset  = utils.load_dataset(self.dataset, train=False, download=True)

        # normalize remove_points to a list of ints
        if remove_points is None:
            rpc = []
        elif isinstance(remove_points, (int, np.integer)):
            rpc = [int(remove_points)]
        else:
            rpc = list(remove_points)

        original_size = len(self.trainset)
        effective_size = original_size - len(rpc)

        # build the sequence of indices, dropping all in rpc
        self.sequence = utils.create_sequences(
            batch_size=self.batch_size,
            dataset_size=original_size,
            epochs=self.epochs,
            poisson=poisson,
            sample_data=sample_data,
            remove_points=rpc
        )

        # network setup
        try:
            arch = getattr(model, architecture)
        except AttributeError:
            arch = getattr(torchvision.models, architecture)
        in_channels = 1 if self.dataset.upper() == "MNIST" else 3
        self.net = arch(norm_type=self.norm_type, in_channels=in_channels).to(self.device)

        # standard loaders (used by Opacus, but training uses self.sequence)
        self.train_loader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset,  batch_size=self.batch_size, shuffle=False, pin_memory=True
        )

        # optimizer & scheduler
        num_batches = original_size / self.batch_size
        self.optimizer, self.scheduler = utils.get_optimizer(
            self.dataset, self.net, self.lr, num_batches,
            dec_lr=self.dec_lr, optimizer=self.optimizer, gamma=self.gamma
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # DP wrapping (if requested), using effective_size for sample_rate
        if self.dp:
            self.privacy_engine = PrivacyEngine()
            sample_rate = self.batch_size / effective_size
            self.net, self.optimizer, _ = self.privacy_engine.make_private(
                module=self.net,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=get_noise_multiplier(
                    target_epsilon=self.eps,
                    target_delta=self.delta,
                    sample_rate=sample_rate,
                    epochs=self.epochs,
                    accountant=self.privacy_engine.accountant.mechanism(),
                ),
                max_grad_norm=self.cn,
                loss_reduction=self.reduction,
            )
            self.sigma = self.optimizer.noise_multiplier
        else:
            self.privacy_engine = None

        # optionally load a pretrained checkpoint
        if model_dir:
            self.load(model_dir)

    def save(self, epoch=None, save_path=None):
        assert epoch is not None or save_path is not None
        target = save_path or os.path.join(self.save_dir, f"model_step_{epoch+1}")
        if not os.path.exists(target):
            state = {
                'net': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            if self.scheduler:
                state['scheduler'] = self.scheduler.state_dict()
            if self.privacy_engine:
                state['privacy_engine_accountant'] = self.privacy_engine.accountant
            torch.save(state, target)

    def load(self, path):
        state = torch.load(path, weights_only=False)
        self.net.load_state_dict(state['net'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler and 'scheduler' in state:
            self.scheduler.load_state_dict(state['scheduler'])
        if self.privacy_engine and 'privacy_engine_accountant' in state:
            self.privacy_engine.accountant = state['privacy_engine_accountant']

    def predict(self, inputs):
        out = self.net(inputs)
        # handle tuple or extra dims
        if isinstance(out, tuple) and len(out) == 1:
            out = out[0]
        if out.ndim > 2:
            out = out.squeeze()
        if not isinstance(out, torch.Tensor):
            out = out.logits
        return out

    def update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler:
            self.scheduler.step()

    def compute_loss(self, batch):
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        out = self.predict(x)
        loss = self.criterion(out, y)
        return loss

    def train(self, step):
        self.net.train()
        # skip if checkpoint already beyond this step
        if self.save_dir:
            last = utils.get_last_ckpt(self.save_dir, "model_step_")
            if last >= step+1:
                if last == step+1:
                    self.load(os.path.join(self.save_dir, f"model_step_{last}"))
                return True

        # get this step's indices and train
        idxs = self.sequence[step]
        subset = torch.utils.data.Subset(self.trainset, idxs)
        loader = torch.utils.data.DataLoader(subset, batch_size=len(idxs))
        for data in loader:
            loss = self.compute_loss(data)
            loss.backward()
            self.update()
        # save if needed
        if self.save_dir and (step+1) % self.save_freq == 0:
            self.save(step)
        return False

    def validate(self):
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.testloader:
                x, y = x.to(self.device), y.to(self.device)
                preds = torch.max(self.predict(x), 1)[1]
                total += y.size(0)
                correct += (preds == y).sum().item()
        acc = correct / total
        print(f"Test Accuracy: {100*acc:.2f}%")
        return acc


    def compute_grad(self, data=None, indices=None, step=None, cn=-1):
        self.net.train()
        model_state = self.net.state_dict()
        if data is None:
            if indices is None:
                assert step is not None
                indices = self.sequence[step]
            subset = torch.utils.data.Subset(self.trainset, indices)
            sub_trainloader = torch.utils.data.DataLoader(subset, batch_size=indices.shape[0])
            for data in sub_trainloader:
                break
        batch_size = data[0].shape[0]
        inputs, labels = data[0].to(self.device), data[1].to(self.device)
        outputs = self.predict(inputs.contiguous())
        with torch.no_grad():
            correct = (torch.max(outputs.data, 1)[1] == labels).int().cpu().numpy()
        loss = self.criterion(outputs, labels)
        loss.backward()
        per_sample_grad = []
        for p in self.net.parameters():
            if hasattr(p, 'grad_sample'):
                per_sample_grad.append(p.grad_sample.detach().reshape([batch_size, -1]))
        per_sample_grad = torch.concat(per_sample_grad, 1)
        if cn >= 0:
            per_sample_norm = per_sample_grad.norm(2, dim=-1)
            per_sample_clip_factor = (cn / per_sample_norm).clamp(max=1.0).unsqueeze(-1)
            per_sample_grad = per_sample_grad * per_sample_clip_factor
        self.net.load_state_dict(model_state)
        self.optimizer.zero_grad()
        return per_sample_grad, correct

    def grad_to_sensitivity(self, per_sample_grad, batch_size, expected_batch_size):
        # compute the difference in gradient
        if self.reduction == 'mean':
            scale = (1 / batch_size - 1 / (batch_size - 1))
            sum_grad = torch.sum(per_sample_grad, 0, keepdim=True)
            res = torch.norm(scale * sum_grad + per_sample_grad / (batch_size - 1), p=2, dim=1)
            res = res.cpu().numpy()
        elif self.reduction == 'sum':
            res = torch.norm(per_sample_grad, p=2, dim=1) / expected_batch_size
            res = res.cpu().numpy()
        else:
            raise NotImplementedError(f"reduction strategy {self.reduction} is not recognized")
        return res

    def sensitivity(self, data=None, indices=None, step=None, cn=-1, expected_batch_size=0):
        # indices = [index of point interested, e.g., 0; random indices of a batch, e.g., 9, 4, 14, 90]
        # get the batch of data points
        if data is None:
            if indices is None:
                assert step is not None
                indices = self.sequence[step]
            batch_size = indices.shape[0]
        else:
            batch_size = data[0].shape[0]
        # compute per-sample gradient`
        per_sample_grad, correct = self.compute_grad(data, indices, step, cn)
        res = self.grad_to_sensitivity(per_sample_grad, batch_size, expected_batch_size)
        return res, correct

    def renyi_sen_eqn(self, g, gs, alpha):
        term1 = torch.sum(torch.pow(torch.norm(gs, p=2, dim=1), 2))
        term2 = (alpha - 1) * torch.pow(torch.norm(g, p=2), 2)
        term3 = torch.pow(torch.norm(torch.sum(gs, 0) - (alpha - 1) * g, p=2), 2)
        return term1 - term2 - term3

    def sensitivity_renyi(self, target_batch_index, alpha_batch_indices, alpha, cn=-1):
        # self.net = self.net.to_standard_module()
        # self.net = GradSampleModule(self.net)
        target_grad, _ = self.compute_grad(indices=target_batch_index, cn=cn)
        target_grad = target_grad.cpu()
        alpha_grads = []
        for batch_index in alpha_batch_indices:
            alpha_grads.append(torch.mean(self.compute_grad(indices=batch_index, cn=cn)[0], 0).cpu())
        res = []
        if self.reduction == 'mean':
            target_g = torch.mean(target_grad, 0)
            alpha_g = torch.stack(alpha_grads)
            res.append(self.renyi_sen_eqn(target_g, alpha_g, alpha).item())

        if self.reduction == 'sum':
            target_g = torch.sum(target_grad, 0)
            alpha_g = torch.stack([g * b.shape[0] for b, g in zip(alpha_batch_indices, alpha_grads)])
            res.append(self.renyi_sen_eqn(target_g, alpha_g, alpha).item())
        return res
