import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from tqdm import tqdm 


def train_Q3(n_epoch, train_dataloader, test_dataloader, autoencoder, koopman_operator, optimiser_autoencoder, optimiser_koopman):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_batch = len(train_dataloader)  # To be tuned 
    loss1_ep, loss2_ep, loss3_ep, total_loss_ep = [], [], [], []
    loss1_step, loss2_step, loss3_step, total_loss_step = [], [], [], []
    loss1_ep_test, loss2_ep_test, loss3_ep_test, total_loss_ep_test = [], [], [], []

    for epoch in tqdm(range(n_epoch)):
        autoencoder.train()
        koopman_operator.train()
        loss1_sum, loss2_sum, loss3_sum, total_loss_sum = 0, 0, 0, 0

        for tensor2d_batch_x, tensor2d_batch_x_next in train_dataloader:
            tensor2d_batch_x = tensor2d_batch_x.to(device)
            tensor2d_batch_x_next = tensor2d_batch_x_next.to(device)

            optimiser_autoencoder.zero_grad()
            optimiser_koopman.zero_grad()

            tensor2d_observable = autoencoder.encoder(tensor2d_batch_x)
            tensor2d_observable_next = autoencoder.encoder(tensor2d_batch_x_next)

            tensor2d_decoded_x = autoencoder.decoder(tensor2d_observable)

            tensor2d_koopman_observable_next = koopman_operator(tensor2d_observable)

            tensor2d_predict_x_next = autoencoder.decoder(tensor2d_koopman_observable_next)

            total_loss, loss_1, loss_2, loss_3 = loss_koopman(tensor2d_batch_x,
                                        tensor2d_batch_x_next,
                                        tensor2d_decoded_x,
                                        tensor2d_observable_next,
                                        tensor2d_koopman_observable_next,
                                        tensor2d_predict_x_next,
                                        return_all_losses = True)

            total_loss.backward()
            optimiser_autoencoder.step()
            optimiser_koopman.step()

            total_loss_step.append(total_loss.item())
            loss1_step.append(loss_1.item())
            loss2_step.append(loss_2.item())
            loss3_step.append(loss_3.item())

            loss1_sum += loss_1.item()
            loss2_sum += loss_2.item()
            loss3_sum += loss_3.item()
            total_loss_sum += total_loss.item()

        #end of the epoch 
        
        loss1_ep.append(loss1_sum / n_batch)
        loss2_ep.append(loss2_sum / n_batch)
        loss3_ep.append(loss3_sum / n_batch)
        total_loss_ep.append(total_loss_sum / n_batch)

        total_loss_test, loss1_test, loss2_test, loss3_test = test(test_dataloader, autoencoder, koopman_operator, device)
        total_loss_ep_test.append(total_loss_test)
        loss1_ep_test.append(loss1_test)
        loss2_ep_test.append(loss2_test)
        loss3_ep_test.append(loss3_test)

        # if epoch % 1 == 0:
        #     print(epoch, total_train_loss)

        # if epoch % 10 == 0:

    train_step = {
        'total_loss': total_loss_step,
        'loss1': loss1_step,
        'loss2': loss2_step,
        'loss3': loss3_step
    }

    train_ep = {
        'total_loss': total_loss_ep,
        'loss1': loss1_ep,
        'loss2': loss2_ep,
        'loss3': loss3_ep
    }

    test_ep = {
        'total_loss': total_loss_ep_test,
        'loss1': loss1_ep_test,
        'loss2': loss2_ep_test,
        'loss3': loss3_ep_test
    }

    # Return the three dictionaries
    return train_step, train_ep, test_ep

    # return total_loss_ep, loss1_ep, loss2_ep, loss3_ep, total_loss_step, loss1_step, loss2_step, loss3_step, total_loss_ep_test, loss1_ep_test, loss2_ep_test, loss3_ep_test

def test(test_dataloader, autoencoder, koopman_operator, device):

    autoencoder.eval()
    koopman_operator.eval()
    loss1_sum, loss2_sum, loss3_sum, total_loss_sum = 0, 0, 0, 0
    # loss1_ep, loss2_ep, loss3_ep, total_loss_ep = [], [], [], []
    # loss1_step, loss2_step, loss3_step, total_loss_step = [], [], [], []

    tensor_loss_val = None
    with torch.no_grad():
        # total_test_loss = 0
        for tensor2d_batch_x, tensor2d_batch_x_next in test_dataloader:
            tensor2d_batch_x = tensor2d_batch_x.to(device)
            tensor2d_batch_x_next = tensor2d_batch_x_next.to(device)

            tensor2d_observable = \
                autoencoder.encoder(tensor2d_batch_x)
            tensor2d_observable_next = \
                autoencoder.encoder(tensor2d_batch_x_next)
            tensor2d_decoded_x = \
                autoencoder.decoder(tensor2d_observable)
            tensor2d_koopman_observable_next = \
                koopman_operator(tensor2d_observable)
            tensor2d_predict_x_next = \
                autoencoder.decoder(tensor2d_koopman_observable_next)

            total_loss, loss_1, loss_2, loss_3 = loss_koopman(tensor2d_batch_x,
                                        tensor2d_batch_x_next,
                                        tensor2d_decoded_x,
                                        tensor2d_observable_next,
                                        tensor2d_koopman_observable_next,
                                        tensor2d_predict_x_next,
                                        return_all_losses = True)

            # total_loss_step.append(total_loss.item())
            # loss1_step.append(loss_1.item())
            # loss2_step.append(loss_2.item())
            # loss3_step.append(loss_3.item())

            loss1_sum += loss_1.item()
            loss2_sum += loss_2.item()
            loss3_sum += loss_3.item()
            total_loss_sum += total_loss.item()
    n_test = len(test_dataloader)
    total_loss_sum, loss1_sum, loss2_sum, loss3_sum = total_loss_sum / n_test, loss1_sum / n_test, loss2_sum / n_test, loss3_sum / n_test
    return total_loss_sum, loss1_sum, loss2_sum, loss3_sum

def loss_koopman(tensor2d_x: torch.Tensor,
                 tensor2d_x_next: torch.Tensor,
                 tensor2d_decoded_x: torch.Tensor,
                 tensor2d_observable_next: torch.Tensor,
                 tensor2d_koopman_observable_next: torch.Tensor,
                 tensor2d_predict_x_next: torch.Tensor,
                 return_all_losses = False):

    # TODO: Implement the loss function here

    loss_1 = torch.norm(tensor2d_decoded_x - tensor2d_x)
    loss_2 = torch.norm(tensor2d_koopman_observable_next - tensor2d_observable_next)
    loss_3 = torch.norm(tensor2d_predict_x_next - tensor2d_x_next)
    total_loss = loss_1 + loss_2 + loss_3 
    if return_all_losses:
        return total_loss, loss_1, loss_2, loss_3
    return total_loss