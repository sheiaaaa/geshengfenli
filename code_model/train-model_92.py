import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.backends import cudnn



from model.model92 import InceptionResUNet2D

import utils.myutil as myutil
import const
dir_voice = 'FFT/'

dir_checkpoint = 'model/'


def get_corr(fake_Y, Y):
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = torch.mean(fake_Y), torch.mean(Y)
    corr = (torch.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
            torch.sqrt(torch.sum((fake_Y - fake_Y_mean) ** 2)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2)))
    return corr



def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.00002,

              save_cp=True):


    net.cuda()

    Xlist, Xlistcos,Xlistsin,Ylist,Ylistr,Ylisti,inst0,instr0,insti0= myutil.LoadDataset(target="vocal")

    global_step = 0



    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9,0.99))

    criterion2=nn.L1Loss()



    for epoch in range(epochs):
        net.train()

        itemcnt = len(Xlist)
        itemlength = [x.shape[1] for x in Xlist]

        subepoch = sum(itemlength) // const.PATCH_LENGTH // const.BATCH_SIZE
        epoch_loss = 0
        print("subepoch=",subepoch)
        idx = np.arange(0, 999, 1)
        for k in range(500):
            l = random.randint(0, 998)

            idx[l], idx[k] = idx[k], idx[l]
        for subep in range(subepoch):

                X = np.zeros((const.BATCH_SIZE, 1, 512, const.PATCH_LENGTH),
                             dtype="float32")
                Z = np.zeros((const.BATCH_SIZE,  1, 512, const.PATCH_LENGTH),
                             dtype="float32")
                S = np.zeros((const.BATCH_SIZE, 1, 512, const.PATCH_LENGTH),
                             dtype="float32")
                Y = np.zeros((const.BATCH_SIZE, 1, 512, const.PATCH_LENGTH),
                             dtype="float32")
                Yr = np.zeros((const.BATCH_SIZE, 1, 512, const.PATCH_LENGTH),
                              dtype="float32")
                Yi = np.zeros((const.BATCH_SIZE, 1, 512, const.PATCH_LENGTH),
                              dtype="float32")
                inst = np.zeros((const.BATCH_SIZE, 1, 512, const.PATCH_LENGTH),
                              dtype="float32")
                instr = np.zeros((const.BATCH_SIZE, 1, 512, const.PATCH_LENGTH),
                                dtype="float32")
                insti = np.zeros((const.BATCH_SIZE, 1, 512, const.PATCH_LENGTH),
                                dtype="float32")
                cosx = np.zeros((const.BATCH_SIZE, 1, 512, const.PATCH_LENGTH),
                                dtype="float32")
                sinx = np.zeros((const.BATCH_SIZE, 1, 512, const.PATCH_LENGTH),
                               dtype="float32")


                idx_item=[0]*const.BATCH_SIZE
                for k  in range(const.BATCH_SIZE):
                    idx_item[k]=idx[subep%999+k]
                for i in range(const.BATCH_SIZE):
                    tidx = itemlength[idx_item[i]]  - 64

                    randidx = np.random.randint(0,tidx)

                    X[i, 0,:, :] =Xlist[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]

                    Z[i, 0, :, :] = Xlistcos[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    S[i, 0, :, :] = Xlistsin[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Y[i, 0, :, :] =Ylist[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Yr[i, 0, :, :] = Ylistr[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Yi[i, 0, :, :] = Ylisti[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    inst[i, 0, :, :] = inst0[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    instr[i, 0, :, :] = instr0[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    insti[i, 0, :, :] = insti0[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    cosx[i, 0, :, :] = Xlistcos[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    sinx[i, 0, :, :] = Xlistsin[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]

                X = torch.Tensor(X)

                Y = torch.Tensor(Y)
                Yr = torch.Tensor(Yr)
                Yi=torch.Tensor(Yi)
                cosx=torch.Tensor(cosx)
                sinx=torch.Tensor(sinx)
                inst=torch.Tensor(inst)
                instr = torch.Tensor(instr)
                insti = torch.Tensor(insti)
                Z = torch.Tensor(Z)
                S = torch.Tensor(S)
                B=torch.Tensor(B)
                maskb = torch.Tensor(maskb)
                one=torch.Tensor(one)
                zero=torch.Tensor(zero)



                X = X.to(device)

                Z=Z.to(device)
                S = S.to(device)
                Y = Y.to(device)
                Yr = Yr.to(device)
                Yi=Yi.to(device)
                inst=inst.to(device)
                instr=instr.to(device)
                insti=insti.to(device)
                cosx=cosx.to(device)
                sinx=sinx.to(device)


                masks_pred0,masks_pred1,masks= net( X,Z,S )


                pred=X*masks


                pred0=pred*masks_pred0

                pred1=pred*masks_pred1

                r1=cosx*Yr+sinx*Yi

                r2=cosx*Yi-sinx*Yr

                loss1=criterion2(pred0, r1)

                loss2 = criterion2(pred1, r2)

                loss3=criterion2(pred, Y)

                if not torch.isnan(loss3):
                    loss=loss1+loss3+loss2
                else:
                    loss=loss1+loss3
                if subep%100==0:
                    print(loss3.item(),loss1.item(),loss2.item())
                epoch_loss += loss1.item()+loss3.item()

                if subep%1000==0:
                    print("subepoch=",subep,'Loss/train', epoch_loss, global_step)
                    epoch_loss=0

                optimizer.zero_grad()
                loss.backward(retain_graph=True)


                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                if subep%1000==0 and subep>0:
                    torch.save(net.state_dict(),
                               dir_checkpoint + f'MyCP_epoch.pth')


        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'MyCP_epoch_lstm_inception_92_{epoch + 300}.pth')   ##278 is loss4 musdb19 #300 MIR1K

            logging.info(f'Checkpoint {epoch + 1} saved !')



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=True,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Using device {device}')


    net=InceptionResUNet2D()



    if args.load:
        net.load_state_dict(

            torch.load('model/model_92_mir1k.pth', map_location=device)
        )


        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)


    cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=0.1)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
