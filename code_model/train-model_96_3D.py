import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.backends import cudnn

from model.model96_3D import InceptionResUNet3D

import utils.myutil as myutil
import const
dir_voice = 'FFT3D/'
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
    Xlistch0, Xlistch1, Xlistch2, Xlistch3, Xlistch2r,Ylist0,Ylist1,Ylist2,Ylist3,Xitd0,Xitd1,Xitd2,Xitd3, \
        Xitd4,Xitd5,Xild0,Xild1,Xild3,Xild4,Xild5,cos20,cos21,cos23= myutil.LoadDataset3D_8channel(target="vocal")
    print()

    global_step = 0
    gap=16


    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    criterion2=nn.L1Loss()


    for epoch in range(epochs):
        net.train()
        itemcnt = len(Xlistch0)
        itemlength = [x.shape[1] for x in Xlistch0]

        subepoch = sum(itemlength) // const.PATCH_LENGTH // const.BATCH_SIZE * 4
        epoch_loss = 0
        print("subepoch=",subepoch)
        for subep in range(subepoch):

                X = np.zeros((const.BATCH_SIZE, 1,4, 512, const.PATCH_LENGTH),
                             dtype="float32")
                Z = np.zeros((const.BATCH_SIZE, 1, 8, 512, const.PATCH_LENGTH),
                             dtype="float32")

                Y0 = np.zeros((const.BATCH_SIZE, 1, 1, 512, const.PATCH_LENGTH),
                             dtype="float32")
                Y1 = np.zeros((const.BATCH_SIZE, 1, 1, 512, const.PATCH_LENGTH),
                              dtype="float32")
                Y2 = np.zeros((const.BATCH_SIZE, 1,1, 512, const.PATCH_LENGTH),
                              dtype="float32")
                Y3 = np.zeros((const.BATCH_SIZE, 1, 1, 512, const.PATCH_LENGTH ),
                              dtype="float32")
                Ild0 = np.zeros((const.BATCH_SIZE, 1, 1, 512, const.PATCH_LENGTH),
                              dtype="float32")
                Ild1 = np.zeros((const.BATCH_SIZE, 1, 1, 512, const.PATCH_LENGTH ),
                              dtype="float32")
                Ild3 = np.zeros((const.BATCH_SIZE, 1, 1, 512, const.PATCH_LENGTH ),
                              dtype="float32")
                Ild4 = np.zeros((const.BATCH_SIZE, 1, 1, 512, const.PATCH_LENGTH),
                                dtype="float32")
                Ild5 = np.zeros((const.BATCH_SIZE, 1, 1, 512, const.PATCH_LENGTH),
                                dtype="float32")
                idx_item = np.random.randint(0, itemcnt, const.BATCH_SIZE)

                if itemlength[idx_item[0]]<160:
                    continue
                for i in range(const.BATCH_SIZE):
                    tidx = itemlength[idx_item[i]] - 2*gap-const.PATCH_LENGTH - 1

                    if tidx < 0:
                        print(tidx)
                    randidx = np.random.randint(gap,tidx)

                    X[i, 0, 0,:, :] =Xlistch0[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    X[i, 0, 1, :, :] = Xlistch1[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    X[i, 0, 2, :, :] = Xlistch2[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    X[i, 0, 3, :, :] = Xlistch3[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]



                    Z[i, 0,0, :, :] = Xitd3[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Z[i, 0, 1, :, :] = Xitd4[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Z[i, 0, 2, :, :] = Xitd5[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Z[i, 0, 3, :, :] = cos23[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Z[i, 0, 4, :, :] =cos21[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Z[i, 0,5, :, :] = cos20[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]

                    Z[i, 0, 6, :, :] = Xild0[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Z[i, 0, 7, :, :] = Xild3[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Y0[i, 0,0, :, :] =Ylist0[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Y1[i, 0,0, :, :] = Ylist1[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Y2[i, 0,0, :, :] = Ylist2[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Y3[i, 0,0, :, :] = Ylist3[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Ild0[i, 0, 0, :, :] = Xild0[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Ild1[i, 0, 0, :, :] = Xild1[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Ild3[i, 0, 0, :, :] = Xild3[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Ild4[i, 0, 0, :, :] = Xild4[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]
                    Ild5[i, 0, 0, :, :] = Xild5[idx_item[i]][1:, randidx:randidx + const.PATCH_LENGTH]

                X = torch.Tensor(X)

                Y0 = torch.Tensor(Y0)
                Y1 = torch.Tensor(Y1)
                Y2 = torch.Tensor(Y2)
                Y3 = torch.Tensor(Y3)

                Z = torch.Tensor(Z)
                Ild0=torch.Tensor(Ild0)
                Ild1 = torch.Tensor(Ild1)
                Ild3 = torch.Tensor(Ild3)
                Ild4 = torch.Tensor(Ild4)
                Ild5 = torch.Tensor(Ild5)
                X = X.to(device)

                Z=Z.to(device)

                Y0 = Y0.to(device)
                Y1 = Y1.to(device)
                Y2 = Y2.to(device)
                Y3 = Y3.to(device)

                Ild0=Ild0.to(device)
                Ild1 = Ild1.to(device)
                Ild3 = Ild3.to(device)
                Ild4 = Ild4.to(device)
                Ild5 = Ild5.to(device)
                mag,mask_cos2,mask_sin2= net( X,Z )

                X_0=X[:,:,0,:,:const.PATCH_LENGTH]
                X_1 = X[:, :, 1, :, :const.PATCH_LENGTH]
                X_2 = X[:, :, 2, :, :const.PATCH_LENGTH]
                X_3 = X[:, :, 3, :, :const.PATCH_LENGTH]

                Y_0=Y0[:,:,0,:,:const.PATCH_LENGTH]
                Y_1 = Y1[:, :, 0, :, :const.PATCH_LENGTH]
                Y_2 = Y2[:, :, 0, :, :const.PATCH_LENGTH]
                Y_3 = Y3[:, :, 0, :, :const.PATCH_LENGTH]
                Ild_0 = Ild0[:, :, 0, :, :const.PATCH_LENGTH]
                Ild_1 = Ild1[:, :, 0, :, :const.PATCH_LENGTH]
                Ild_3 = Ild3[:, :, 0, :, :const.PATCH_LENGTH]
                Ild_4 = Ild4[:, :, 0, :, :const.PATCH_LENGTH]
                Ild_5 = Ild5[:, :, 0, :, :const.PATCH_LENGTH]

                pred2=X_2*mag
                pred0 =pred2 * mask_cos2

                pred1 = pred2 * mask_sin2

                r1 = Ild_0 * Ild_1 + Ild_3 * Ild_4

                r2 = Ild_0 * Ild_4 - Ild_3 * Ild_1


                loss1=criterion2(pred0, r1)+criterion2(pred1,r2)
                loss2=criterion2(pred2,Y_2)

                loss3 = (torch.tensor(1.0) - get_corr(pred2, Y_2))



                if not torch.isnan(loss3):
                    loss=loss1+loss2+loss3
                else:
                    loss=loss1+loss2
                if subep%100==0:
                    print(loss3.item(),loss1.item(),loss2.item())
                epoch_loss += loss1.item()+loss2.item()+loss3.item()
                if subep%1000==0:
                    print("subepoch=",subep,'Loss/train', epoch_loss, global_step)
                    epoch_loss=0


                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                if subep%1000==0 and subep>0:
                    torch.save(net.state_dict(),
                               dir_checkpoint + f'model96_3D.pth')


        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'mdoel96_3d_{epoch}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=30,
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Using device {device}')


    net=InceptionResUNet3D()

    if args.load:
        net.load_state_dict(

            torch.load('unet/model96_3D.pth', map_location=device)
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
