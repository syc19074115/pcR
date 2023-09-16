from torch.utils.tensorboard import SummaryWriter
from work_model.DIT import DiT_basic
from work_model.resDIT import resDiT
from dataset import train_dataset, val_dataset
import config
import os

os.environ['CUDA_VISIBLE_DEVICES']='2'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
import numpy as np
from torch.utils.data import DataLoader
import torch, gc
from tqdm import tqdm
import random
from utilis.scheduler import CosineAnnealingLRWarmup
import time
from utilis.pnp import EMA
from work_model.DIT import DiT_basic

def randseed(args):
    torch.cuda.empty_cache()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    #cudnn.deterministic = True

def l1_norm(model,lamda,classify_loss):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(abs(param))
    
    #calssify_loss = criterion(pred,target)
    loss = classify_loss + lamda * regularization_loss
    return loss

def main():
    args = config.args
    randseed(args)
    save_path = args.save_path
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # device = "cpu"
    print(device)
    if args.model == 'resDiT':
        print("resDiT")
        net = resDiT(basic_model='t2t',  # 使用vit还是t2t vit只支持both 因为其他的不需要消融探究
                        patch_emb='isolated',
                        #patch_emb='share',
                        time_emb=True,
                        pos_emb='share',
                        use_scale=True,
                        #use_scale=False,
                        loss_f='ce',  # ce  # focal of ce
                        input_type='both',
                        pool='cls',
                        #pool='mean',
                        output_type='probability')
    else:
        print("DiT")
        net = DiT_basic(basic_model='t2t',  # 使用vit还是t2t vit只支持both 因为其他的不需要消融探究
                        patch_emb='isolated',
                        #patch_emb='share',
                        time_emb=True,
                        pos_emb='share',
                        use_scale=True,
                        #use_scale=False,
                        loss_f='ce',  # ce  # focal of ce
                        input_type='both',
                        pool='cls',
                        #pool='mean',
                        output_type='probability')

    net = net.to(device)
    net.loss = net.loss.to(device)
    #0print(next(net.parameters()).device) 
    ckpts = args.save_path
    os.makedirs(ckpts, exist_ok=True)
    #ema = EMA(net, 0.999)
    #ema.register()

    # data
    train_ds = train_dataset.Train_Dataset(args)
    val_ds = val_dataset.Val_Dataset(args)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
    val_dl = DataLoader(dataset=val_ds, batch_size=1, num_workers=args.n_threads, shuffle=True)
    tb_writer = SummaryWriter(args.log_path)  # 创立tensorboard，日志为空则其自动创立
    # optimizer
    optimizer = torch.optim.AdamW(net.parameters(),
                                  lr=args.lr,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.03)
    lr_scheduler_warmup = CosineAnnealingLRWarmup(optimizer,
                                                  T_max=args.epochs,
                                                  eta_min=1.0e-6, #1.0e-6
                                                  last_epoch=-1,
                                                  warmup_steps=args.warmup,
                                                  warmup_start_lr=1.0e-9)
    optimizer.zero_grad()  #

    begin_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume,map_location=device)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler_warmup.load_state_dict(checkpoint['scheduler'])
        ckpt_epoch = checkpoint['epoch']
        begin_epoch = ckpt_epoch
    best_train_loss = 100000
    best_val_loss = 100000

    for epoch in range(args.epochs - begin_epoch):
        # 训练
        if args.resume is not None:
            epoch = ckpt_epoch
            ckpt_epoch += 1

        net.train()  # 设置成训练模式，作用其实不大，对Dropout、Batchnorm层起作用
        # print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        print("=======Epoch:{}=======lr:{}".format(epoch, lr_scheduler_warmup.get_last_lr()))
        # tb_writer.add_scalar("lr", optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        # tb_writer.add_scalar("lr", lr_scheduler_warmup.get_last_lr(),epoch)
        total_train_loss = 0
        total_train_acc = 0
        for i, (pre, post, label) in tqdm(enumerate(train_dl), total=len(train_dl)):
            pre = pre.to(device)
            post = post.to(device)
            label = label.to(device)
            train_loss, train_acc = net(pre, post, label, label)
            total_train_loss += train_loss
            total_train_acc += train_acc
            train_loss = l1_norm(net,args.lamda,train_loss)
            optimizer.zero_grad()  #
            train_loss.backward()
            optimizer.step()
            #ema.update()
            # print(label)
        lr_scheduler_warmup.step()

        print(
            "Train_loss = {}, Train_acc = {}".format(total_train_loss / len(train_dl), total_train_acc / len(train_dl)))
        tb_writer.add_scalar("tarin_loss", total_train_loss / len(train_dl), epoch)
        tb_writer.add_scalar("train_acc", total_train_acc / len(train_dl), epoch)

        # save model
        ckpts_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler_warmup.state_dict(),

        },
            ckpts_name)
        if (epoch+1) % 100 == 0:
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': lr_scheduler_warmup.state_dict(),
            }
            save_name = os.path.join(save_path, "Dit_{}.pth".format(epoch+1))
            torch.save(state, save_name)

        if total_train_loss < best_train_loss:
            best_train_loss = total_train_loss
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': lr_scheduler_warmup.state_dict(),
            }
            save_name = os.path.join(save_path, "best_train.pth".format(epoch))
            torch.save(state, save_name)

        # 测试
        net.eval()
        total_val_loss = 0
        total_val_acc = 0
        with torch.no_grad():
            #ema.apply_shadow()
            for i, (pre, post, label) in tqdm(enumerate(val_dl), total=len(val_dl)):
                pre = pre.to(device)
                post = post.to(device)
                label = label.to(device)
                val_loss, val_acc = net(pre, post, label, label)
                total_val_loss += val_loss
                total_val_acc += val_acc
                # print(label)
            #ema.restore()
        print("Val_loss = {}, Val_acc = {}".format(total_val_loss / len(val_dl), total_val_acc / len(val_dl)))
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': lr_scheduler_warmup.state_dict(),
            }
            save_name = os.path.join(save_path, "best_val.pth".format(epoch))
            torch.save(state, save_name)
        tb_writer.add_scalar("loss", total_val_loss / len(val_dl), epoch)
        tb_writer.add_scalar("acc", total_val_acc / len(val_dl), epoch)
        tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        gc.collect()
        torch.cuda.empty_cache()
    tb_writer.close()

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("总耗时：{}h{}m{}s".format((end_time - start_time)//60//60,(end_time - start_time)//60%60,(end_time - start_time)%60))
