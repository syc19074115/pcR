# -- coding: utf-8 --
from torch.utils.tensorboard import SummaryWriter
from DIT import DiT_basic
from dataset import train_dataset,val_dataset
import config
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:32'
import numpy as np
from torch.utils.data import DataLoader
import torch,gc
from torch import nn
from tqdm import tqdm
from collections import OrderedDict

def main(args):
    args = args
    save_path = args.save_path
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    print(device)
    net = DiT_basic(basic_model='t2t',  # 使用vit还是t2t vit只支持both 因为其他的不需要消融探究
                    patch_emb='isolated',
                    time_emb=True,
                    pos_emb='share',
                    use_scale=True,
                    loss_f='ce',  # focal of ce
                    input_type='both',
                    pool='cls',
                    output_type='probability')
    # net = net.to(device)
    # net.loss = net.loss.to(device)
    gc.collect()
    torch.cuda.empty_cache()
    #dataset
    train_ds = train_dataset.Train_Dataset(args)
    val_ds = val_dataset.Val_Dataset(args)
    #dataloader
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
    val_dl = DataLoader(dataset=val_ds,batch_size=1,num_workers=args.n_threads,shuffle=False)
    #tensorboard
    tb_writer = SummaryWriter('runs/dit-first') #创立tensorboard，日志为空则其自动创立
    #jiazia
    net_list = net.state_dict()
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location='cpu')
        new_model_dict = {}
        for layer_name, val in net.state_dict().items():
            if "transformer" in layer_name:
                new_model_dict[layer_name] = val
            else:
                pass
        module_lst = [i for i in new_model_dict]
        print(module_lst)
        weights = OrderedDict()
        new_model_dict = {}
        for layer_name, val in weights_dict['state_dict_ema'].items():
            if "block" in layer_name:
                new_model_dict[layer_name] = val
            else:
                pass
        premodule_lst = [i for i in new_model_dict]
        print(premodule_lst)
        for i in range(len(module_lst)):
            if net_list[module_lst[i]].numel() == weights_dict['state_dict_ema'][premodule_lst[i]].numel():
                weights[module_lst[i]] = weights_dict['state_dict_ema'][premodule_lst[i]]
                print(module_lst[i])

        print(net.load_state_dict(weights, strict=False))

    #optimizer
    optimizer = torch.optim.AdamW(net.parameters(),lr=args.lr,betas=(0.9,0.999),weight_decay=0.03)
    # optimizer = torch.optim.AdamW(pg,lr=args.lr,betas=(0.9,0.999),weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0.0000000001,  verbose=False)
    optimizer.zero_grad()  #
    #step
    total_train_step = 0
    total_test_step = 0
    warmup_steps = 2
    learning_rate = 0.01
    for epoch in range(args.epochs):

        if epoch < warmup_steps:
            # warmup_percent_done = epoch / warmup_steps
            # warmup_learning_rate = args.lr * warmup_percent_done  #gradual warmup_lr
            # learning_rate = warmup_learning_rate
            # learning_rate = 0.01
            print(learning_rate)
        # else:
        #     #learning_rate = np.sin(learning_rate)  # 预热学习率结束后,学习率呈sin衰减
        #     learning_rate = learning_rate**1.1 #预热学习率结束后,学习率呈指数衰减(近似模拟指数衰减)
        #     print(learning_rate)

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
            print(optimizer.state_dict()['param_groups'][0]['lr'])
        #训练
        net.train()  # 设置成训练模式，作用其实不大，对Dropout、Batchnorm层起作用
        print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        # if epoch > warmup_steps:
        #     print(1)
        #     scheduler.step()
        #     learning_rate = scheduler.get_lr()[0]
        # continue
        total_train_loss = 0
        total_train_acc = 0
        for i, (pre, post, label) in tqdm(enumerate(train_dl),total=len(train_dl)):
            # pre = pre.to(device)
            # post = post.to(device)
            # label = label.to(device)
            # output , predict = net(pre, post, label,label)
            train_loss , train_acc = net(pre,post,label,label)
            print(label)
            # train_loss = my_loss(output,label)
            # train_acc = (predict == label).sum().item() / output.size(0)
            total_train_loss += train_loss
            total_train_acc += train_acc
            train_loss.backward()
            # total_train_step += 1
            optimizer.step()
            optimizer.zero_grad() #
            # print(train_loss)
            # print("Train_loss = {}, Train_acc = {}".format(total_train_loss,total_train_acc))

        print("Train_loss = {}, Train_acc = {}".format(total_train_loss/len(train_ds),total_train_acc/len(train_ds)))
        if epoch > warmup_steps:
            scheduler.step()

        #save model
        if epoch/10 == 0:
            state = {'net': net.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
            save_name = os.path.join(save_path,"Dit_{}.pth".format(epoch))
            torch.save(state, save_name)

        #测试
        net.eval()
        total_val_loss = 0
        total_val_acc = 0
        with torch.no_grad():
            for i, (pre, post, label) in tqdm(enumerate(val_dl), total=len(val_dl)):
                # pre = pre.to(device)
                # post = post.to(device)
                # label = label.to(device)
                # output, predict = net(pre, post, label, label)
                # val_loss = my_loss(output, label)
                # val_acc = (predict == label).sum().item() / output.size(0)
                val_loss, val_acc = net(pre, post, label,label)
                total_val_loss += val_loss
                total_val_acc += val_acc
        print("Val_loss = {}, Val_acc = {}".format(total_val_loss/len(val_ds),total_val_acc/len(val_ds)))
        tb_writer.add_scalar("loss",total_val_loss/len(val_ds),epoch)
        tb_writer.add_scalar("acc",total_val_acc/len(val_ds),epoch)
        tb_writer.add_scalar("lr",optimizer.param_groups[0]["lr"],epoch)
        gc.collect()
        torch.cuda.empty_cache()
    tb_writer.close()

if __name__ == '__main__':
    main(config.args)