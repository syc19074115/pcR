# -- coding: utf-8 --
from DIT import DiT_basic
from dataset import train_dataset,val_dataset
import config
from torch.utils.data import DataLoader
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score,matthews_corrcoef,accuracy_score,f1_score

from utilis.metrics import evaluate as Eva
if __name__ == '__main__':
    args = config.args
    save_path = args.save_path
    device = args.gpu_id
    net = DiT_basic(basic_model='t2t',  # 使用vit还是t2t vit只支持both 因为其他的不需要消融探究
                    patch_emb='isolated',
                    time_emb=True,
                    pos_emb='share',
                    use_scale=True,
                    loss_f='ce',  # focal of ce
                    input_type='both',
                    pool='cls',
                    output_type='score')
    # net = net.to(device)
    test_ds = val_dataset.Val_Dataset(args)
    test_dl = DataLoader(dataset=test_ds,batch_size=1,num_workers=args.n_threads,shuffle=False)
    ##
    ckpt = torch.load('{}/Dit_330.pth'.format(save_path))
    net.load_state_dict(ckpt['net'])
    '''10(有变化)
    tensor([[[-0.2725,  0.6177, -0.7709,  ..., -0.3871, -1.4804,  0.8689],
         [ 1.4212,  0.4864,  1.1093,  ...,  0.0847,  0.4912,  0.9296],
         [-0.7085, -0.1875,  0.4142,  ...,  0.2948,  0.3230, -0.0348],
         ...,
         [-0.3889,  0.1912,  0.6849,  ...,  0.5187,  0.1575, -0.1259],
         [ 0.6347,  0.3645,  0.0762,  ...,  0.8837,  0.2755,  0.1288],
         [ 0.4390,  0.6216, -0.5221,  ..., -0.2441, -1.5565, -0.5753]]],
    
    '''
    # for name, parameter in net.named_parameters():
    #     print(name,parameter)


    true = []
    # predict = []
    proba = []
    net.eval()
    with torch.no_grad():
        for i, (pre, post, label) in tqdm(enumerate(test_dl),total=len(test_dl)):
            out = net(pre,post,None,None)
            # print(out.shape) #(1,2)
            prob = nn.Softmax(dim=1)(out)
            temp = prob[:,1]
            # _, predicted = torch.max(prob.data, 1)
            true.append(label.item())
            # predict.append(predicted.item())
            proba.append(temp.item())
            # print(prob,temp)
    print(true)
    # print(predict)
    print(proba)
    evaluation = Eva(true,proba)
    print(evaluation)


