import argparse
import torch
from torch.utils.data import Dataset
import torch.optim as optim
from torch.autograd import Variable
from model.TSANet import TemporalSpectralAttnNet as mymodel
from utils import *
from utils import LoadDataset, load_folds_data
import time
import random


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

seed = 1029
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    np.random.seed(int(seed))

def preliminary_result(model, train_files):
    train_preliminary = result_save()
    model.eval()
    for file in train_files:
        #print(file)
        data = LoadDataset([file])
        label_y = data.y_data.reshape((1, -1)).type(torch.FloatTensor).cpu().detach().numpy()
        #print("label_y:", label_y.shape)
        outputs = model(Variable(data.x_data.cuda())).reshape((1, -1, 5)).cpu().detach().numpy()
        train_preliminary.add_preliminary_result(file, outputs, label_y)
        #print("outputs:", outputs.shape)
    return train_preliminary


def test(model, model_crf, val_files):
    # Test
    time_begin = time.time()
    val_result = result_save()
    model.eval()
    for file in val_files:
        data = LoadDataset([file])
        label_y = data.y_data.reshape((1, -1)).cpu().detach().numpy()
        output1 = model(Variable(data.x_data.cuda()))
        output2 = model_crf.decode(output1.reshape((1, -1, 5)))
        _, preliminary = torch.max(output1.data, 1)
        val_result.add_all(file, preliminary.cpu().detach().numpy(), np.array(output2[0]), label_y[0])  #(x,)

    return val_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TSANet')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--fold_num', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_ckpt', type=bool, default=True)
    parser.add_argument('--np_data_dir', type=str, default="/data1/fgd/TANDF/edf_78_npz")
    parser.add_argument('--output', type=str, default="./output78-20fold2/")
    args = parser.parse_args()

    fold_data = load_folds_data(args.np_data_dir, args.fold_num)
    for fold in range(args.fold_num):
        train_file, val_file = fold_data.get_file(fold)

        print(val_file)
        train_dataset = LoadDataset(train_file)
        val_dataset = LoadDataset(val_file)
        print(train_dataset.x_data.shape, train_dataset.y_data.shape)
        print(val_dataset.x_data.shape, val_dataset.y_data.shape)

        data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  drop_last=False,
                                                  num_workers=1, worker_init_fn=_init_fn)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=1, worker_init_fn=_init_fn)
        data_count = calc_count(train_dataset, val_dataset)
        weights_for_each_class = calc_class_weight(data_count)
        log_step = int(data_loader.batch_size) * 1
        # print(test_dataset.x_data)
        model = mymodel().cuda()
        lr = 0.001
        criterion = weighted_CrossEntropyLoss
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001, amsgrad=True)
        acc_best = 0
        # train
        for epoch in range(args.epochs):
            model.train()
            tims = time.time()
            correct = 0
            total = 0
            class_correct = list(0. for i in range(5))
            class_total = list(0. for i in range(5))
            for i, (data, target) in enumerate(data_loader):
                data = data.type(torch.FloatTensor)
                target = target.type(torch.FloatTensor)
                data = Variable(data.cuda())
                target = Variable(target.cuda())
                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target.long().cuda(), weights_for_each_class)
                loss.backward()
                optimizer.step()

            train_result = preliminary_result(model, train_file)
            model_crf = train_CRF(model, train_result)

            # val
            val_result = test(model, model_crf, val_file)
            acc = cal_metric(fold, epoch, val_result)
            if acc_best < acc:
                acc_best = acc
                print("[Epoch: %d] acc：%.4f, best_acc: %.4f" %(epoch, acc, acc_best))
                if args.save_ckpt:
                    model_file = str(fold+1)+"_"+str(epoch+1)+"_20.pth"
                    torch.save(model, os.path.join(args.output, "fold{}_model.pth".format(fold)))
                    torch.save(model_crf, os.path.join(args.output, "fold{}_crf.pth".format(fold)))
            # Decaying Learning Rate
            if epoch == 10:
                lr /= 10
                print('reset learning rate to:', lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    print(param_group['lr'])
        # np.save(os.path.join(args.output, "output_fold{}".format(fold)), val_result)


                    



