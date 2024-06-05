from itertools import count
import sys, os,pprint
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
DEPENDENCIES_PATH = os.path.join(FILE_DIR, '..', 'deep-head-pose-master/code')
sys.path.append(DEPENDENCIES_PATH)
from common.logger import create_logger
from dataset.data_loader import ImagerLoader, TestImagerLoader
# from dataset.data_loader_old import TestImagerLoader

from common.utils import PostProcessor, get_transform, save_checkpoint, TestPostProcessor
from common.config import argparser
import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import hopenet
dic={}

def main(args):
    logger = create_logger(args)
    logger.info(pprint.pformat(args))
    # train_dataset = ImagerLoader(args.source_path, args.train_file, args.json_path,
    #                                  args.gt_path, stride=1, transform=get_transform(False),args=args)
    
    # val_dataset = ImagerLoader(args.source_path, args.val_file, args.json_path,
    #                                  args.gt_path, stride=1, transform=get_transform(False),args=args)
    
    test_dataset = TestImagerLoader(args.test_path, args, stride=1, transform=get_transform(False))
    
    # print(test_dataset[4990][1])
    
    global dic
    params = {'shuffle': False}
    
    # train_loader = torch.utils.data.DataLoader(
    #             train_dataset,
    #             batch_size=args.batch_size,
    #             num_workers=args.num_workers,
    #             pin_memory=False,
    #             **params)
    
    # val_loader = torch.utils.data.DataLoader(
    #             val_dataset,
    #             batch_size=args.batch_size,
    #             num_workers=args.num_workers,
    #             pin_memory=False,
    #             **params)
    
    test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                num_workers=args.num_workers,
                pin_memory=False,
                **params)
    
    snapshot_path = 'checkpoints/hopenet_robust_alpha1.pkl'
    gpu = 0
    cudnn.enabled = True
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.to(gpu)
    model.eval()

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    head_dic={}
    
    transformations = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    cnt=0
    for i,  (source_frame, target,angle,head2D,max_index) in enumerate(test_loader):
    # for i, (source_frame, target) in enumerate(test_loader):  
        
        # print(target)
        
        
        frames=source_frame[:,3]
        frames=frames.cuda()
        source_frame=source_frame.cuda()
        
        with torch.no_grad():
            yaw_batch, pitch_batch, roll_batch = model(frames)
        batch_size=source_frame.shape[0]
        # uid1, trackid1, frameid1, _, label1=target
        uid1, trackid1, unique_id1, frameid1 = target
        
        for j in range(batch_size):
            yaw=yaw_batch[j].unsqueeze(0)
            pitch=pitch_batch[j].unsqueeze(0)
            roll=roll_batch[j].unsqueeze(0)
            yaw_predicted = F.softmax(yaw)
            pitch_predicted = F.softmax(pitch)
            roll_predicted = F.softmax(roll)
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
            # uid, trackid,frameid,label = uid1[j],trackid1[j],frameid1[j].item(),label1[j]
            uid, trackid,unique_id, frameid = uid1[j],trackid1[j],unique_id1[j],frameid1[j]
            key=uid+":"+trackid+":"+unique_id+":"+str(frameid)
            if key=='d4029258eac9865310f98ae3122f7824:beab8bb572fc02abd3fb9c878d000986:63a38dBeec9fc770e51f3e8f2e479dcd:02056':
                print(1)
            if key not in head_dic.keys():
                # head_dic[key]=(yaw_predicted.item(),pitch_predicted.item(),roll_predicted.item(),label.item(),0,0)
                head_dic[key]=(yaw_predicted.item(),pitch_predicted.item(),roll_predicted.item())
        if i%100==0:
            print('Processed: [{0}/{1}]\t'.format(
                        i, len(test_loader)))
        
        break

    with open('test_headpose_1.json', 'w') as f:
        print(len(head_dic.keys()))
        json.dump(head_dic, f)

def run():
    args = argparser.parse_args()
    main(args)
if __name__ == '__main__':
    run()
    