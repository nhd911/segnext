import argparse, os, time, sys, tqdm
import cv2
import logging
import torch
from dataset import DetectorDataset, collate_fn, categories
from model import Model
from lib.modules.loss import Criterion
from torch.utils.data import DataLoader
from utils import unnormalize, normalize, multi_apply, post_process
import matplotlib.pyplot as plt

logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
use_gpu = torch.cuda.is_available()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(opt):
    train_data = DetectorDataset(opt, mode='train')
    valid_data = DetectorDataset(opt, mode='val')

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                              num_workers=opt.workers, collate_fn=collate_fn, pin_memory=use_gpu)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True, drop_last=False, num_workers=opt.workers,
                              collate_fn=collate_fn, pin_memory=use_gpu)
    model = torch.nn.DataParallel(Model(opt))

    optimizer = torch.optim.Adadelta(model.parameters(), lr=opt.lr)
    criterion = Criterion()

    logging.info('Number of parameters: %d' % count_parameters(model))
    if use_gpu:
        logging.info('Device name: %s' % torch.cuda.get_device_name(torch.cuda.current_device()))

    if use_gpu:
        model.cuda()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        criterion = criterion.cuda()

    curr_epoch = 0
    if os.path.exists(opt.saved_model):
        if use_gpu:
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        ckpt = torch.load(opt.saved_model, map_location=map_location)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        curr_epoch = ckpt['global_step']
        logging.info('Restored model is loaded')

    logging.info('Training...')

    for epoch in range(curr_epoch+1, opt.epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, (image, gt_masks) in enumerate(train_loader):
            start = time.time()
            if use_gpu:
                image, gt_masks = image.cuda(), gt_masks.cuda()

            P, _, B = model(image)
            loss_P, loss_B = criterion([P, B], [gt_masks, gt_masks])
            loss = loss_P + loss_B
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.grad_clip, norm_type=2)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            end = time.time()
            sys.stdout.write('\rEpoch: %03d, Step: %04d/%d, Probability Loss: %.9f, Binary Loss: %.9f, Time training: %.2f secs' % (epoch, step+1, len(train_loader), loss_P.item(), loss_B.item(), end-start))

        torch.save({'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': epoch}, opt.saved_model)

    #     if epoch % opt.valInterval == 0 or epoch == opt.epochs:
    # # print()
    #         logging.info('Validating...')

    #         model.eval()
            
    #         categories = ['ticket_1', 'address_2', 'employment_0', 'no_0', 'employment_1', 
    #                 'ticket_0', 'no_1', 'backside_0', 'address_3', 'no_2', 'address_0', 'address_1']
    #         font = cv2.FONT_HERSHEY_SIMPLEX

    #         fontScale = 0.6

    #         # Blue color in BGR
    #         color = (0, 0, 255)

    #         with torch.no_grad():
    #             for idx, (image, gt_masks) in tqdm.tqdm(enumerate(valid_loader)):
    #                 if use_gpu:
    #                     image = image.cuda()
            
    #                 P, T = model(image)
    #                 binary_map = P >= T
            
    #                 # Post-processing
    #                 image = unnormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #                 image = image[0].permute(1, 2, 0).cpu().numpy()
    #                 image = (image * 255).astype('uint8').copy()
            
    #                 P = P[0].sum(dim=0)
    #                 P = normalize(P.flatten(), dim=0).reshape(P.size())
    #                 P = P.cpu().numpy()
            
    #                 post_binary_map = [m for m in binary_map[0].cpu().numpy().astype('uint8')]
    #                 pred_bboxes = multi_apply(post_process, post_binary_map)
    #                 # print(pred_bboxes)
    #                 for i, bboxes in enumerate(pred_bboxes):
    #                     for (x1, y1, x2, y2) in bboxes:
    #                         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #                         cv2.putText(image, categories[i], (x1, y1-5), font,
    #                                     fontScale, color, 1, cv2.LINE_AA)
            
    #                 binary_map = binary_map[0].sum(dim=0)
    #                 binary_map = normalize(binary_map.flatten(), dim=0).reshape(binary_map.size())
    #                 binary_map = binary_map.cpu().numpy()


    #                 fig, axs = plt.subplots(2, 1)
    #                 axs[0].imshow(image)
    #                 axs[0].axis('off')
    #                 axs[0].set_title('Output')
    #                 axs[1].imshow(P)
    #                 axs[1].axis('off')
    #                 axs[1].set_title('Probability map')
    #                 fig.tight_layout()
    #                 plt.show()
    #                 plt.close('all')


    # from PIL import Image, ImageOps
    # from dataset import DataTransformer
    # import matplotlib.pyplot as plt
    # from utils import unnormalize

    # dt = DataTransformer(opt, mode='val')

    # model.eval()

    # categories = ['ticket_1', 'address_2', 'employment_0', 'no_0', 'employment_1', 
    #         'ticket_0', 'no_1', 'backside_0', 'address_3', 'no_2', 'address_0', 'address_1']
    # font = cv2.FONT_HERSHEY_SIMPLEX

    # fontScale = 0.6

    # # Blue color in BGR
    # color = (0, 0, 255)
    
    # eval_dir = './eval'
    # with torch.no_grad():
    #     for fname in os.listdir(eval_dir):
    #         image = Image.open(os.path.join(eval_dir, fname))
    #         image = ImageOps.exif_transpose(image)
    #         image = dt(image).unsqueeze(dim=0)
            
    #         if use_gpu:
    #             image = image.cuda()

    #         P, T = model(image)
    #         binary_map = P >= T

    #         # Post-processing
    #         image = unnormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #         image = image[0].permute(1, 2, 0).cpu().numpy()
    #         image = (image * 255).astype('uint8').copy()

    #         P = P[0].sum(dim=0)
    #         P = normalize(P.flatten(), dim=0).reshape(P.size())
    #         P = P.cpu().numpy()

    #         post_binary_map = [m for m in binary_map[0].cpu().numpy().astype('uint8')]
    #         pred_bboxes = multi_apply(post_process, post_binary_map)
    #         # print(pred_bboxes)

    #         for i, bboxes in enumerate(pred_bboxes):
    #             for (x1, y1, x2, y2) in bboxes:
    #                 cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #                 cv2.putText(image, categories[i], (x1, y1-5), font,
    #                             fontScale, color, 1, cv2.LINE_AA)

    #         binary_map = binary_map[0].sum(dim=0)
    #         binary_map = normalize(binary_map.flatten(), dim=0).reshape(binary_map.size())
    #         binary_map = binary_map.cpu().numpy()


    #         fig, axs = plt.subplots(2, 1)
    #         axs[0].imshow(image)
    #         axs[0].axis('off')
    #         axs[0].set_title('Output')
    #         axs[1].imshow(P)
    #         axs[1].axis('off')
    #         axs[1].set_title('Probability map')
    #         fig.tight_layout()
    #         plt.show()
    #         plt.close('all')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data', type=str, required=True, help='path to training dataset')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=4, help='Interval between each validation')
    parser.add_argument('--saved_model', default='./models/model_v0.pth', help="path to model to continue training")
    parser.add_argument('--lr', type=float, default=1., help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--grad_clip', type=float, default=2, help='gradient clipping value. default=2')
    parser.add_argument('--imgH', type=int, default=700, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=1000, help='the width of the input image')
    parser.add_argument('--k', type=int, default=50, help='the confidence k')

    opt = parser.parse_args()
    opt.categories = categories
    train(opt)