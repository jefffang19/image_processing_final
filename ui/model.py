import cv2
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np

class Dataset(BaseDataset):
    """
    Args:
        path (str): path to dataset
        task (str): which task to do
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    def __init__(
            self,
            path,
            task='CT', # CT, FT, MN
            augmentation=None, 
            preprocessing=None,
    ):
        self.patientId = sorted(os.listdir(path))
        self.num_imgs_of_each_pat = [] # count number of images for each patient
        self.ct = []
        self.ft = []
        self.mn = []
        self.t1 = []
        self.t2 = []

        # append file path
        for i in self.patientId:
          # append each class of images
          _cts = sorted(os.listdir('{}/{}/CT/'.format(path, i)), reverse=True)
          _cts = ['{}/{}/CT/{}'.format(path, i, j) for j in _cts]
          self.ct.append(_cts)
          _fts = sorted(os.listdir('{}/{}/FT/'.format(path, i)), reverse=True)
          _fts = ['{}/{}/FT/{}'.format(path, i, j) for j in _fts]
          self.ft.append(_fts)
          _mns = sorted(os.listdir('{}/{}/MN/'.format(path, i)), reverse=True)
          _mns = ['{}/{}/MN/{}'.format(path, i, j) for j in _mns]
          self.mn.append(_mns)
          _t1s = sorted(os.listdir('{}/{}/T1/'.format(path, i)), reverse=True)
          _t1s = ['{}/{}/T1/{}'.format(path, i, j) for j in _t1s]
          self.t1.append(_t1s)
          _t2s = sorted(os.listdir('{}/{}/T2/'.format(path, i)), reverse=True)
          _t2s = ['{}/{}/T2/{}'.format(path, i, j) for j in _t2s]
          self.t2.append(_t2s)

          # count number of images in each patient
          self.num_imgs_of_each_pat.append(len(_cts))
          

        # debug
        # print(self.ct)
        # print(self.ft)
        # print(self.mn)
        # print(self.t1)
        # print(self.t2)
        # print(self.num_imgs_of_each_pat)


        # check if do augmentation and preprocess
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):

        # count number i in which patient
        i_img = i
        pat = 0
        for n, max_value in enumerate(self.num_imgs_of_each_pat):
          if i_img < max_value:
            break
          else:
            i_img -= max_value
            pat += 1
          
          # if i still bigger than the last images number, raise error
          if n == len(self.num_imgs_of_each_pat) - 1:
            raise 'Dataset index out of range !'

        # debug
        # print(self.t1[pat][i_img])
        # print(self.t2[pat][i_img])
        # print(self.ct[pat][i_img])
        # print(self.ft[pat][i_img])
        # print(self.mn[pat][i_img])

        # now we know we want to get
        # patient id: pat
        # image #: i_img
        
        # read img
        image_t1 = cv2.imread(self.t1[pat][i_img], cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(image, (800, 640)) # image size of multiple of 2^network_depth ie. 32px
        image_t2 = cv2.imread(self.t2[pat][i_img], cv2.IMREAD_GRAYSCALE)
        mask_ct = cv2.imread(self.ct[pat][i_img], cv2.IMREAD_GRAYSCALE)
        mask_ft = cv2.imread(self.ft[pat][i_img], cv2.IMREAD_GRAYSCALE)
        mask_mn = cv2.imread(self.mn[pat][i_img], cv2.IMREAD_GRAYSCALE)
        # mask = cv2.resize(mask, (800, 640)) # iamge size of multiple of 2^network_depth ie. 32px

        # debug
        # print(image_t1.shape)
        # print(image_t2.shape)
        # print(mask_ct.shape)
        # print(mask_ft.shape)
        # print(mask_mn.shape)


        # stack the images and masks
        image = np.stack((image_t1, image_t2), axis=2)
        mask = np.stack((mask_ct, mask_ft, mask_mn), axis=2)
        
        
        # apply augmentations
        if self.augmentation:
            # 3 channel needed to use augmentation
            # so we pad a channel for image
            _image = np.stack((image[...,0], image[...,1], np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)), axis=2)
            sample = self.augmentation(image=_image, mask=mask)
            _image, mask = sample['image'], sample['mask']

            # assign the first two channel back
            image = np.stack((_image[...,0], _image[...,1]), axis=2)
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # normalize needs to be done or training evalutation will be wrong
        # normalize needs to do after augmentations, or will cause error
        image = image/255 # normalize image
        mask = mask/255 # normalize image
            
        return image, mask
        
    def __len__(self):
        return np.sum(self.num_imgs_of_each_pat)

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def inference():
    import segmentation_models_pytorch as smp
    import torch

    DEVICE = 'cpu'
    if torch.cuda.is_available():
        print('using gpu')
        DEVICE = "cuda:0"
    else:
        print('using cpu')

    load_model_path = "unetpp_best_model.pth"

    model = smp.UnetPlusPlus(encoder_weights=None, in_channels=2, classes=3, activation='sigmoid')
    model.load_state_dict(torch.load(load_model_path, map_location=DEVICE))
    if torch.cuda.is_available():
        model.cuda()

    testset_path = "test"

    testset = Dataset(
        testset_path,
    )

    x, y = testset[1]

    # convert numpy to tensor
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.ToTensor()])

    # pad to three dimension so we can use transform
    _x = np.stack((x[...,0], x[...,1], np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)), axis=2)

    # convert to tensor
    image_tensor = transform(_x)

    # take the first two images
    image_tensor = image_tensor[0:2]

    # make batch size == 1
    image_tensor = image_tensor.unsqueeze(0)

    # cast to float
    image_tensor = image_tensor.float()

    # fit model
    model.eval()

    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    pred = model(image_tensor)

    # convert tensor to numpy
    # convert tensor to numpy
    np_pred = pred.detach().cpu().numpy().squeeze()

    # switch axis
    np_pred = np.transpose(np_pred, (1, 2, 0))


    # evalute and save predict results
    import segmentation_models_pytorch.utils.metrics as metrics

    xs = []
    ys = []
    preds = []
    cts = []
    fts = []
    mns = []
    stacks = []

    for i in range(len(testset)):
        x, y = testset[i]
        xs.append(x)
        ys.append(y)
        # pad to three dimension so we can use transform
        _x = np.stack((x[...,0], x[...,1], np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)), axis=2)
        # convert to tensor
        image_tensor = transform(_x)
        # take the first two images
        image_tensor = image_tensor[0:2]
        # make batch size == 1
        image_tensor = image_tensor.unsqueeze(0)
        # cast to float
        image_tensor = image_tensor.float()
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        # fit model
        model.eval()
        pred = model(image_tensor)
        # convert tensor to numpy
        np_pred = pred.detach().cpu().numpy().squeeze()
        # switch axis
        np_pred = np.transpose(np_pred, (1, 2, 0))
        # transfrom groundtruth to tensor
        _y = transform(y)
        _y = _y.unsqueeze(0)
        if torch.cuda.is_available():
            _y = _y.cuda()
        
        # print each image
        print('{}'.format(i))
        # print('ct iou')
        # print(metrics.IoU()(pred[:, 0, ...], _y[:, 0, ...]))
        # print('ct dice', end='  ')
        # print(metrics.Fscore()(pred[:, 0, ...], _y[:, 0, ...]).item() )
        # print('ft iou')
        # print(metrics.IoU()(pred[:, 1, ...], _y[:, 1, ...]))
        # print('ft dice', end='  ')
        # print(metrics.Fscore()(pred[:, 1, ...], _y[:, 1, ...]).item() )
        # print('mn iou')
        # print(metrics.IoU()(pred[:, 2, ...], _y[:, 2, ...]))
        # print('mn dice', end='  ')
        # print(metrics.Fscore()(pred[:, 2, ...], _y[:, 2, ...]).item() )
        # print('\n')

        # save dice
        cts.append(metrics.Fscore()(pred[:, 0, ...], _y[:, 0, ...]).item())
        fts.append(metrics.Fscore()(pred[:, 1, ...], _y[:, 1, ...]).item())
        mns.append(metrics.Fscore()(pred[:, 2, ...], _y[:, 2, ...]).item())

        # save pred results
        preds.append(np_pred)

        # stack image
        ct = (np_pred[..., 0] * 255).astype(np.uint8)
        ft = (np_pred[..., 1] * 255).astype(np.uint8)
        mn = (np_pred[..., 2] * 255).astype(np.uint8)

        blurred = cv2.GaussianBlur(ct, (11, 11), 0)
        binaryIMG_ct = cv2.Canny(blurred, 20, 160)

        blurred = cv2.GaussianBlur(ft, (11, 11), 0)
        binaryIMG_ft = cv2.Canny(blurred, 20, 160)

        blurred = cv2.GaussianBlur(mn, (11, 11), 0)
        binaryIMG_mn = cv2.Canny(blurred, 20, 160)

        # ct red, mn yellow, ft blue
        stacked = np.stack((binaryIMG_ct + binaryIMG_mn, binaryIMG_mn, binaryIMG_ft), axis=2)

        clone_t1 = (x[...,0]*255).astype(np.uint8).copy()
        clone_t1 = np.stack((clone_t1, clone_t1, clone_t1), axis=2)
        draw_t1 = (clone_t1*0.5 + stacked*0.5).astype(np.uint8)

        stacks.append(draw_t1)

    print('count len')
    print(len(xs))
    print(len(ys))
    print(len(preds))
    print(len(cts))
    print(len(fts))
    print(len(mns))
    print(len(stacks))

    print(np.mean(np.array(cts)))
    print(np.mean(np.array(fts)))
    print(np.mean(np.array(mns)))

    # visual input data and mask and predict result
    # visualize(
    #     t1=xs[0][...,0],
    #     t2=xs[0][...,1],
    #     ct=ys[0][...,0],
    #     ft=ys[0][...,1],
    #     mn=ys[0][...,2],
    #     p_ct=preds[0][..., 0],
    #     p_ft=preds[0][..., 1],
    #     p_mn=preds[0][..., 2],
    #     stack=stacks[0]
    # )

    return xs, ys, preds, cts, fts, mns, stacks
