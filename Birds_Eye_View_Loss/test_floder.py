import numpy as np
import torch
from PIL import Image
import cv2
import imageio
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_weightmap(M, weightmap_zeros, beta0, beta1, beta2, beta3, images, i):
    resize=256
    no_ortho = False
    M = M.data.cpu().numpy()[0]
    x = np.zeros(3)

    # line_class = line_class[0].cpu().numpy()
    # left_lane = True if line_class[0] != 0 else False
    # right_lane = True if line_class[3] != 0 else False
    left_lane = True
    right_lane = True

    wm0_zeros = weightmap_zeros.data.cpu()[0, 0].numpy()
    wm1_zeros = weightmap_zeros.data.cpu()[0, 1].numpy()

    # im = images.permute(0, 2, 3, 1).data.cpu().numpy()[0] #交换维度
    im = images.permute(0, 2, 3, 1).data.cpu().numpy()[0] #交换维度
    im_orig = np.copy(im)
    im_orig = im_orig.copy() #change
    # gt_orig = gt.permute(0, 2, 3, 1).data.cpu().numpy()[0, :, :, 0]
    # im_orig = draw_homography_points(im_orig, x, resize)

    im, M_scaledup = test_projective_transform(im, resize, M)

    # im, _ = draw_fitted_line(im, gt_params_rhs[0], resize, (0, 255, 0))
    # im, _ = draw_fitted_line(im, gt_params_lhs[0], resize, (0, 255, 0))
    im, lane0 = draw_fitted_line(im, beta0[0], resize, (255, 0, 0))
    im, lane1 = draw_fitted_line(im, beta1[0], resize, (0, 0, 255))
    if beta2 is not None:
        # im, _ = draw_fitted_line(im, gt_params_llhs[0], resize, (0, 255, 0))
        # im, _ = draw_fitted_line(im, gt_params_rrhs[0], resize, (0, 255, 0))
        if left_lane:
            im, lane2 = draw_fitted_line(im, beta2[0], resize, (255, 255, 0))
        if right_lane:
            im, lane3 = draw_fitted_line(im, beta3[0], resize, (255, 128, 0))


    if not no_ortho:
        im_inverse = cv2.warpPerspective(im, np.linalg.inv(M_scaledup), (2*resize, resize))
    else:
        im_inverse = im_orig

    im_orig = np.clip(im_orig, 0, 1)
    im_inverse = np.clip(im_inverse, 0, 1)
    im = np.clip(im, 0, 1)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    #ax2 = fig.add_subplot(422)
    ax3 = fig.add_subplot(122)
    #ax4 = fig.add_subplot(424)
    #ax5 = fig.add_subplot(425)
    #ax6 = fig.add_subplot(426)
    #ax7 = fig.add_subplot(427)
    ax1.imshow(im_orig)
    #ax2.imshow(wm0_zeros)
    ax3.imshow(im_inverse)
    #ax4.imshow(wm1_zeros)
    #ax5.imshow(wm0_zeros/np.max(wm0_zeros)+wm1_zeros/np.max(wm1_zeros))
    # ax6.imshow(im_orig)
    # ax7.imshow(gt_orig)
    # fig.savefig(save_path + '/example/{}/weight_idx-{}_batch-{}'.format(train_or_val, idx, i))
    #fig.savefig('output')
    fig.savefig('./result/'+str(i))
    plt.clf()
    plt.close(fig)

def test_projective_transform(input, resize, M):
    # test grid using built in F.grid_sampler method.
    M_scaledup = np.array([[M[0,0],M[0,1]*2,M[0,2]*(2*resize-1)],[0,M[1,1],M[1,2]*(resize-1)],[0,M[2,1]/(resize-1),M[2,2]]])
    inp = cv2.warpPerspective(np.asarray(input), M_scaledup, (2*resize,resize))
    return inp, M_scaledup


def draw_fitted_line(img, params, resize, color=(255,0,0)):
    params = params.data.cpu().tolist()
    y_stop = 0.7
    y_prime = np.linspace(0, y_stop, 20)
    params = [0] * (4 - len(params)) + params
    d, a, b, c = [*params]
    x_pred = d*(y_prime**3) + a*(y_prime)**2 + b*(y_prime) + c
    x_pred = x_pred*(2*resize-1)
    y_prime = (1-y_prime)*(resize-1)
    lane = [(xcord, ycord) for (xcord, ycord) in zip(x_pred, y_prime)] 
    img = cv2.polylines(img, [np.int32(lane)], isClosed = False, color = color,thickness = 1)
    return img, lane
    
def draw_homography_points(img, x, resize=256, color=(255,0,0)):
    y_start1 = (0.3+x[2])*(resize-1)
    y_start = 0.3*(resize-1)
    y_stop = resize-1
    src = np.float32([[0.45*(2*resize-1),y_start],[0.55*(2*resize-1), y_start],[0.1*(2*resize-1),y_stop],[0.9*(2*resize-1), y_stop]])
    dst = np.float32([[(0.45+x[0])*(2*resize-1), y_start1],[(0.55+x[1])*(2*resize-1), y_start1],[(0.45+x[0])*(2*resize-1), y_stop],[(0.55+x[1])*(2*resize-1),y_stop]])
    dst_ideal = np.float32([[0.45*(2*resize-1), y_start],[0.55*(2*resize-1), y_start],[0.45*(2*resize-1), y_stop],[0.55*(2*resize-1),y_stop]])
    [cv2.circle(np.asarray(img), tuple(idx), radius=5, thickness=-1, color=(255,0,0)) for idx in src]
    [cv2.circle(np.asarray(img), tuple(idx), radius=5, thickness=-1, color=(0,255,0)) for idx in dst_ideal]
    [cv2.circle(np.asarray(img), tuple(idx), radius=5, thickness=-1, color=(0,0,255)) for idx in dst]
    return img

def main():
    from torchvision import transforms
    totensor = transforms.ToTensor()
    from Networks.LSQ_layer import Net
    from Networks.utils import define_args
    global args
    parser = define_args()
    args = parser.parse_known_args()[0]
    print(args.resize)
    
    model = Net(args)
    checkpoint = torch.load("Saved/Mod_erfnet_opt_adam_loss_area_lr_0.0001_batch_8_end2end_True_lanes_2_resize_256_pretrainFalse_clasFalse/checkpoint_model_epoch_349.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])

    #device = torch.device('cuda')
    #model = Net(args)
    #model = torch.load("model_best_epoch_319.pth.tar")
    model = model.cuda()
    
    root_path = './images'
    list = os.listdir(root_path)  # 列出文件夹下所有的目录与文件
    i = 1
    for file in list:
        dir_file_path = os.path.join(root_path,file)
        image = imageio.imread(dir_file_path)


        #image = imageio.imread("2.jpg")
        image = cv2.resize(image,(args.resize*2,args.resize),interpolation=Image.BILINEAR)
        image = totensor(image).float()
        #input_image = image
        #input_image = torch.unsqueeze(input_image, dim=0)
        #image = input_image
        #print("input_image size: ",input_image.size())

        image = image[None,:,:,:]
        input_image = image

        image = image.cuda()
        print(type(image))

        print(image.shape)

        with torch.no_grad():
            beta0, beta1, beta2, beta3, weightmap_zeros, M, output_net, outputs_line, outputs_hozizon = model(image, True)
            # beta0, beta1, beta2, beta3, weightmap_zeros, M, output_net, outputs_line, outputs_hozizon = model.forward(image, True)
            print("outputs_line: ",outputs_line)
            print("beta0: ",beta0)
            print("outputs_hozizon: ", outputs_hozizon)
            print("output_net: ", output_net)
            print(output_net.shape)
            save_weightmap(M, weightmap_zeros, beta0, beta1, beta2, beta3, input_image, i)
            i = i+1

if __name__ == '__main__':
    main()
