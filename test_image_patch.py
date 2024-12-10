import argparse
import torch
import os
import numpy as np
import datasets.crowd as crowd
import FSGformer
import torch.nn.functional as F
from sklearn.metrics import r2_score

def tensor_divideByfactor(img_tensor, factor=32):
    _, _, h, w = img_tensor.size()
    h, w = int(h//factor*factor), int(w//factor*factor)
    img_tensor = F.interpolate(img_tensor, (h, w), mode='bilinear', align_corners=True)

    return img_tensor
def cal_new_tensor(img_tensor, min_size=256):
    _, _, h, w = img_tensor.size()
    if min(h, w) < min_size:
        ratio_h, ratio_w = min_size / h, min_size / w
        if ratio_h >= ratio_w:
            img_tensor = F.interpolate(img_tensor, (min_size, int(min_size / h * w)), mode='bilinear', align_corners=True)
        else:
            img_tensor = F.interpolate(img_tensor, (int(min_size / w * h), min_size), mode='bilinear', align_corners=True)
    return img_tensor

parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--batch_size', type=int, default=1,
                        help='train batch size')
parser.add_argument('--crop_size', type=int, default=256,
                    help='the crop size of the train image')
parser.add_argument('--model_path',default=r'C:\Users\ming\Desktop\code\FSGformer\best_model_mae-2.35_epoch-34.pth', type=str,
                    help='saved model path')
parser.add_argument('--data_path', type=str,default=r'C:\Users\ming\Desktop\code\FSGformer\CCD',
                    help='dataset path')
parser.add_argument('--dataset', type=str, default='ccd',
                    help='dataset name: dgcd,ccd')
parser.add_argument('--pred-density-map-path', type=str, default='inference_results',
                    help='save predicted density maps when pred-density-map-path is not empty.')

def test(args, isSave = True):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    model_path = args.model_path
    crop_size = args.crop_size
    data_path = args.data_path
    if args.dataset.lower() == 'dgcd' or args.dataset.lower() == 'ccd':
        dataset = crowd.Crowd_fish(os.path.join(data_path, 'test_data'), crop_size, 8, method='val')
    else:
        raise NotImplementedError
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,
                                             num_workers=1, pin_memory=True)

    model = FSGformer.alt_gvt_large(pretrained=True)
    model.to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()
    image_errs = []
    image_naes = []
    gt_counts = []  
    predicted_counts = [] 
    result = []
    for inputs, count, name in dataloader:
        with torch.no_grad():
            inputs = inputs.to(device)
            crop_imgs, crop_masks = [], []
            b, c, h, w = inputs.size()
            rh, rw = args.crop_size, args.crop_size
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                    crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                    mask = torch.zeros([b, 1, h, w]).to(device)
                    mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                    crop_masks.append(mask)
            crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))

            crop_preds = []
            nz, bz = crop_imgs.size(0), args.batch_size
            for i in range(0, nz, bz):
                gs, gt = i, min(nz, i + bz)
                crop_pred, _ = model(crop_imgs[gs:gt])

                _, _, h1, w1 = crop_pred.size()
                
                crop_pred = F.interpolate(crop_pred, size=(h1 * 8, w1 * 8), mode='bilinear', align_corners=True) / 64

                crop_preds.append(crop_pred)
            crop_preds = torch.cat(crop_preds, dim=0)

            idx = 0
            pred_map = torch.zeros([b, 1, h, w]).to(device)
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                    pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                    idx += 1
            mask = crop_masks.sum(dim=0).unsqueeze(0)
            outputs = pred_map / mask

            img_err = count[0].item() - torch.sum(outputs).item()
            img_nae = (count[0].item() - torch.sum(outputs).item()) / count[0].item()
            image_naes.append(img_nae)
            print("Img name: ", name, "Error: ", img_err, "GT count: ", count[0].item(), "Model out: ", torch.sum(outputs).item())
            image_errs.append(img_err)
            gt_counts.append(count[0].item())  
            predicted_counts.append(torch.sum(outputs).item())  
            result.append([name, count[0].item(), torch.sum(outputs).item(), img_err])




    image_errs = np.array(image_errs)
    image_naes = np.array(image_naes)
    gt_counts = np.array(gt_counts)
    predicted_counts = np.array(predicted_counts)
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    nae = np.mean(np.abs(image_naes))

    R = np.corrcoef(gt_counts, predicted_counts)[0, 1]


    R2 = r2_score(gt_counts, predicted_counts)


    mean_gt_counts = np.mean(gt_counts)
    nse = 1 - np.sum((gt_counts - predicted_counts) ** 2) / np.sum((gt_counts - mean_gt_counts) ** 2)


    print('{}: mae {}, mse {}, nae {}, R {}, R2 {}, NSE {}\n'.format(model_path, mae, mse, nae, R, R2, nse))

    # if isSave:
    #     with open("DGCD.txt","w") as f:
    #         for i in range(len(result)):
    #             f.write(str(result[i]).replace('[','').replace(']','').replace(',', ' ')+"\n")
    #         f.close()

if __name__ == '__main__':
    args = parser.parse_args()
    test(args, isSave= True)


