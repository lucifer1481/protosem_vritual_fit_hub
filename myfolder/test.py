import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm

from datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)

    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)
    parser.add_argument('--shuffle', action='store_true')

    parser.add_argument('--dataset_dir', type=str, default='./datasets/test')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('--dataset_list', type=str, default='test_pairs.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--save_dir', type=str, default='./results/')

    parser.add_argument('--display_freq', type=int, default=1)

    parser.add_argument('--seg_checkpoint', type=str, default='seg_final.pth')
    parser.add_argument('--gmm_checkpoint', type=str, default='gmm_final.pth')
    parser.add_argument('--alias_checkpoint', type=str, default='alias_final.pth')

    parser.add_argument('--semantic_nc', type=int, default=13)
    parser.add_argument('--init_type', type=str, default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02)

    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--num_upsampling_layers', type=str, default='most')

    return parser.parse_args()

def test(opt, seg, gmm, alias, device):
    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).to(device)

    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            print(f"[INFO] Batch {i+1} - {inputs['img_name']}, Cloth: {inputs['c_name']['unpaired']}")

            img_names = inputs['img_name']
            c_names = inputs['c_name']['unpaired']

            img_agnostic = inputs['img_agnostic'].to(device)
            parse_agnostic = inputs['parse_agnostic'].to(device)
            pose = inputs['pose'].to(device)
            c = inputs['cloth']['unpaired'].to(device)
            cm = inputs['cloth_mask']['unpaired'].to(device)

            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size()).to(device)), dim=1)

            parse_pred_down = seg(seg_input)
            parse_pred = gauss(up(parse_pred_down))
            parse_pred = parse_pred.argmax(dim=1)[:, None]

            parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float).to(device)
            parse_old.scatter_(1, parse_pred, 1.0)

            labels = {
                0: ['background', [0]],
                1: ['paste', [2, 4, 7, 8, 9, 10, 11]],
                2: ['upper', [3]],
                3: ['hair', [1]],
                4: ['left_arm', [5]],
                5: ['right_arm', [6]],
                6: ['noise', [12]]
            }

            parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float).to(device)
            for j in range(len(labels)):
                for label in labels[j][1]:
                    parse[:, j] += parse_old[:, label]

            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
            c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

            _, warped_grid = gmm(gmm_input, c_gmm)
            warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0.0] = 0.0
            parse_div = torch.cat((parse, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask

            output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

            unpaired_names = [f"{img_name.split('_')[0]}_{c_name}" for img_name, c_name in zip(img_names, c_names)]
            save_path = os.path.join(opt.save_dir, opt.name)
            save_images(output, unpaired_names, save_path)
            print(f"[✅] Saved to {save_path} => {unpaired_names}")

            if (i + 1) % opt.display_freq == 0:
                print("[🔁] Step:", i + 1)

def main():
    opt = get_opt()
    print("[⚙️] Running test with options:", opt)

    os.makedirs(os.path.join(opt.save_dir, opt.name), exist_ok=True)

    device = torch.device("cpu")

    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13

    load_checkpoint(seg, os.path.join(opt.checkpoint_dir, opt.seg_checkpoint))
    load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint))
    load_checkpoint(alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint))

    seg.to(device).eval()
    gmm.to(device).eval()
    alias.to(device).eval()

    test(opt, seg, gmm, alias, device)

if __name__ == '__main__':
    main()
