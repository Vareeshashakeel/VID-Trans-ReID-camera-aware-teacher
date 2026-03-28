
import argparse
import os
import random
import time

import numpy as np
import torch
from torch.cuda import amp

try:
    from torch_ema import ExponentialMovingAverage
except Exception:
    ExponentialMovingAverage = None

from Dataloader import dataloader
from Loss_fun import make_loss
from VID_Test import test
from VID_Trans_model import VID_Trans
from utility import AverageMeter, optimizer, scheduler


def set_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera-aware VID-Trans-ReID teacher training')
    parser.add_argument('--Dataset_name', required=True, type=str)
    parser.add_argument('--dataset_root', required=True, type=str, help='Root folder of the selected dataset')
    parser.add_argument('--model_path', required=True, type=str, help='ViT pretrained weight path')
    parser.add_argument('--output_dir', default='./output_camera_aware_teacher', type=str)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--eval_every', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seq_len', default=4, type=int)
    parser.add_argument('--num_instances', default=4, type=int)
    parser.add_argument('--center_w', default=0.0005, type=float)
    parser.add_argument('--attn_w', default=1.0, type=float)
    parser.add_argument('--seed', default=1234, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    train_loader, _, num_classes, camera_num, view_num, q_val_loader, g_val_loader = dataloader(
        args.Dataset_name,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
        num_instances=args.num_instances,
    )

    model = VID_Trans(num_classes=num_classes, camera_num=camera_num, pretrainpath=args.model_path)
    loss_fun, center_criterion = make_loss(num_classes=num_classes)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    center_criterion = center_criterion.to(device)

    optimizer_main = optimizer(model)
    lr_scheduler = scheduler(optimizer_main)
    scaler = amp.GradScaler(enabled=(device == 'cuda'))
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5) if args.center_w > 0 else None

    loss_meter = AverageMeter()
    id_meter = AverageMeter()
    acc_meter = AverageMeter()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995) if ExponentialMovingAverage is not None else None
    best_rank1 = 0.0

    print(f'[INFO] Device: {device}')
    print(f'[INFO] Dataset root: {args.dataset_root}')
    print(f'[INFO] Loss weights -> center: {args.center_w}, attn: {args.attn_w}')

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_meter.reset(); id_meter.reset(); acc_meter.reset()
        lr_scheduler.step(epoch)
        model.train()

        for iteration, (img, pid, target_cam, labels2) in enumerate(train_loader, start=1):
            optimizer_main.zero_grad()
            if optimizer_center is not None:
                optimizer_center.zero_grad()

            img = img.to(device)
            pid = pid.to(device)
            target_cam = target_cam.to(device)
            labels2 = labels2.to(device)
            seq_cam = target_cam[:, 0].contiguous() if target_cam.ndim > 1 else target_cam.contiguous()

            with amp.autocast(enabled=(device == 'cuda')):
                score, feat, a_vals = model(img, pid, cam_label=seq_cam)
                attn_loss = (a_vals * labels2).sum(1).mean()
                loss_id, center = loss_fun(score, feat, pid, seq_cam)
                loss = loss_id + args.center_w * center + args.attn_w * attn_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer_main)
            if optimizer_center is not None:
                for param in center_criterion.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1.0 / args.center_w)
                scaler.step(optimizer_center)
            scaler.update()

            if ema is not None:
                ema.update()

            acc = (score[0].max(1)[1] == pid).float().mean() if isinstance(score, list) else (score.max(1)[1] == pid).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            id_meter.update(loss_id.item(), img.shape[0])
            acc_meter.update(acc.item(), 1)

            if device == 'cuda':
                torch.cuda.synchronize()
            if iteration % 50 == 0:
                print('Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, ID+Tri: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}'.format(
                    epoch, iteration, len(train_loader), loss_meter.avg, id_meter.avg, acc_meter.avg, lr_scheduler._get_lr(epoch)[0]))

        print('Epoch {} finished in {:.1f}s'.format(epoch, time.time() - start_time))

        if epoch % args.eval_every == 0:
            model.eval()
            rank1, mAP = test(model, q_val_loader, g_val_loader)
            print('CMC: %.4f, mAP : %.4f' % (rank1, mAP))
            latest_path = os.path.join(args.output_dir, f'{args.Dataset_name}_camera_aware_latest.pth')
            torch.save(model.state_dict(), latest_path)
            if best_rank1 < rank1:
                best_rank1 = rank1
                best_path = os.path.join(args.output_dir, f'{args.Dataset_name}_camera_aware_best.pth')
                torch.save(model.state_dict(), best_path)
                print(f'[OK] Saved best checkpoint: {best_path}')
