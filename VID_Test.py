
import numpy as np
import torch


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc, all_AP, num_valid_q = [], [], 0.
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = np.asarray([x / (i + 1.) for i, x in enumerate(tmp_cmc)]) * orig_cmc
        all_AP.append(tmp_cmc.sum() / num_rel)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'
    all_cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP


def _extract_sequence_feature(model, imgs, pids, camids, device, pool='avg'):
    imgs = imgs.to(device)
    camids = camids.to(device)
    features = model(imgs, pids, cam_label=camids)
    bsz = imgs.size(0)
    features = features.view(bsz, -1)
    if pool == 'avg':
        return torch.mean(features, 0).cpu()
    return torch.max(features, 0)[0].cpu()


def test(model, queryloader, galleryloader, pool='avg'):
    device = next(model.parameters()).device
    model.eval()
    qf, q_pids, q_camids = [], [], []
    with torch.no_grad():
        for imgs, pids, camids, _ in queryloader:
            qf.append(_extract_sequence_feature(model, imgs, pids, camids, device, pool='avg'))
            q_pids.append(pids)
            q_camids.extend(camids.tolist())
        qf = torch.stack(qf)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print(f'Extracted features for query set, obtained {qf.size(0)}-by-{qf.size(1)} matrix')

        gf, g_pids, g_camids = [], [], []
        for imgs, pids, camids, _ in galleryloader:
            gf.append(_extract_sequence_feature(model, imgs, pids, camids, device, pool=pool))
            g_pids.append(pids)
            g_camids.extend(camids.tolist())
        gf = torch.stack(gf)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print(f'Extracted features for gallery set, obtained {gf.size(0)}-by-{gf.size(1)} matrix')
        print('Computing distance matrix')

        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) +                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
        distmat = distmat.numpy()

        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
        print('Results ----------')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        print('Rank-1  : {:.1%}'.format(cmc[0]))
        print('Rank-5  : {:.1%}'.format(cmc[4]))
        print('Rank-10 : {:.1%}'.format(cmc[9]))
        print('Rank-20 : {:.1%}'.format(cmc[19]))
        return cmc[0], mAP
