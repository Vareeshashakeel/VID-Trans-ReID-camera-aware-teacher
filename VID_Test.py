import numpy as np
import torch
import torch.nn.functional as F


def _to_tensor(x, device=None, dtype=torch.long):
    """
    Convert tensor / list / tuple / numpy / scalar to torch tensor.
    """
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype) if device is not None else x.to(dtype=dtype)
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        return t.to(device=device, dtype=dtype) if device is not None else t.to(dtype=dtype)
    if isinstance(x, (list, tuple)):
        t = torch.tensor(x, dtype=dtype)
        return t.to(device=device) if device is not None else t
    t = torch.tensor([x], dtype=dtype)
    return t.to(device=device) if device is not None else t


def _to_numpy(x):
    """
    Convert tensor / list / tuple / scalar to numpy array.
    """
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    return np.asarray([x])


def _flatten_meta(x):
    """
    Make pid/camid arrays 1D.
    """
    arr = _to_numpy(x)
    return arr.reshape(-1)


def _unpack_eval_batch(batch):
    """
    Robustly unpack validation/test loader batches.

    Many old ReID loaders return more than 3 values.
    We only need:
        imgs, pids, camids
    """
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Unsupported batch type in evaluation: {type(batch)}")

    if len(batch) < 3:
        raise ValueError(f"Evaluation batch has too few elements: got {len(batch)}")

    imgs = batch[0]
    pids = batch[1]
    camids = batch[2]
    return imgs, pids, camids


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.0

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
        num_valid_q += 1.0

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = np.asarray([x / (i + 1.0) for i, x in enumerate(tmp_cmc)]) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


@torch.no_grad()
def _extract_sequence_feature(model, imgs, pids, camids, device):
    imgs = imgs.to(device)
    pids_t = _to_tensor(pids, device=device, dtype=torch.long).view(-1)
    camids_t = _to_tensor(camids, device=device, dtype=torch.long).view(-1)

    outputs = model(imgs, pids_t, cam_label=camids_t)

    # old repos may return tuple/list
    if isinstance(outputs, (list, tuple)):
        # usually (score, feat, ...)
        if len(outputs) >= 2:
            feat = outputs[1]
        else:
            feat = outputs[0]
    else:
        feat = outputs

    # some repos may wrap feat again
    if isinstance(feat, (list, tuple)):
        feat = feat[0]

    # if temporal dimension still exists, average it
    if feat.dim() == 3:
        feat = feat.mean(dim=1)

    feat = F.normalize(feat, p=2, dim=1)
    return feat.cpu()


@torch.no_grad()
def test(model, q_loader, g_loader):
    device = next(model.parameters()).device
    model.eval()

    qf, q_pids, q_camids = [], [], []
    for batch in q_loader:
        imgs, pids, camids = _unpack_eval_batch(batch)
        feat = _extract_sequence_feature(model, imgs, pids, camids, device)
        qf.append(feat)
        q_pids.extend(_flatten_meta(pids).tolist())
        q_camids.extend(_flatten_meta(camids).tolist())

    qf = torch.cat(qf, dim=0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    print(f'Extracted features for query set, obtained {qf.shape[0]}-by-{qf.shape[1]} matrix')

    gf, g_pids, g_camids = [], [], []
    for batch in g_loader:
        imgs, pids, camids = _unpack_eval_batch(batch)
        feat = _extract_sequence_feature(model, imgs, pids, camids, device)
        gf.append(feat)
        g_pids.extend(_flatten_meta(pids).tolist())
        g_camids.extend(_flatten_meta(camids).tolist())

    gf = torch.cat(gf, dim=0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print(f'Extracted features for gallery set, obtained {gf.shape[0]}-by-{gf.shape[1]} matrix')

    print('Computing distance matrix')
    distmat = torch.cdist(qf, gf, p=2).cpu().numpy()

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print('Results ----------')
    print(f'mAP: {mAP * 100:.1f}%')
    print('CMC curve')
    if len(cmc) > 0:
        print(f'Rank-1  : {cmc[0] * 100:.1f}%')
    if len(cmc) > 4:
        print(f'Rank-5  : {cmc[4] * 100:.1f}%')
    if len(cmc) > 9:
        print(f'Rank-10 : {cmc[9] * 100:.1f}%')
    if len(cmc) > 19:
        print(f'Rank-20 : {cmc[19] * 100:.1f}%')

    return cmc[0], mAP
