
from __future__ import print_function, absolute_import

from collections import defaultdict
from scipy.io import loadmat
import os.path as osp
import numpy as np


class Mars(object):
    """MARS dataset."""

    def __init__(self, root='MARS', min_seq_len=0):
        self.root = root
        self.train_name_path = osp.join(root, 'info/train_name.txt')
        self.test_name_path = osp.join(root, 'info/test_name.txt')
        self.track_train_info_path = osp.join(root, 'info/tracks_train_info.mat')
        self.track_test_info_path = osp.join(root, 'info/tracks_test_info.mat')
        self.query_IDX_path = osp.join(root, 'info/query_IDX.mat')
        self._check_before_run()

        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info']
        track_test = loadmat(self.track_test_info_path)['track_test_info']
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze()
        query_IDX -= 1
        track_query = track_test[query_IDX, :]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX, :]

        train, num_train_tracklets, num_train_pids, num_train_imgs = self._process_data(
            train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)
        query, num_query_tracklets, num_query_pids, num_query_imgs = self._process_data(
            test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = self._process_data(
            test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_train_cams = 6
        self.num_query_cams = 6
        self.num_gallery_cams = 6
        self.num_train_vids = num_train_tracklets
        self.num_query_vids = num_query_tracklets
        self.num_gallery_vids = num_gallery_tracklets

    def _check_before_run(self):
        for path in [self.root, self.train_name_path, self.test_name_path, self.track_train_info_path,
                     self.track_test_info_path, self.query_IDX_path]:
            if not osp.exists(path):
                raise RuntimeError("'{}' is not available".format(path))

    def _get_names(self, fpath):
        with open(fpath, 'r') as f:
            return [line.rstrip() for line in f]

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 2].tolist()))
        num_pids = len(pid_list)
        pid2label = {pid: label for label, pid in enumerate(pid_list)} if relabel else {pid: int(pid) for pid in pid_list}
        tracklets, num_imgs_per_tracklet = [], []
        for tracklet_idx in range(num_tracklets):
            start_index, end_index, pid, camid = meta_data[tracklet_idx, ...]
            if pid == -1:
                continue
            assert 1 <= camid <= 6
            pid = pid2label[pid]
            camid -= 1
            img_names = names[start_index - 1:end_index]
            assert len(set([img_name[:4] for img_name in img_names])) == 1
            assert len(set([img_name[5] for img_name in img_names])) == 1
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                tracklets.append((tuple(img_paths), pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))
        return tracklets, len(tracklets), num_pids, num_imgs_per_tracklet
