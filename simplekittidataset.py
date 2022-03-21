import torch
import numpy as np
import os
import utils.geom

class SimpleKittiDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle=True, S=1, dset='t', seq_name=None, mod='am', kitti_data_seqlen=1, sort=False, root_dir='/projects/katefgroup/datasets/kitti/processed/npzs', use_complete=False, return_valid=False):

        kitti_data_mod = 'an'
        kitti_data_incr = 1

        trainset = "t%ss%si%s%s" % (kitti_data_mod, kitti_data_seqlen, kitti_data_incr, dset)

        trainset_format = "ktrack"
        trainset_consec = False
        dataset_location = "%s" % root_dir

        dataset_path = '%s/%s.txt' % (dataset_location, trainset)

        print('dataset_path = %s' % dataset_path)
        with open(dataset_path) as f:
            content = f.readlines()
        dataset_location = dataset_path.split('/')[:-1]
        dataset_location = '/'.join(dataset_location)
        print('dataset_loc = %s' % dataset_location)

        records = [dataset_location + '/' + line.strip() for line in content]
        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_path))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))

        if seq_name is not 'any' and seq_name is not None:
            records = [fn for fn in records if (seq_name in fn)]
            print('trimmed to %d records that match %s' % (len(records), seq_name))

        if sort:
            records.sort()
 
        self.records = records
        self.shuffle = shuffle
        self.S = S
        self.use_complete = use_complete
        self.return_valid = return_valid

    def __getitem__(self, index):
        filename = self.records[index]
        d = np.load(filename, allow_pickle=True)
        d = dict(d)

        rgb_camXs = d['rgb_camXs']
        xyz_veloXs = d['xyz_veloXs']
        cam_T_velos = d['cam_T_velos']
        pix_T_rects = d['pix_T_rects']
        rect_T_cams = d['rect_T_cams']
        boxlists = d['boxlists']
        tidlists = d['tidlists']
        clslists = d['clslists']
        scorelists = d['scorelists']

        if self.return_valid and np.sum(scorelists[0])==0:
            return self.__getitem__(np.random.choice(self.__len__()))
        # print('proceeding with np.sum(scorelists)', np.sum(scorelists[0]))
        
        S_ = pix_T_rects.shape[0]
        if self.shuffle:
            inds = np.random.choice(S_, size=self.S, replace=False)
        else:
            inds = list(range(self.S))
~
        rgb_camXs = rgb_camXs[inds]
        xyz_veloXs = xyz_veloXs[inds]
        cam_T_velos = cam_T_velos[inds]
        pix_T_rects = pix_T_rects[inds]
        rect_T_cams = rect_T_cams[inds]
        boxlists = boxlists[inds]
        tidlists = tidlists[inds]
        clslists = clslists[inds]
        scorelists = scorelists[inds]

        rgb_camXs = torch.from_numpy(rgb_camXs) # S, H, W, 3
        xyz_veloXs = torch.from_numpy(xyz_veloXs) # S, N, 3
        cam_T_velos = torch.from_numpy(cam_T_velos)
        pix_T_rects = torch.from_numpy(pix_T_rects)
        rect_T_cams = torch.from_numpy(rect_T_cams)
        boxlists = torch.from_numpy(boxlists)
        tidlists = torch.from_numpy(tidlists)
        clslists = torch.from_numpy(clslists)
        scorelists = torch.from_numpy(scorelists)

        xyz_camXs = utils.geom.apply_4x4(cam_T_velos, xyz_veloXs)
        xyz_rectXs = utils.geom.apply_4x4(rect_T_cams, xyz_camXs)
        
        # move channels in
        rgb_camXs = rgb_camXs.permute(0, 3, 1, 2)

        lrtlists_cam = utils.geom.convert_boxlist_to_lrtlist(boxlists)

        samp = {}
        if self.S==1:
            samp['rgb_cam'] = rgb_camXs[0]
            samp['xyz_cam'] = xyz_rectXs[0]
            samp['pix_T_cam'] = pix_T_rects[0]
            samp['lrtlist_cam'] = lrtlists_cam[0]
            samp['scorelist'] = scorelists[0]
            samp['tidlist'] = tidlists[0]
        else:
            samp['rgb_cams'] = rgb_camXs
            samp['xyz_cams'] = xyz_rectXs
            samp['pix_T_cams'] = pix_T_rects
            samp['lrtlists_cam'] = lrtlists_cam
            samp['scorelists'] = scorelists
            samp['tidlists'] = tidlists
            
        return samp

    def __len__(self):
        return len(self.records)
        # return 10

