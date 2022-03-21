import torch
import numpy as np
import os
import glob

class PseudoKittiDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle=True, input_name='asdf', load_seg=False, load_i=False):

        dataset_location = './tcr_pseudo'
        records = glob.glob('%s/*_%s_*.npz' % (dataset_location, input_name))
        print('records', records)

        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_location))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))

        self.records = records
        self.shuffle = shuffle

        self.load_i = load_i
        self.load_seg = load_seg

    def __getitem__(self, index):
        # print('getting index', index)
        filename = self.records[index]
        d = np.load(filename, allow_pickle=True)
        d = dict(d)

        rgb_cam = d['rgb_cam']
        xyz_cam = d['xyz_cam']
        pix_T_cam = d['pix_T_cam']
        lrtlist_cam = d['lrtlist_cam']
        # rylist = d['rylist']
        # scorelist = d['scorelist']

        rgb_cam = torch.from_numpy(rgb_cam) # 3, H, W
        xyz_cam_ = torch.from_numpy(xyz_cam) # V_, 3
        V_ = xyz_cam_.shape[0]
        V = 130000
        xyz_cam = torch.zeros((V, 3), dtype=torch.float32)
        xyz_cam[:V_] = xyz_cam_
        
        pix_T_cam = torch.from_numpy(pix_T_cam) # 4, 4
        lrtlist_cam_ = torch.from_numpy(lrtlist_cam) # N_, 19
        N_ = lrtlist_cam_.shape[0]
        N = 32
        lrtlist_cam = torch.zeros((N, 19), dtype=torch.float32)
        lrtlist_cam[:N_] = lrtlist_cam_
        
        # rylist = torch.from_numpy(rylist) # N
        # scorelist = torch.from_numpy(scorelist) # N
        scorelist = torch.ones_like(lrtlist_cam[:,0])
        tidlist = torch.ones_like(lrtlist_cam[:,0]).long()
        
        samp = {}
        samp['rgb_cam'] = rgb_cam
        samp['xyz_cam'] = xyz_cam
        samp['pix_T_cam'] = pix_T_cam
        samp['lrtlist_cam'] = lrtlist_cam
        samp['scorelist'] = scorelist
        samp['tidlist'] = tidlist
        # samp['rylist'] = rylist
        # samp['scorelist'] = scorelist


        if self.load_seg:
            seg_xyz_cam_list = d['seg_xyz_cam_list']
            seg_xyz_cam_list = [torch.from_numpy(xyz) for xyz in seg_xyz_cam_list] # [?,3]
            samp['seg_xyz_cam_list'] = seg_xyz_cam_list

        if self.load_i:
            xyz_cam_i = d['xyz_cam_i']
            xyz_cam_i = torch.from_numpy(xyz_cam_i)
            samp['xyz_cam_i'] = xyz_cam_i
        
        return samp

    def __len__(self):
        return len(self.records)

