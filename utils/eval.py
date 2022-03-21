import torch
import numpy as np
import utils.basic
import utils.box
import utils.geom

def compute_matches(name, overlaps, pred_scores, 
                    iou_threshold=0.5, score_threshold=0.0,
                    oracle=False):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    n_boxes, n_gt_boxes = overlaps.shape
    # Sort predictions by score from high to low
    # indices = np.argsort(pred_scores)[::-1]
    # print('SORTING BY OVERLAPS[:,0]')
    if oracle:
        indices = np.argsort(np.sum(overlaps, axis=1))[::-1]
    else:
        indices = np.argsort(pred_scores)[::-1]
        
    pred_scores = pred_scores[indices]
    overlaps = overlaps[indices]

    # print('pred_scores', pred_scores)
    # print('overlaps', overlaps)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([n_boxes])
    gt_match = -1 * np.ones([n_gt_boxes])
    for i in list(range(n_boxes)):
        # Find best matching ground truth box
        
        # 1. Sort matches by overlap
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            #if pred_class_ids[i] == gt_class_ids[j]:
            match_count += 1
            gt_match[j] = i
            pred_match[i] = j
            break

    return gt_match, pred_match, overlaps


def compute_ap(name, pred_scores, overlaps, iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        name, overlaps, pred_scores, iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1).astype(np.float32) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)
    
    # if iou_threshold==0.1:
    #     print("pred_match", pred_match, "gt_match", gt_match)
    #     print('iou_threshold', iou_threshold)
    #     print('precisions', precisions)
    #     print('recalls', recalls)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in list(range(len(precisions) - 2, -1, -1)):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])
    
    # if iou_threshold==0.1:
    #     print('map', mAP)
    #     input()
        
    return mAP, precisions, recalls, overlaps


def drop_invalid_lrts(lrtlist_e, lrtlist_g, scorelist_e, scorelist_g):
    B, N, D = lrtlist_e.shape
    assert(B==1)
    # lrtlists are shaped B x N x 19
    # scorelists are shaped B x N
    lrtlist_e_, scorelist_e_, lrtlist_g_, scorelist_g_ = [], [], [], []
    lenlist_e, _ = utils.geom.split_lrtlist(lrtlist_e)
    for i in list(range(len(lrtlist_e))):
        lrt_e = lrtlist_e[i]
        score_e = scorelist_e[i]
        len_e = lenlist_e[i]
        valid_e = torch.where(len_e[:, 0] > 0.01)
        lrtlist_e_.append(lrt_e[valid_e])
        scorelist_e_.append(score_e[valid_e])
    for i in list(range(len(lrtlist_g))):
        lrt_g = lrtlist_g[i]
        score_g = scorelist_g[i]
        valid_g = torch.where(score_g > 0.5)
        lrtlist_g_.append(lrt_g[valid_g])
        scorelist_g_.append(score_g[valid_g])
    lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = torch.stack(lrtlist_e_), torch.stack(lrtlist_g_), torch.stack(scorelist_e_), torch.stack(scorelist_g_)
    return lrtlist_e, lrtlist_g, scorelist_e, scorelist_g

def get_mAP_from_lrtlist(lrtlist_e, scores, lrtlist_g, iou_thresholds):
    # lrtlist are 1 x N x 19
    # the returns are numpy
    B, Ne, _ = list(lrtlist_e.shape)
    B, Ng, _ = list(lrtlist_g.shape)
    assert(B==1)
    scores = scores.detach().cpu().numpy()
    # print("e", boxes_e, "g", boxes_g, "score", scores)
    scores = scores.flatten()
    # size [N, 8, 3]
    ious_3d = np.zeros((Ne, Ng), dtype=np.float32)
    ious_2d = np.zeros((Ne, Ng), dtype=np.float32)
    for i in list(range(Ne)):
        for j in list(range(Ng)):
            iou_3d, iou_2d = get_iou_from_corresponded_lrtlists(lrtlist_e[:, i:i+1], lrtlist_g[:, j:j+1])
            ious_3d[i, j] = iou_3d[0, 0]
            ious_2d[i, j] = iou_2d[0, 0]
    maps_3d = []
    maps_2d = []
    for iou_threshold in iou_thresholds:
        map3d, precision, recall, overlaps = compute_ap(
            "box3d_" + str(iou_threshold), scores, ious_3d, iou_threshold=iou_threshold)
        maps_3d.append(map3d)
        map2d, precision, recall, overlaps = compute_ap(
            "box2d_" + str(iou_threshold), scores, ious_2d, iou_threshold=iou_threshold)
        maps_2d.append(map2d)
    maps_3d = np.stack(maps_3d, axis=0).astype(np.float32)
    maps_2d = np.stack(maps_2d, axis=0).astype(np.float32)
    if np.isnan(maps_3d).any():
        # print('got these nans in maps; setting to zero:', maps)
        maps_3d[np.isnan(maps_3d)] = 0.0
    if np.isnan(maps_2d).any():
        # print('got these nans in maps; setting to zero:', maps)
        maps_2d[np.isnan(maps_2d)] = 0.0

    # print("maps_3d", maps_3d)
    return maps_3d, maps_2d

def get_iou_from_corresponded_lrtlists(lrtlist_a, lrtlist_b):
    B, N, D = list(lrtlist_a.shape)
    assert(D==19)
    B2, N2, D2 = list(lrtlist_b.shape)
    assert(B2==B)
    assert(N2==N)
    
    xyzlist_a = utils.geom.get_xyzlist_from_lrtlist(lrtlist_a)
    xyzlist_b = utils.geom.get_xyzlist_from_lrtlist(lrtlist_b)
    # these are B x N x 8 x 3

    xyzlist_a = xyzlist_a.detach().cpu().numpy()
    xyzlist_b = xyzlist_b.detach().cpu().numpy()

    # ious = np.zeros((B, N), np.float32)
    ioulist_3d = torch.zeros(B, N, dtype=torch.float32, device=torch.device('cuda'))
    ioulist_2d = torch.zeros(B, N, dtype=torch.float32, device=torch.device('cuda'))
    for b in list(range(B)):
        for n in list(range(N)):
            iou_3d =  utils.box.new_box3d_iou(lrtlist_a[b:b+1,n:n+1],lrtlist_b[b:b+1,n:n+1])
            _, iou_2d = utils.box.box3d_iou(xyzlist_a[b,n], xyzlist_b[b,n]+1e-4)
            # print('computed iou %d,%d: %.2f' % (b, n, iou))
            ioulist_3d[b,n] = iou_3d
            ioulist_2d[b,n] = iou_2d
            
    # print('ioulist_3d', ioulist_3d)
    # print('ioulist_2d', ioulist_2d)
    return ioulist_3d, ioulist_2d
