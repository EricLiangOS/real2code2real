from pathlib import Path
import cv2
import matplotlib.cm as cm
import torch
import numpy as np

from submodules.SuperGluePretrainedNetwork.models.matching import Matching
from submodules.SuperGluePretrainedNetwork.models.utils import (process_resize, VideoStreamer,
                                             make_matching_plot_fast, frame2tensor)

def prepare_image(image, resize_dimensions):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    w, h = gray_image.shape[1], gray_image.shape[0]
    w_new, h_new = process_resize(w, h, resize_dimensions)

    gray_image = cv2.resize(gray_image, (w_new, h_new), interpolation=cv2.INTER_AREA)

    return gray_image

def prepare_depth(depth, resize_dimensions):
    w, h = depth.shape[1], depth.shape[0]
    w_new, h_new = process_resize(w, h, resize_dimensions)

    depth = cv2.resize(depth, (w_new, h_new), interpolation=cv2.INTER_AREA)

    return depth

# Returns a dictionary with size num_frames - 1 representing the matches between i and i - 1
def pairwise_matching(data, resize_dimensions, output_dir = None, alignment_frames = None, max_matches = 15):
    torch.set_grad_enabled(False)

    show_keypoints = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        },
        'superglue': {
            'weights': "indoor",
            'sinkhorn_iterations': 20,
            'match_threshold': 0.65,
        }
    }

    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    all_matches = {}
    last_frame = None
    last_frame_depth = None
    last_frame_tensor = None
    last_data = {}
    last_frame_id = -1

    for i in data["frames"]:
        if alignment_frames is not None and i not in alignment_frames:
            continue

        if last_frame is None:
            H, W = data["frames"][i][0].shape[:2]
            last_frame = prepare_image(data["frames"][i][0], resize_dimensions)
            last_frame_depth = prepare_depth(data["frames"][i][1], resize_dimensions)
            last_frame_tensor = frame2tensor(last_frame, device)
            last_data = matching.superpoint({'image': last_frame_tensor})
            last_data = {k+'0': last_data[k] for k in keys}
            last_data['image0'] = last_frame_tensor
            last_frame_id = i
            continue

        frame = prepare_image(data["frames"][i][0], resize_dimensions)
        frame_depth = prepare_depth(data["frames"][i][1], resize_dimensions)
        frame_tensor = frame2tensor(frame, device)

        pred = matching({**last_data, 'image1': frame_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]    

        pairs = []
        for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, confidence[valid]):
            d0 = last_frame_depth[int(y0), int(x0)]
            d1 = frame_depth[int(y1), int(x1)]
            avg_depth = 0.5 * (d0 + d1)
            # Only keep pairs with a positive (non-zero) average depth
            if avg_depth > 0:
                pairs.append(((x0, y0), (x1, y1), c, avg_depth))

        # Sort pairs by their average depth, descending
        pairs_sorted = sorted(pairs, key=lambda x: x[3])[:max_matches]

        # Replace mkpts0, mkpts1 with the sorted arrays
        mkpts0 = np.array([p[0] for p in pairs_sorted])
        mkpts1 = np.array([p[1] for p in pairs_sorted])
        confidence_valid = np.array([p[2] for p in pairs_sorted])
        depth_vals = np.array([p[3] for p in pairs_sorted])

        pair_matches = [[], []]

        for (x0, y0), (x1, y1), c, d in pairs_sorted:

            x0 = x0 *  W / resize_dimensions[0]
            y0 = y0 * H / resize_dimensions[1]

            x1 = x1 * W / resize_dimensions[0]
            y1 = y1 * H / resize_dimensions[1]

            pair_matches[0].append([int(x0), int(y0)])
            pair_matches[1].append([int(x1), int(y1)])

        all_matches[i] = pair_matches

        color = cm.jet(confidence_valid)
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {:06}:{:06}'.format(last_frame_id, i),
        ]
        out = make_matching_plot_fast(
            last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=show_keypoints, small_text=small_text)

        if output_dir is not None:
            stem = 'matches_{:06}_{:06}'.format(last_frame_id, i)
            out_file = str(Path(output_dir, stem + '.png'))
            cv2.imwrite(out_file, out)

        last_frame = prepare_image(data["frames"][i][0], resize_dimensions)
        last_frame_depth = prepare_depth(data["frames"][i][1], resize_dimensions)
        last_frame_tensor = frame2tensor(last_frame, device)
        last_data = matching.superpoint({'image': last_frame_tensor})
        last_data = {k+'0': last_data[k] for k in keys}
        last_data['image0'] = last_frame_tensor
        last_frame_id = i

    torch.set_grad_enabled(True)
    return all_matches

def target_matching(target, images, resize_dimensions, output_path = None, max_matches = 15):
    torch.set_grad_enabled(False)

    show_keypoints = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        },
        'superglue': {
            'weights': "indoor",
            'sinkhorn_iterations': 20,
            'match_threshold': 0.7,
        }
    }

    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    target_frame = prepare_image(target, resize_dimensions)
    target_tensor = frame2tensor(target_frame, device)

    target_data = matching.superpoint({'image': target_tensor})
    target_data = {k+'0': target_data[k] for k in keys}
    target_data['image0'] = target_tensor

    best_matches = [[], []]
    best_matches_frame = 0
    best_matches_info = {}

    H1, W1 = images[0].shape[:2]

    for i, frame in enumerate(images):

        frame = prepare_image(frame, resize_dimensions)
        frame_tensor = frame2tensor(frame, device)

        pred = matching({**target_data, 'image1': frame_tensor})
        kpts0 = target_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        target_matches = [[], []]

        pairs = list(zip(mkpts0, mkpts1, confidence[valid]))
        pairs_sorted = sorted(pairs, key = lambda x: x[2], reverse = True)
        pairs_sorted = pairs_sorted[:max_matches]

        for (x0, y0), (x1, y1), c in pairs_sorted:
            target_matches[0].append([int(x0), int(y0)])
            target_matches[1].append([int(x1), int(y1)])        

        if len(target_matches[0]) > len(best_matches[0]):
            best_matches = target_matches
            best_matches_frame = i
            best_matches_info = {
                "kpts0": kpts0,
                "kpts1": kpts1,
                "mkpts0": mkpts0,
                "mkpts1": mkpts1,
                "color": cm.jet(confidence[valid]),
                "frame": frame,
                "valid": valid
            }

    out = None
    if best_matches_info:
        kpts0 = best_matches_info["kpts0"]
        kpts1 = best_matches_info["kpts1"]
        mkpts0 = best_matches_info["mkpts0"]
        mkpts1 = best_matches_info["mkpts1"]
        color = best_matches_info["color"]
        frame = best_matches_info["frame"]

        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: target:{:06}'.format(best_matches_frame),
        ]
        out = make_matching_plot_fast(
            target_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=output_path, show_keypoints=show_keypoints, small_text=small_text)
            
    for i in range(len(best_matches[0])):
        new_x0 = best_matches[0][i][0] *  target.shape[1] / resize_dimensions[0]
        new_y0 = best_matches[0][i][1] * target.shape[0] / resize_dimensions[1]
        best_matches[0][i] = [int(new_x0), int(new_y0)]

        new_x1 = best_matches[1][i][0] * W1/ resize_dimensions[0]
        new_y1 = best_matches[1][i][1] * H1 / resize_dimensions[1]
        best_matches[1][i] = [int(new_x1), int(new_y1)]

    torch.set_grad_enabled(True)
        
    return best_matches, best_matches_frame, out