"""
Crop upper boddy in every video frame, square bounding box is averaged among all frames and fixed.
"""

import os
import cv2
import argparse
from tqdm import tqdm
import face_recognition
import torch
import util
import numpy as np
import face_detection
import face_alignment
import imutils


def calc_bbox(image_list, batch_size=5):
    """Batch infer of face location, batch_size should be factor of total frame number."""
    fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda')

    top_best = 10000
    bottom_best = 0
    right_best = 0
    left_best = 10000

    for i in tqdm(range(len(image_list) // batch_size - batch_size)):
        image_batch = []

        for j in range(i * batch_size, (i + 1) * batch_size):
            image = face_recognition.load_image_file(image_list[j])
            image_batch.append(image)
        
        # face_locations = face_recognition.batch_face_locations(image_batch, number_of_times_to_upsample=0, batch_size=batch_size)
        preds = fa.get_detections_for_batch(np.asarray(image_batch))

        for face_location in preds:
            left, top, right, bottom = face_location  # assuming only one face detected in the frame
            if top_best > top:
                top_best = top
            if bottom_best < bottom:
                bottom_best = bottom
            if right_best < right:
                right_best = right
            if left_best > left:
                left_best = left

    return top_best, right_best, bottom_best, left_best


def crop_image(data_dir, dest_size, crop_level, vertical_adjust):
    image_list = util.get_file_list(os.path.join(data_dir, 'full'))
    H, W, _ = face_recognition.load_image_file(image_list[0]).shape
    top, right, bottom, left = calc_bbox(image_list)
    height = bottom - top
    width = right - left

    crop_size = int(height * crop_level)

    horizontal_delta = (crop_size - width) // 2
    vertical_delta = (crop_size - height) // 2

    left = max(left - horizontal_delta, 0)
    right = min(right + horizontal_delta, W)

    top = max(top - int(vertical_delta * 0.5), 0)
    bottom = min(bottom + int(vertical_delta * 1.5), H)

    for i in tqdm(range(len(image_list))):
        image =cv2.imread(image_list[i])
        image = image[top:bottom, left:right]

        image = cv2.resize(image, (dest_size, dest_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(data_dir, 'crop', os.path.basename(image_list[i])), image)
        torch.save([top, bottom, left, right], os.path.join(data_dir, 'crop_region', os.path.basename(image_list[i]))[:-4]+'.pt')



def crop_per_image(data_dir, dest_size, crop_level):
    fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda')

    image_list = util.get_file_list(os.path.join(data_dir, 'full'))
    batch_size = 5
    frames = []

    for i in tqdm(range(len(image_list))):
        frame = face_recognition.load_image_file(image_list[i])
        frames.append(frame)

    H, W, _ = frames[0].shape

    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]

    for idx in tqdm(range(len(batches))):
        fb = batches[idx]
        preds = fa.get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            if f is None:
                print('no face in image {}'.format(idx * batch_size + j))
            else:
                left, top, right, bottom = f

            
            height = bottom - top
            width = right - left
            crop_size = int(height * crop_level)

            horizontal_delta = (crop_size - width) // 2
            vertical_delta = (crop_size - height) // 2

            left = max(left - horizontal_delta, 0)
            right = min(right + horizontal_delta, W)
            top = max(top - int(vertical_delta * 0.5), 0)
            bottom = min(bottom + int(vertical_delta * 1.5), H)
            
            crop_f = cv2.imread(image_list[idx * batch_size + j])
            crop_f = crop_f[top:bottom, left:right]
            crop_f = cv2.resize(crop_f, (dest_size, dest_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(data_dir, 'crop', os.path.basename(image_list[idx * batch_size + j])), crop_f)
            torch.save([top, bottom, left, right], os.path.join(data_dir, 'crop_region', os.path.basename(image_list[idx * batch_size + j]))[:-4]+'.pt')


def normalize_and_crop_lip_region(data_dir):
    util.create_dir(os.path.join(data_dir, 'crop_lip'))
    image_list = util.get_file_list(os.path.join(data_dir, 'full'))

    fa_3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cuda')

    for i in tqdm(range(len(image_list))):
        image_name = image_list[i]
        image = cv2.imread(image_name)
        preds = fa_3d.get_landmarks(image)

        assert preds is not None
        left_eye_landmark = preds[0][:, :2][22:27].astype(int)
        right_eye_landmark = preds[0][:, :2][17:22].astype(int)
        
        leftEyeCenter = left_eye_landmark.mean(axis=0).astype(int)
        rightEyeCenter = right_eye_landmark.mean(axis=0).astype(int)
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        
        desiredRightEyeX = 1.0 - 0.35
        desiredLeftEyeX = 0.35
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - desiredLeftEyeX)
        desiredDist *= 256
        scale = desiredDist / dist

        eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) // 2), int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))

        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        
        tX = 256 * 0.5
        tY = 256 * 0.35    
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        
        (w, h) = (256, 256)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)    
        
        preds = fa_3d.get_landmarks(output)
        lip_landmark = preds[0][:, :2][48:].astype(int)
        (x, y, w, h) = cv2.boundingRect(lip_landmark)
        resized_roi = np.zeros((256,256,3))
        roi = output[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=256, inter=cv2.INTER_CUBIC)
        
        # h, w, _ = roi.shape
        # resized_roi[:h, :, :] = roi
        cv2.imwrite(os.path.join(data_dir, 'crop_lip', os.path.basename(image_list[i])), roi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--dest_size', type=int, default=256)
    parser.add_argument('--crop_level', type=float, default=2.0, help='Adjust crop image size.')
    parser.add_argument('--vertical_adjust', type=float, default=0.3, help='Adjust vertical location of portrait in image.')
    args = parser.parse_args()
    util.create_dir(os.path.join(args.data_dir, 'crop_region'))
    # crop_per_image(args.data_dir, dest_size=args.dest_size, crop_level=args.crop_level)
    # crop_image(args.data_dir, dest_size=args.dest_size, crop_level=args.crop_level, vertical_adjust=args.vertical_adjust)
    
    normalize_and_crop_lip_region(args.data_dir)
