import argparse
import os
import tarfile
import time

import numpy as np
import oxford_robotcar_python.camera_model as oxford_camera
import oxford_robotcar_python.image as oxford_image

from learnlarge.util.cv import resize_img
from learnlarge.util.helper import fs_root
from learnlarge.util.io import load_csv, save_img, save_txt, save_csv
from learnlarge.util.sge import run_one_job


def create_array_job(ins_root, log_dir):
    run_one_job(script=__file__, queue='long', cpu_only=True, memory=10, script_parameters=[], out_dir=log_dir,
                name='downsize_240', overwrite=True, hold_off=False, num_jobs=len(os.listdir(ins_root)), array=True)


def downsize_images(task_id, max_side, img_root, ins_root, tar_root, out_img_root, out_root, cams):
    # Find all dates with INS data (not all images have ins, but all ins should have images)
    all_dates = sorted(os.listdir(ins_root))  # Sort to make sure we always get the same order

    date = all_dates[int(task_id) - 1]
    print(date)

    out_file = os.path.join(out_root, 'img_info_{}'.format(max_side), '{}.csv'.format(date))
    if os.path.exists(out_file):
        print('Output already exists.')
        return

    imgs = load_csv(os.path.join(img_root, date, 'stereo.timestamps'), has_header=False, delimiter=' ',
                    keys=['t', 'folder'])
    cam = oxford_camera.CameraModel(cams,
                                    '/stereo/centre/')
    exposures = [0] * len(imgs['t'])
    max_folder = max(np.array(imgs['folder'], dtype=int))

    if date == '2015-09-02-10-37-32':
        max_folder = 4  # Folders 5 and 6 are missing from the website
        imgs['t'] = [t for f, t in zip(imgs['folder'], imgs['t']) if int(f) <= max_folder]
        imgs['folder'] = [f for f in imgs['folder'] if int(f) <= max_folder]

    for folder in range(1, max_folder + 1):
        filename = os.path.join(tar_root, '{}_stereo_centre_{:02d}.tar'.format(date, folder))
        print(filename)
        if not os.path.exists(filename):
            print("MISSING!!")
            save_txt(txt=filename, mode='a', out_file=os.path.join(out_root, 'missing.txt'))

        with tarfile.open(filename) as archive:
            print(archive)
            for entry in archive.getmembers():
                img_name = os.path.basename(entry.name)
                if '.png' not in img_name:
                    continue
                ts = img_name.split('.')[0]
                img_path = entry.name
                with archive.extractfile(archive.getmember(img_path)) as file:
                    timer = time.time()
                    index = imgs['t'].index(ts)  # Assuming that timestamps are not ordered
                    try:
                        img = oxford_image.load_image(file, cam)  # One file has unloadable image...
                        img = resize_img(img, max_side)
                        exposures[index] = sum(np.array(img).flatten())
                        out_img_folder = os.path.join(out_img_root, '{}_stereo_centre_{:02d}'.format(date, folder))
                        if not os.path.exists(out_img_folder):
                            os.makedirs(out_img_folder)
                        out_img_path = os.path.join(out_img_folder, img_name)
                        save_img(img, out_img_path)
                        print('Processed {} in {}s.'.format(ts, time.time() - timer))
                    except:
                        del exposures[index]
                        del imgs['t'][index]
                        del imgs['folder'][index]

    imgs['exposure'] = exposures
    save_csv(imgs, out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=0)
    parser.add_argument('--max_side', type=int, default=240)
    parser.add_argument('--img_root', default='/path/to/server/files/data/datasets/oxford_ext_raw_stereo_centre')
    parser.add_argument('--ins_root', default=os.path.join(fs_root(), 'data/datasets/oxford_extracted'))
    parser.add_argument('--tar_root', default='/path/to/files/oxford_scraped')
    parser.add_argument('--out_img_root', default=os.path.join(fs_root(), 'datasets/oxford_240'))
    parser.add_argument('--out_root', default=os.path.join(fs_root(), 'data/learnlarge'))
    parser.add_argument('--log_dir', default=os.path.join(fs_root(), 'cpu_logs/downsize'))
    parser.add_argument('--cams', default=os.path.join(fs_root(), 'code/robotcar-dataset-sdk-2.1/models'))
    args = parser.parse_args()

    task_id = args.task_id
    max_side = args.max_side
    img_root = args.img_root
    ins_root = args.ins_root
    tar_root = args.tar_root
    out_img_root = args.out_img_root
    out_root = args.out_root
    log_dir = args.log_dir
    cams = args.cams

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    if task_id == -1:
        create_array_job(ins_root, log_dir)
    elif task_id == 0:
        for task_id in range(1, len(os.listdir(ins_root)) + 1):
            downsize_images(task_id, max_side, img_root, ins_root, tar_root, out_img_root, out_root, cams)
    else:
        downsize_images(task_id, max_side, img_root, ins_root, tar_root, out_img_root, out_root, cams)


if __name__ == "__main__":
    main()
