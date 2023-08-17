import json
import os
import SimpleITK as sitk
import numpy as np
import argparse
import pandas as pd
from medpy.metric import binary


def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2.0 * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())


def hd(pred, gt):
    return 0
    if pred.sum() > 0 and gt.sum() > 0:
        return binary.hd95(pred, gt)
    else:
        return 0


def test(pred_folder, gt_folder):
    pred_list = [
        (each.split("/")[-1], f"{pred_folder}/{each}")
        for each in sorted(os.listdir(pred_folder))
        if each.endswith("gz")
    ]
    gt_list = [
        (each.split("/")[-1], f"{gt_folder}/{each}")
        for each in sorted(os.listdir(gt_folder))
        if each.endswith("gz")
    ]

    assert all(pred[0] == gt[0] for pred, gt in zip(pred_list, gt_list))
    table = pd.DataFrame()
    row_index = []
    for prediction, label in zip(pred_list, gt_list):
        print(prediction[0], label[0], sep="\n", end="\n\n")
        pred_data = read_nii(prediction[1])
        label_data = read_nii(label[1])
        current_result = {}
        fg_dice, fg_hd = [], []
        row_index.append(prediction[0])
        for each_class in sorted(np.unique(pred_data.flatten())):
            pred_class, label_class = (pred_data == each_class).astype(int), (
                label_data == each_class
            ).astype(int)
            dice_class, hd_class = dice(pred_class, label_class), hd(pred_class, label_class)
            current_result[f"class_{each_class}_dice"] = [dice_class]
            current_result[f"class_{each_class}_hd"] = [hd_class]
            if each_class != 0:
                fg_dice.append(dice_class)
                fg_hd.append(hd_class)
        current_result["fg_dice"] = [np.mean(fg_dice)]
        current_result["fg_hd"] = [np.mean(fg_hd)]
        table = pd.concat([table, pd.DataFrame(current_result)])

    mean_all = pd.DataFrame(data=[table.mean(axis=0).to_numpy()], columns=table.columns)
    std_all = pd.DataFrame(data=[table.std(axis=0).to_numpy()], columns=table.columns)
    table = pd.concat([table, mean_all, std_all]) * 100
    table.index = row_index + ["mean", "std"]
    table.to_csv(f"{pred_folder}/result.csv")
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="prediction label folder")
    parser.add_argument("-t", help="ground truth label folder")
    args = parser.parse_args()
    test(args.p.rstrip("/"), args.t.rstrip("/"))


"""
nnUNet_predict -i /mnt/SSD_RAID/data/MedicalData/testing/data/ -o /mnt/SSD_RAID/data/incoming/nnUNetV2_NoAug_val/Task005_Prostate/nnUNetTrainerV2__nnUNetPlansv2.1/fold_4/testing/ -t Task005_Prostate -m 3d_fullres -f 4
"""
