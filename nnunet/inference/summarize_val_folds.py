import os
import pickle
import shutil
import subprocess


tmp_input_folder = 'some_temporary_folder_,_content_and_folder_will_be_deleted_once_script_finishes'
tmp_gt_folder = 'some_temporary_folder_,_content_and_folder_will_be_deleted_once_script_finishes'

raw_data_path = os.environ["nnUNet_raw_data_base"] + "/nnUNet_raw_data"
processed_data_path = os.environ["nnUNet_preprocessed"]
result_path = os.environ["RESULTS_FOLDER"] + "/nnUNet/3d_fullres"


print(f"working with [{raw_data_path=}]")
print(f"working with [{processed_data_path=}]")
print(f"working with [{result_path=}]")


def make_folder(folder):
    if os.path.isdir(folder):
        subprocess.run(["rm", "-rf", folder])
    os.mkdir(folder)


def create_soft_link(src, dst, candidates):
    for each_file in os.listdir(src):
        if any(each_file.startswith(each) for each in candidates):
            in_file, out_file = f"{src}/{each_file}", f"{dst}/{each_file}"
            in_file, out_file = in_file.replace('//', '/'), out_file.replace('//', '/')
            # subprocess.run(["cp", "-ns", in_file, out_file])
            subprocess.run(["cp", in_file, out_file])


def remove_soft_link(folder):
    for each_file in os.listdir(folder):
        # subprocess.run(["unlink", f"{folder}/{each_file}"])
        subprocess.run(["rm", f"{folder}/{each_file}"])


def main():
    for each_task in sorted(os.listdir(result_path)):
        folds_data = pickle.load(open(f"{processed_data_path}/{each_task}/splits_final.pkl", "rb"))

        if not os.path.isdir(f"{result_path}/{each_task}"):
            continue
        for each_model_folder in sorted(os.listdir(f"{result_path}/{each_task}")):
            model_name = each_model_folder.split("__")[0]
            
            if 'BIDMC' not in each_task:
                print(f"passed [{each_task}] [{model_name}]")
                continue 

            for each_fold in range(5):
                current_val = sorted(list(folds_data[each_fold]["val"]))
                fold_output = f"{result_path}/{each_task}/{each_model_folder}/fold_{each_fold}/testing"

                # if os.path.isdir(fold_output) and len(os.listdir(fold_output)) == len(current_val) + 3:
                #     continue
                if os.path.isdir(fold_output):
                    shutil.rmtree(fold_output)

                print(f"\n\nworking on [{each_task}] [{model_name}] [{each_fold}]")

                os.makedirs(fold_output)
                create_soft_link(f"{raw_data_path}/{each_task}/imagesTr", tmp_input_folder, current_val)
                create_soft_link(f"{raw_data_path}/{each_task}/labelsTr", tmp_gt_folder, current_val)

                if os.path.isfile(f"{result_path}/{each_task}/{each_model_folder}/fold_{each_fold}/model_final_checkpoint.model"):
                    checkpoint = 'model_final_checkpoint'
                else:
                    checkpoint = 'model_latest'

                try:
                    subprocess.run(
                        [
                            "nnUNet_predict",
                            "-i",
                            tmp_input_folder,
                            "-o",
                            fold_output,
                            "-t",
                            each_task,
                            "-f",
                            str(each_fold),
                            "-tr",
                            model_name,
                            "-chk",
                            checkpoint,
                            "--disable_tta",
                        ],
                        stdout=subprocess.DEVNULL,
                    )

                    print('generating csv')
                    subprocess.run(
                        [
                            "python",
                            "/mnt/TrueNAS_Storage/Code/DDAug/nnunet_layer_aug/nnUNet/nnunet/inference/generate_result.py",
                            "-p",
                            fold_output,
                            "-t",
                            tmp_gt_folder,
                        ],
                        # stdout=subprocess.DEVNULL,
                    )
                except Exception as e:
                    shutil.rmtree(fold_output)
                    print(f"error {e} occurred with -> [{each_task}] [{model_name}] [{each_fold}]")

                remove_soft_link(tmp_input_folder)
                remove_soft_link(tmp_gt_folder)

    remove_soft_link(tmp_input_folder)
    remove_soft_link(tmp_gt_folder)
    shutil.rmtree(tmp_input_folder)
    shutil.rmtree(tmp_gt_folder)


if __name__ == "__main__":
    make_folder(tmp_gt_folder)
    make_folder(tmp_input_folder)
    main()
