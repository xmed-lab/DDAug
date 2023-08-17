export nnUNet_raw_data_base="/home/michael/SSD_Cache/MedicalData/nnunet_dir/"
export nnUNet_preprocessed="/home/michael/SSD_Cache/MedicalData/nnunet_dir/processed"
export RESULTS_FOLDER="/home/michael/SSD_Cache/MedicalData/nnunet_dir/result"

for task in 3 4 6 7 8 9 10 17
do 
	nnUNet_plan_and_preprocess -t $task --verify_dataset_integrity 
done 
