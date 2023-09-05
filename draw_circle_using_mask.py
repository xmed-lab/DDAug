import os 
import sys 

import cv2 
import tifffile 
import nibabel as nib
from PIL import Image, ImageFilter

from tqdm import tqdm 

import torch 
import numpy as np 

'''
the program expects src_dir to include background files, and foldername associated to background files name with different mask in it, e.g. 

src_dir: 
    input_file.tif 
    input_file: 
        mask_file1.tif
        mask_file2.tif 
        mask_file3.tif 
    input_file2.tif 
    input_file2: 
        mask_file1.tif
        mask_file2.tif 
        mask_file3.tif 

and will generate corresponding files in output_dir, e.g. 
output_dir: 
    input_file: 
        mask_file1_index1.png 
        mask_file1_index2.png 
        ...
        mask_file2_index1.png 
        ...
    input_file2: 
        mask_file1_index1.png 
        mask_file1_index2.png 
        ... 
'''

color_mapping = {
    'red': np.array([255, 0, 0], dtype=np.uint8).reshape((1,1,3)),
    'green': np.array([0, 255, 0], dtype=np.uint8).reshape((1,1,3))
}

clip_mapping = {
    'BIDMC_06': (0, 2467.31),
    'BIDMC_08': (0, 3203.12),
    'I2CVB_05': (0, 920),
    'I2CVB_10': (0, 682),
    'I2CVB_17': (0, 314),
    'I2CVB_18': (0, 248),
}
gt_color = color_mapping['green']
stroke_color = color_mapping['red']
edge_expand_kernel_size = 3 

src_dir = ''
output_dir = ''


def create_one_slice(src, edges, name): 
    assert src.shape == edges.shape, f'{src.shape = } | {edges.shape = }'
    dir = '/'.join(name.split('/')[:-1])
    os.makedirs(dir) if not os.path.isdir(dir) else None 
    
    with torch.no_grad():
        kernel = torch.nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=edge_expand_kernel_size, 
            stride=1, 
            padding=edge_expand_kernel_size//2, 
            bias=False
        )
        kernel.weight.data = torch.ones((1, 1, edge_expand_kernel_size, edge_expand_kernel_size)).float()

        edges = torch.from_numpy(edges).unsqueeze(0).unsqueeze(0).float()
        edges = kernel(edges).squeeze().numpy() * 10
    
    src = src.astype(np.float32) / src.max() * 255
    image = Image.fromarray(src.astype(np.uint8), mode='L').convert('RGB')
    image = np.array(image)
    image[edges > 0] = stroke_color if 'gt' not in name else gt_color
    
    image = Image.fromarray(image)
    image.save(name)


def load_data(path):
    if path.endswith('nii'):
        image = nib.load(path)
        image = np.array(image.dataobj)
        image = image.transpose(2,0,1)
    elif path.endswith(('tif', 'tiff')): 
        image = tifffile.imread(path)
    else:
        print(f'file format not recognized: {path}')
        sys.exit()
    return image 


def main(): 
    background = filter(lambda each: os.path.isfile(f'{src_dir}/{each}'), os.listdir(src_dir))
    
    for each_name in sorted(background): 
        current_bg = load_data(f'{src_dir}/{each_name}')
        current_bg = current_bg.clip(*clip_mapping[each_name.split('.')[0]])
        assert len(current_bg.shape) == 3 
        
        index, _, _ = current_bg.shape 
        
        mask_folder = each_name.split('.')[0]
        mask_folder = f'{src_dir}/{mask_folder}'
        for each_mask in sorted(os.listdir(mask_folder)): 
            current_mask = load_data(f'{mask_folder}/{each_mask}').astype(np.int8)
            assert current_bg.shape[-2:] == current_mask.shape[-2:]
            each_name = each_name.split('.')[0]
            each_mask = each_mask.split('.')[0]
            
            for each_index in tqdm(range(index), ncols=80, desc=f'{each_name} [{each_mask}]'): 
                background_slice = current_bg[each_index]
                mask_slice = current_mask[each_index]
                
                edges = Image.fromarray(mask_slice, mode='L')
                edges = edges.filter(ImageFilter.FIND_EDGES)
                edges = np.array(edges)
                
                if background_slice.shape != edges.shape: 
                    print(f'background slice: {background_slice.shape} shape is different from edge shape: {edges.shape}, make sure this is expected behavior')
                    background_slice = cv2.resize(background_slice, edges.shape, interpolation=cv2.INTER_LANCZOS4)
                
                create_one_slice(background_slice, edges, f'{output_dir}/{each_name}/index{each_index+1}_{each_mask}.png')


if __name__ == '__main__':
    main() 