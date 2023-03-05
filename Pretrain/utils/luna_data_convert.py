
import SimpleITK as sitk
import glob 
import os 

luna16_paths = glob.glob("/mnt/xingzhaohu/data/luna16/*/*.mhd")

new_dir = "/mnt/xingzhaohu/data/luna16_convert/"
os.makedirs(new_dir, exist_ok=True)
index = 0

from multiprocessing import Pool, Process

def handle(data):
    filename = data.split("/")[-1]
    filename = filename[:-4]
    data = sitk.ReadImage(data)
    sitk.WriteImage(data, os.path.join(new_dir, f'{filename}.nii.gz'))
    print(f"{filename} save done.")
    return 

p = Pool(16)
p.map_async(handle, luna16_paths, chunksize=16)
p.close()
p.join()
