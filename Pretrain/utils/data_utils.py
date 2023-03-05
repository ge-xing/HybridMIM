from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    RandSpatialCropd,
    SpatialPadd,
    ToTensord,
    RandFlipd
)
import glob

atm22_paths = glob.glob("/mnt/xingzhaohu/data/ATM22_1/imagesTr/*.nii.gz") + \
            glob.glob("/mnt/xingzhaohu/data/TrainBatch2_new/imagesTr/*.nii.gz")
luna16_paths = glob.glob("/mnt/xingzhaohu/data/luna16_convert/*.nii.gz")
covid19_paths = glob.glob("/mnt/xingzhaohu/data/COVID-19-20_v2/*/*.nii.gz")
flare21_paths = glob.glob("/mnt/xingzhaohu/data/FLARE2021/*.nii.gz") + \
             glob.glob("/mnt/xingzhaohu/data/FLARE2021/ValidationImg/*.nii.gz")

def build_ATM22():
    data_list = []
    for p in atm22_paths:
        data_list.append({"image": p})

    return data_list

def bulid_covid19():
    data_list = []
    for p in covid19_paths:
        data_list.append({"image": p})

    return data_list 

def build_flare2021():
    data_list = []
    for p in flare21_paths:
        data_list.append({"image": p})

    return data_list  

def build_luna16():
    data_list = []
    for p in luna16_paths:
        data_list.append({"image": p})
    
    return data_list

def random_selected(data: list, n):
    import random 
    total = len(data)
    val_data = []
    for i in range(n):
        random_i = random.randint(0, total-1)
        val_data.append(data[random_i])
        data.pop(random_i)
        total = len(data)
    
    return data, val_data


def get_loader(args):
    datalist1 = bulid_covid19()
    datalist2 = build_ATM22()
    datalist3 = build_flare2021()
    datalist4 = build_luna16()

    num_workers = 8
    print("Dataset 1 covid-19: number of data: {}".format(len(datalist1)))
    print("Dataset 2 ATM22: number of data: {}".format(len(datalist2)))
    print("Dataset 3 FLARE21: number of data: {}".format(len(datalist3)))
    print("Dataset 4 Luna16: number of data: {}".format(len(datalist4)))
   
    datalist = datalist1 + datalist2 + datalist3 + datalist4
    print("Dataset all training: number of data: {}".format(len(datalist)))

    train_data, val_data = random_selected(datalist, 270)

    print(f"training data is {len(train_data)}, validation data is {len(val_data)}")

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=1000, b_min=0.001, b_max=1.0, clip=True
            ),
            SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
            CropForegroundd(keys=["image"], source_key="image", k_divisible=[96, 96, 96]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[96, 96, 96],
                num_samples=2,
                random_center=True,
                random_size=False,
            ),
            # RandSpatialCropd(keys=["image"], roi_size=[96, 96, 96],
            #                             random_size=False,
            #                             allow_missing_keys=True),
            # RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            # RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            # RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            # RandSpatialCropd(keys=["image", "label"], roi_size=[96,
            #                                                     96,
            #                                                     96],
                                                                                                
            #                             random_size=False,
            #                             allow_missing_keys=True),
            ToTensord(keys=["image"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=1000, b_min=0.01, b_max=1.0, clip=True
            ),
            SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
            CropForegroundd(keys=["image"], source_key="image", k_divisible=[96, 96, 96]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[96, 96, 96],
                num_samples=2,
                random_center=True,
                random_size=False,
            ),
            # RandSpatialCropd(keys=["image"], roi_size=[96, 96, 96],
            #                             random_size=False,
            #                             allow_missing_keys=True),
            # RandSpatialCropd(keys=["image", "label"], roi_size=[96,
            #                                                     96,
            #                                                     96],
                                                                                                
            #                             random_size=False,
            #                             allow_missing_keys=True),
            ToTensord(keys=["image"]),
        ]
    )

    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=0.5, num_workers=num_workers)
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=datalist,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=8,
        )
    else:
        print("Using generic dataset")
        train_ds = Dataset(data=datalist, transform=train_transforms)

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, drop_last=True
    )
    
    if args.val_cache:
        val_ds = CacheDataset(data=val_data, transform=val_transforms, num_workers=num_workers)
    else :
        val_ds = Dataset(data=val_data, transform=val_transforms)

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=8, drop_last=True, shuffle=False
    )

    return train_loader, val_loader 