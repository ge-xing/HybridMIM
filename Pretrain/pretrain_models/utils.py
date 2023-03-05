
import torch
import numpy as np
from einops import rearrange

def patchify(in_channels, imgs, patch_size):
    """
    imgs: (N, 4, D, H, W)
    x: (N, L, patch_size**3 *4)
    """
    p = patch_size[0]
    assert imgs.shape[3] == imgs.shape[4] and imgs.shape[2] % p == 0
    d = h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], in_channels, d, p, h, p, w, p))
    x = torch.einsum('ncdkhpwq->ndhwkpqc', x)
    x = x.reshape(shape=(imgs.shape[0], d * h * w, p**3 * in_channels))
    return x

def unpatchify(in_channels, x, patch_size, image_size):
    """
    x: (N, L, patch_size**3 *4)
    imgs: (N, 4, D, H, W)
    """
    p = patch_size[0]
    d, h, w = image_size
    assert h * w * d == x.shape[1]

    x = x.reshape(shape=(x.shape[0], d, h, w, p, p, p, in_channels))
    x = torch.einsum('ndhwkpqc->ncdkhpwq', x)
    imgs = x.reshape(shape=(x.shape[0], in_channels, d * p, h * p, h * p))
    return imgs

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

def mask_func(x, in_channels, mask_ratio, patch_size, image_size, mask_value=0.0):
    batch = x.shape[0]
    x_patch = patchify(in_channels, x, patch_size)

    mask_patch, mask, id = random_masking(x_patch, mask_ratio)
    mask_tokens = torch.ones(1, 1, in_channels * patch_size[0] * patch_size[1] * patch_size[2]) * mask_value
    device = x.device
    mask_tokens = mask_tokens.repeat(batch,  id.shape[1] - mask_patch.shape[1], 1)
    mask_tokens = mask_tokens.to(device)

    x_ = torch.cat([mask_patch, mask_tokens], dim=1)  # no cls token
    x_ = torch.gather(x_, dim=1, index=id.unsqueeze(-1).repeat(1, 1, mask_patch.shape[2]))  # unshuffle
    # mask the input
    x = unpatchify(in_channels, x_, patch_size=patch_size, image_size=image_size)
    return x, mask

def get_region_nums(mask_nums, patches_of_region):
    assert mask_nums % patches_of_region == 0
    return mask_nums // patches_of_region, patches_of_region

def get_mask_labels(batch_size, num_regions, mask, mask_region_patches, device):
    mask_labels = []
    for b in range(batch_size):
        mask_label_b = []
        for i in range(num_regions):
            mask_label_b.append(mask[b, i*mask_region_patches: (i+1)*mask_region_patches].sum().item())
        mask_labels.append(mask_label_b)
    mask_labels = torch.tensor(mask_labels, device=device).long()

    return mask_labels

def get_mask_labelsv2(batch_size, num_regions, mask, mask_region_patches, device):
    mask_labels = torch.zeros(batch_size, num_regions, mask_region_patches).to(device)
    for b in range(batch_size):
        for i in range(len(mask[b])):
            region_i = i // mask_region_patches
            patch_i = i % mask_region_patches
            mask_labels[b, region_i, patch_i] = mask[b, i]
    return mask_labels

def get_random_patch(img,
                     downsample_scale,
                     mask_labels,
                     patches_of_region):

    device = img.device
    batch_size = img.shape[0]
    in_channels = img.shape[1]
    d, w, h = img.shape[2], img.shape[3], img.shape[4]
    patch_scale = (d // downsample_scale[0], w // downsample_scale[1], h // downsample_scale[2])
    img = rearrange(img, "b c (p f) (q g) (o h) -> b (f g h) (c p q o)",
                    p=downsample_scale[0], q=downsample_scale[1], o=downsample_scale[2],
                    f=patch_scale[0], g=patch_scale[1], h=patch_scale[2])
    rec_patchs = torch.zeros(img.shape[0],
                             in_channels,
                             downsample_scale[0],
                             downsample_scale[1],
                             downsample_scale[2],
                             device=device)
    index = []
    mask_labels_cpu = mask_labels.cpu().numpy()

    for b in range(batch_size):
        no_all_mask_patches = np.argwhere(mask_labels_cpu[b] < patches_of_region).reshape(-1)
        # get the random patch index
        random_rec_patch_index = no_all_mask_patches[np.random.randint(0, len(no_all_mask_patches))]
        index.append(random_rec_patch_index)
        rec_patchs[b] = rearrange(img[b, random_rec_patch_index], "(c p q o) -> c p q o",
                                  c=in_channels,
                                  p=downsample_scale[0],
                                  q=downsample_scale[1],
                                  o=downsample_scale[2])

    return rec_patchs, index

def get_random_patch_new(img,
                     downsample_scale,):

    device = img.device
    batch_size = img.shape[0]
    in_channels = img.shape[1]
    patch_images = patchify(in_channels, img, downsample_scale)
    num_patchs = patch_images.shape[1]

    rec_patchs = torch.zeros(img.shape[0],
                             in_channels,
                             downsample_scale[0],
                             downsample_scale[1],
                             downsample_scale[2],
                             device=device)
    index = []

    for b in range(batch_size):
        # get the random patch index
        p_sum = 0
        while p_sum == 0:
            random_index = np.random.randint(0, num_patchs)
            random_patch = patch_images[b, random_index]
            p_sum = random_patch.sum()

            random_patch = random_patch.reshape(shape=(downsample_scale[0], downsample_scale[1], downsample_scale[2], in_channels))
            random_patch = torch.einsum("hpqc->chpq", random_patch)

        index.append(random_index)
        rec_patchs[b] = random_patch

    return rec_patchs, index