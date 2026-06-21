"""Add Image Quality transforms to images. 

This scripts handles MONAI and TorchIO transforms. It is called during domain randomisation model training and during both models' evaluation

# Motion Artifact 
Simulates patient movement during MRI acquisition
Param	
degrees: Max rotation in degrees — samples from [-x, x] per axis - 3,10,20 
translation: Max translation (maximum displacement) (in mm) along each axis- samples from [-x, x] per axis
num_transforms: Number of motion events during acquisition
p: probability of applying the transform to any given sample
"""

# Random Ghosting
"""
Simulates periodic motion (breathing, heartbeat, pulsation) causing repeated ghost copies
num_ghosts:	Number of ghost copies, sampled from this range - (2, 3)	(2, 6)	(4, 10) 
intensity:	Ghost brightness/strength of the ghost artifacts relative to the original signal
axes:	Ghosting can occur along any axis (x:0, y:1, z:2) - (0.1, 0.3)	(0.3, 0.6)	(0.6, 0.9)
p:	probability of applying the transform to any given sample
"""

#Random Blur
"""
Simulates scanner resolution loss, reconstruction smoothing, or motion-induced blurring.
Param
std:	Std of Gaussian kernel in mm, sampled from this range - (0.3, 0.8)	(0.5, 2.0)	(1.0, 4.0)
p:	probability of applying the transform to any given sample
"""


# Random bias field
"""
Simulates MRI RF field inhomogeneity causing slow, smooth intensity variations across the image.
keys: Which dict keys to apply to
prob:	probability of applying the transform to any given sample
coeff_range:	Range of polynomial coefficients controlling field strength - (0.0, 0.3)	(0.3, 0.8)	(0.8, 1.5) 
"""

# Intensity transform
"""
Simulates variability in scanner contrast settings or differences in tissue signal intensity across sites/scanners.
Param
keys:	Which dict keys to apply to
prob: probability of applying the transform to any given sample
gamma: Gamma correction exponent: <1 brightens, >1 darkens - (0.9, 1.1)	(0.7, 1.5)	(0.5, 2.0)
"""

# Random Gaussian Noise
"""
Simulates thermal/electronic noise from the MRI scanner receiver coil.
Param
keys:	Which dict keys to apply to
prob: probability of applying the transform to any given sample
mean:	Mean of Gaussian noise distribution
std:	Std of noise  - 0.01–0.03	0.03–0.08	0.08–0.15
"""


import torch
import torchio as tio
import numpy as np
from monai.transforms import Compose, RandBiasFieldd, RandGaussianNoised, RandAdjustContrastd, RandGaussianSharpend
import random
import nibabel as nib



def apply_transform(image, level):
    """
    Apply transforms to an image.
    
    Args:
        image: tio.ScalarImage or tensor
        level: 1 = single random transform, 2 = random combination of , 3 = random combination of intense transforms
    
    Returns:
        Augmented image tensor
    """
    
    # --- Define all transforms ---
    tio_transforms_mild = [
        tio.RandomMotion(degrees=5, translation=5, num_transforms=2, p=1.0),
        tio.RandomGhosting(num_ghosts=(2, 3), intensity=(0.1, 0.5), axes=(0, 1, 2), p=1.0),
        tio.RandomBlur(std=(0.3 , 0.8), p=1.0),
    ]
    
    monai_transforms_mild = [
        RandBiasFieldd(keys=["img"], prob=1.0, coeff_range=(0.0, 0.3)),
        RandAdjustContrastd(keys=["img"], prob=1.0, gamma=(0.9, 1.1)),
        RandGaussianNoised(keys=["img"], prob=1.0, mean=0.0, std=0.03),
    ]
    tio_transforms_medium = [
        tio.RandomMotion(degrees=10, translation=10, num_transforms=2, p=1.0),
        tio.RandomGhosting(num_ghosts=(3, 6), intensity=(0.5, 0.8), axes=(0, 1, 2), p=1.0),
        tio.RandomBlur(std=(0.5 , 2), p=1.0),
    ]
    
    monai_transforms_medium = [
        RandBiasFieldd(keys=["img"], prob=1.0, coeff_range=(0.3, 1)),
        RandAdjustContrastd(keys=["img"], prob=1.0, gamma=(0.7, 1.5)),
        RandGaussianNoised(keys=["img"], prob=1.0, mean=0.03, std=0.08),
    ]
    
    tio_transforms_intense = [
        tio.RandomMotion(degrees=20, translation=20, num_transforms=4, p=1.0),
        tio.RandomGhosting(num_ghosts=(5, 10), intensity=(0.5, 0.8), axes=(0, 1, 2), p=1.0),
        tio.RandomBlur(std=(1.0 , 4.0), p=1.0),
    ]
    

    monai_transforms_intense = [
        RandBiasFieldd(keys=["img"], prob=1.0, coeff_range=(1.0, 1.5)),
        RandAdjustContrastd(keys=["img"], prob=1.0, gamma=(0.5, 2.0)),
        RandGaussianNoised(keys=["img"], prob=1.0, mean=0.08, std=0.2),
    ]

    all_transforms_mild = [("tio", t) for t in tio_transforms_mild] + \
                     [("monai", t) for t in monai_transforms_mild]
    all_transforms_medium = [("tio", t) for t in tio_transforms_medium] + \
                     [("monai", t) for t in monai_transforms_medium]
    all_transforms_intense = [("tio", t) for t in tio_transforms_intense] + \
                     [("monai", t) for t in monai_transforms_intense]
    
    """
    def apply_tio(transform, img_tensor):
        subject = tio.Subject(image=tio.ScalarImage(tensor=img_tensor))
        return transform(subject)["image"].data
    
    def apply_monai(transform, img_tensor):
        data = {"img": img_tensor}
        return transform(data)["img"]
        """
    def apply_tio(transform, img_tensor):
        # Note: img_tensor is [1, C, H, W, D] from DataLoader. TorchIO expects [C, H, W, D] on CPU.
        # Squeeze batch dim, move to CPU for TorchIO, then move back to original device after.
        device = img_tensor.device
        img_tensor = img_tensor.squeeze(0).cpu()  # [1, C, H, W, D] -> [C, H, W, D] on CPU
        subject = tio.Subject(image=tio.ScalarImage(tensor=img_tensor))
        result = transform(subject)["image"].data  # [C, H, W, D]
        return result.unsqueeze(0).to(device)  # [1, C, H, W, D] back on original device

    def apply_monai(transform, img_tensor):
        #Note: baseline_t1 has shape [1, C, H, W, D] because the DataLoader adds a batch dimension (B=1).
        # MONAI transforms only support [C, H, W, D], so we squeeze/unsqueeze around the transform.
        img_tensor = img_tensor.squeeze(0)  # [1, C, H, W, D] -> [C, H, W, D]
        data = {"img": img_tensor}
        result = transform(data)["img"]
        return result.unsqueeze(0)  # [C, H, W, D] -> [1, C, H, W, D]
    
    def apply_single(transform_tuple, img_tensor):
        kind, transform = transform_tuple
        if kind == "tio":
            return apply_tio(transform, img_tensor)
        else:
            return apply_monai(transform, img_tensor)
    
    # Level 1: single random  transform 
    if level == 1:
        chosen = random.choice(all_transforms_medium)
        return apply_single(chosen, image)
    
    # Level 2: random subset of  transforms
    elif level == 2:
        k = random.randint(2, len(all_transforms_mild))
        k_1 = random.randint(1,k-1)
        chosen = random.sample(all_transforms_mild, k_1) + random.sample(all_transforms_medium, k-k_1)
        img = image
        for transform_tuple in chosen:
            img = apply_single(transform_tuple, img)
        return img
    # Level 3: random subset of transforms - Not used in this implementation
    elif level == 3:
        k = random.randint(2, len(all_transforms_mild))
        chosen = random.sample(all_transforms_mild, k)
        img = image
        for transform_tuple in chosen:
            img = apply_single(transform_tuple, img)
        return img
    # Level 4: random combination of medium and intense transforms - Not used in this implementation
    elif level == 4:
        k =  random.randint(2, len(all_transforms_mild))
        k_1 = random.randint(1,k-1)
        chosen = random.sample(all_transforms_medium, k_1) + random.sample(all_transforms_intense, k-k_1)
        img = image
        for transform_tuple in chosen:
            img = apply_single(transform_tuple, img)
        return img
    else:
        raise ValueError(f"level must be 1, 2,3,4 got {level}")

def domain_rand_transform(input_path: str, output_path: str) -> None:
    """Apply sequential domain randomization transforms to a NIfTI image.
    
    Each transform is applied independently with probability 0.5.
    Uses torchio for MRI-specific artifacts and MONAI for intensity transforms.

    Args:
        input_path: Path to input .nii.gz file
        output_path: Path to save the transformed .nii.gz file
    """
    # Load image
    nib_img = nib.load(input_path)
    affine  = nib_img.affine
    header  = nib_img.header
    data    = nib_img.get_fdata(dtype=np.float32)  # (H, W, D)


    # torchio expects (C, H, W, D)
    tio_img = tio.ScalarImage(tensor=torch.from_numpy(data).unsqueeze(0))

    """
    Apply each transformation with probability 50%
    """

    # RandomMotion
    if random.random() < 0.5:
        tio_img = tio.RandomMotion(degrees=10, translation=10, num_transforms=2)(tio_img)

    # RandomGhosting
    if random.random() < 0.5:
        tio_img = tio.RandomGhosting(num_ghosts=(2, 6), intensity=(0.1, 0.8), axes=(0, 1, 2))(tio_img)

    # RandomBlur
    if random.random() < 0.5:
        tio_img = tio.RandomBlur(std=(0.3, 2))(tio_img)

    # Convert back to numpy for MONAI transforms
    # MONAI expects (C, H, W, D) as torch tensor in a dict
    img_tensor = tio_img.tensor.float()  # (1, H, W, D)

    # MONAI transforms (operate on dict with key "img") 
    sample = {"img": img_tensor}

    # RandBiasField
    if random.random() < 0.5:
        sample = RandBiasFieldd(keys=["img"], prob=1.0, coeff_range=(0.0, 1.0))(sample)

    # RandAdjustContrast
    if random.random() < 0.5:
        sample = RandAdjustContrastd(keys=["img"], prob=1.0, gamma=(0.9, 1.1))(sample)

    # RandGaussianNoise
    if random.random() < 0.5:
        sample = RandGaussianNoised(keys=["img"], prob=1.0, mean=0.03, std=0.08)(sample)

    # Save output
    out_data = sample["img"].squeeze(0).numpy()  # (H, W, D)
    out_img  = nib.Nifti1Image(out_data, affine, header)
    nib.save(out_img, output_path)