import cv2
import numpy as np
import albumentations as A
from albumentations import CropNonEmptyMaskIfExists
from albumentations.pytorch import ToTensorV2


class LuminosityStandardizer(object):

    @staticmethod
    def standardize(I, percentile=95):
        """
        Transform image I to standard brightness.
        Modifies the luminosity channel such that a fixed percentile is saturated.
        :param I: Image uint8 RGB.
        :param percentile: Percentile for luminosity saturation. At least (100 - percentile)% of pixels should be fully luminous (white).
        :return: Image uint8 RGB with standardized brightness.
        """
        I_LAB = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
        L_float = I_LAB[:, :, 0].astype(float)
        p = np.percentile(L_float, percentile)
        I_LAB[:, :, 0] = np.clip(255 * L_float / p, 0, 255).astype(np.uint8)
        I = cv2.cvtColor(I_LAB, cv2.COLOR_LAB2RGB)
        return I


def augment_image(image, domain_pixel_size, target_pixel_size, domain_tissue_thickness, target_tissue_thickness, alpha=0.15):
    
    """
    Visualize raw and augmented images 
    
    Parameters
    ----------
    image (numpy.ndarray of shape (height, width, 3)): Image array
    domain_pixel_size (float): Pixel size of the domain images in micrometers
    target_pixel_size (float): Pixel size of the target images in micrometers
    domain_tissue_thickness (float): Tissue thickness of the domain images in micrometers
    target_tissue_thickness (float): Tissue thickness of the target images in micrometers
    alpha (float): Multiplier to control saturation and value scale
    """
    
    # Augment tissue thickness
    tissue_thickness_scale_factor = target_tissue_thickness - domain_tissue_thickness
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    image_hsv[:, :, 1] *= (1 + (alpha * tissue_thickness_scale_factor))
    image_hsv[:, :, 2] *= (1 - (alpha * tissue_thickness_scale_factor))
    image_hsv = image_hsv.astype(np.uint8)
    image_scaled = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    
    # Standardize luminosity
    image_scaled = LuminosityStandardizer.standardize(image_scaled)

    # Augment pixel size
    pixel_size_scale_factor = domain_pixel_size / target_pixel_size
    image_resized = cv2.resize(
        image_scaled,
        dsize=None,
        fx=pixel_size_scale_factor,
        fy=pixel_size_scale_factor,
        interpolation=cv2.INTER_CUBIC
    )
    image_resized = cv2.resize(
        image_resized,
        dsize=(
            image.shape[1],
            image.shape[0]
        ),
        interpolation=cv2.INTER_CUBIC
    )
    
    # Standardize luminosity
    image = LuminosityStandardizer.standardize(image)
    image_augmented = LuminosityStandardizer.standardize(image_resized)
    
    return image, image_augmented


class ColorHsvShift():
    def __init__(self, thres_s=10, thres_v=230, hblue=135, hpink=165, hbr=20, ds=85, dv=40, **kwargs):
        self.thres_s = thres_s
        self.thres_v = thres_v
        self.hblue = hblue
        self.hpink = hpink
        self.hbr = hbr
        self.ds = ds
        self.dv = dv
        
    def __call__(self, img_rgb, **kwargs):
        # (pink - blue)/(blue - brown)
        r = (self.hpink-self.hblue)/(self.hblue-self.hbr)

        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        mask_bg = ~((img_hsv[:,:,1]>self.thres_s) & (img_hsv[:,:,2]<self.thres_v))

        # bg
        _img_bg = img_rgb*np.stack([mask_bg]*3, axis=2)
        img_bg = np.where(_img_bg == 0, 255, img_rgb)

        # main
        _img_main = img_rgb*np.stack([~mask_bg]*3, axis=2)
        img_main = np.where(_img_main == 0, 255, img_rgb)

        img_main_hsv = cv2.cvtColor(img_main, cv2.COLOR_RGB2HSV).astype(np.float64)
        img_main_hsv[:, :, 0] = (self.hblue - img_main_hsv[:, :, 0])*r + self.hblue
        img_main_hsv[:, :, 1] = img_main_hsv[:, :, 1] + self.ds
        img_main_hsv[:, :, 2] = np.clip(img_main_hsv[:, :, 2] + self.dv, 0, 255)
        img_main_rgb = cv2.cvtColor(img_main_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        _img_main_rgb = img_main_rgb*np.stack([~mask_bg]*3, axis=2)
        img_main_rgb = np.where(_img_main_rgb == 0, 255, img_main_rgb)

        img_rgb_new = img_main_rgb + img_bg
        return img_rgb_new

    
def invert_color(image, **kwargs):
    image = image.astype(np.uint8)
    return ~image


class InvertColor():
    def __init__(self, thres_s=10, thres_v=230, **kwargs):
        self.thres_s = thres_s
        self.thres_v = thres_v
        
    def __call__(self, img_rgb, **kwargs):
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        mask_bg = ~((img_hsv[:,:,1]>self.thres_s) & (img_hsv[:,:,2]<self.thres_v))

        # bg
        _img_bg = img_rgb*np.stack([mask_bg]*3, axis=2)
        img_bg = np.where(_img_bg == 0, 255, img_rgb).astype(np.uint8)

        # main
        _img_main = img_rgb*np.stack([~mask_bg]*3, axis=2)
        img_main = np.where(_img_main == 0, 255, img_rgb).astype(np.uint8)

        image = ~img_main + img_bg
        return image


class TrainAug:
    def __init__(self, height, width):
        self.height = height
        self.width = width
    
    def aug(self, organ=None):
        basic_aug = self._default_aug()
        if organ == 'spleen':
            basic_aug = self._spleen_aug()
        if organ == 'kidney':
            basic_aug = self._kidney_aug()
        if organ == 'largeintestine':
            basic_aug = self._largeintestine_aug()
        if organ == 'prostate':
            basic_aug = self._prostate_aug()
        
        common_aug = self._common_aug()
        basic_aug.extend(common_aug)
        basic_aug.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0))
        basic_aug.append(ToTensorV2())
        
        return A.Compose(basic_aug)
        
    def _common_aug(self):
        common_aug = [
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0, 3), p=0.3),
            A.OneOf([
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
            ], p=0.5),
            A.Rotate(limit=(-90, 90), border_mode=cv2.BORDER_DEFAULT, mask_value=0, p=0.5),
#             A.OneOf([
#                 A.OpticalDistortion(p=0.3),
#                 A.GridDistortion(p=.1),
#             ], p=0.3),
        ]
        return common_aug
        
    def _default_aug(self):
        default_aug = [
            A.Resize(height=self.height, width=self.width, interpolation=cv2.INTER_LINEAR, p=1),
            A.OneOf([
#                 A.Compose([
#                     A.Lambda(name='stain_hsv', image=ColorHsvShift(ds=70, dv=20), p=1),
#                     A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=15, val_shift_limit=5, p=0.7),
#                 ], p=0),
#                 A.Compose([
#                     A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=25, val_shift_limit=5, p=0.5),
#                     A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.4), contrast_limit=(0.7, 1), brightness_by_max=True, p=0.5),
#                 ], p=1),
                A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.3), contrast_limit=(0.7, 1), brightness_by_max=True, p=0.5), # 0.3
                A.HueSaturationValue(hue_shift_limit=150, sat_shift_limit=[-10, 35], val_shift_limit=5, p=0.5),
                A.ChannelShuffle(p=0.5),
                A.RandomToneCurve(scale=0.5, p=0.5),
                A.CLAHE(clip_limit=(1,4), p=0.5),
            ], p=0.5),
        ]
        return default_aug
    
    def _spleen_aug(self):
        spleen_aug = [
            # resize
            A.Compose([
                A.OneOf([
                    A.CropNonEmptyMaskIfExists(height=1024, width=1024, p=0.5),
                    A.CropAndPad(percent=(-0.15, 0.4), pad_mode=4, p=0.5),
                ], p=0.3), # 0.2
                A.Resize(height=self.height, width=self.width, interpolation=cv2.INTER_LINEAR, p=1),
            ], p=1),
            # color
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.4), contrast_limit=(0.7, 1), brightness_by_max=True, p=0.5), # 0.3
                A.Lambda(name="invert_color", image=InvertColor(), p=0.5),
                A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=[-10, 35], val_shift_limit=5, p=0.5),
                A.ChannelShuffle(p=0.5),
#                 A.RandomToneCurve(scale=0.5, p=0.5),
            ], p=0.9),
        ]
        return spleen_aug
    
    def _kidney_aug(self):
        kidney_aug = [
            # resize
            A.Compose([
                A.OneOf([
                    A.CropNonEmptyMaskIfExists(height=1536, width=1536, p=0.25),
                    A.CropNonEmptyMaskIfExists(height=1024, width=1024, p=0.25),
                    A.CropNonEmptyMaskIfExists(height=768, width=768, p=0.25),
                    A.CropAndPad(percent=(0.1, 0.4), pad_mode=4, p=0.25),
                ], p=0.5),
                A.Resize(height=self.height, width=self.width, interpolation=cv2.INTER_LINEAR, p=1),
            ], p=1),
            # color
            A.OneOf([
#                 A.Lambda(name='stain_hsv', image=ColorHsvShift(ds=70, dv=20), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.4), contrast_limit=(0.7, 1), brightness_by_max=True, p=0.5), # 0.3
#                 A.Lambda(name="invert_color", image=InvertColor(), p=0.5),
                A.HueSaturationValue(hue_shift_limit=150, sat_shift_limit=[-10, 35], val_shift_limit=5, p=0.5),
                A.ChannelShuffle(p=0.5),
                A.RandomToneCurve(scale=0.5, p=0.5),
                A.CLAHE(clip_limit=(1, 3), p=0.5),
            ], p=0.9),
        ]
        return kidney_aug
    
    def _largeintestine_aug(self):
        largeintestine_aug = [
            # resize
            A.Compose([
                A.OneOf([
                    A.CropNonEmptyMaskIfExists(height=1536, width=1536, p=0.25),
                    A.CropNonEmptyMaskIfExists(height=1024, width=1024, p=0.25),
                    A.CropNonEmptyMaskIfExists(height=512, width=512, p=0.25),
                    A.CropAndPad(percent=(0.1, 0.45), pad_mode=4, p=0.25),
                ], p=0.5),
                A.Resize(height=self.height, width=self.width, interpolation=cv2.INTER_LINEAR, p=1),
            ], p=1),
            # color
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.3), contrast_limit=(0.7, 1), brightness_by_max=True, p=0.5), # 0.3
                A.HueSaturationValue(hue_shift_limit=150, sat_shift_limit=[-10, 40], val_shift_limit=5, p=0.5),
                A.ChannelShuffle(p=0.5),
                A.RandomToneCurve(scale=0.5, p=0.5),
            ], p=0.9),
            A.CLAHE(clip_limit=(1,7), p=0.5),
        ]
        return largeintestine_aug
    
    def _prostate_aug(self):
        prostate_aug = [
            # resize
            A.Compose([
                A.OneOf([
                    A.CropNonEmptyMaskIfExists(height=1536, width=1536, p=0.5),
                    A.CropNonEmptyMaskIfExists(height=1024, width=1024, p=0.5),
                    A.CropNonEmptyMaskIfExists(height=512, width=512, p=0.5),
                    A.CropAndPad(percent=(0.1, 0.4), pad_mode=4, p=0.5),
                ], p=0.4),
                A.Resize(height=self.height, width=self.width, interpolation=cv2.INTER_LINEAR, p=1),
            ], p=1),
            # color
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.4), contrast_limit=(0.7, 1), brightness_by_max=True, p=0.5), # 0.3
                A.HueSaturationValue(hue_shift_limit=150, sat_shift_limit=[-10, 40], val_shift_limit=5, p=0.5),
                A.ChannelShuffle(p=0.5),
                A.CLAHE(clip_limit=(1,7), p=0.5),
                A.RandomToneCurve(scale=0.5, p=0.5),
            ], p=0.9),  # 0.5
        ]
        return prostate_aug

    
############################################################################################
# spleen augmentation
def spleen_transform(h, w):
    return A.OneOf([
        A.Compose([
            A.OneOf([
                A.Compose([
                    A.CropNonEmptyMaskIfExists(height=1536, width=1536, p=1),
                    A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=1),
                ], p=0.35),
                A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=0.65),
            ], p=1),
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0, 2), p=0.3),
            A.OneOf([
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.Rotate(limit=(-60, 60), border_mode=cv2.BORDER_DEFAULT, mask_value=0, p=0.5),
                A.RandomRotate90(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
            ], p=0.3),
            A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=[-5, 35], val_shift_limit=[-15, 20], p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.4), contrast_limit=(0.7, 1), brightness_by_max=True, p=0.5),
            A.CLAHE(clip_limit=2, p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ], p=0.95),
        A.Compose([
            A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ], p=0.05)
    ], p=1)


# kidney transform
def kidney_transform(h, w):
    return A.OneOf([
        A.Compose([
            A.OneOf([
#                 A.Compose([
#                     A.CropNonEmptyMaskIfExists(height=1536, width=1536, p=1),
#                     A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=1),
#                 ], p=0.1),
                A.Compose([
                    A.CropNonEmptyMaskIfExists(height=1024, width=1024, p=1),
                    A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=1),
                ], p=0.2),
#                 A.Compose([
#                     A.CropNonEmptyMaskIfExists(height=768, width=768, p=1),
#                     A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=1),
#                 ], p=0.2),
                A.Compose([
                    A.CropNonEmptyMaskIfExists(height=512, width=512, p=1),
                    A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=1),
                ], p=0.15),
                A.Compose([
                    A.CropNonEmptyMaskIfExists(height=256, width=256, p=1),
                    A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=1),
                ], p=0.15),
                A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=0.6),
            ], p=1),
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0, 2), p=0.3),
            A.OneOf([
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.Rotate(limit=(-60, 60), border_mode=cv2.BORDER_DEFAULT, mask_value=0, p=0.5),
                A.RandomRotate90(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
            ], p=0.3),
#             A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=[-10, 45], val_shift_limit=[-35, 10], p=0.6),
            A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=[-10, 45], val_shift_limit=5, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.3), contrast_limit=(0.7, 1), brightness_by_max=True, p=0.4),
            A.CLAHE(clip_limit=2, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ], p=0.95),
        A.Compose([
            A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ], p=0.05)
    ], p=1)


# largeintestine
def largeintestine_aug(h, w):
    return A.OneOf([
        A.Compose([
            A.OneOf([
                A.Compose([
                    A.CropNonEmptyMaskIfExists(height=1536, width=1536, p=1),
                    A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=1),
                ], p=0.2),
                A.Compose([
                    A.CropNonEmptyMaskIfExists(height=1024, width=1024, p=1),
                    A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=1),
                ], p=0.2),
                A.Compose([
                    A.CropNonEmptyMaskIfExists(height=768, width=768, p=1),
                    A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=1),
                ], p=0.2),
                A.Compose([
                    A.CropNonEmptyMaskIfExists(height=512, width=512, p=1),
                    A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=1),
                ], p=0.1),
                A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=0.3),
            ], p=1),
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0, 2), p=0.3),
            A.OneOf([
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.Rotate(limit=(-60, 60), border_mode=cv2.BORDER_DEFAULT, mask_value=0, p=0.5),
                A.RandomRotate90(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
            ], p=0.3),
            A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=[-15, 35], val_shift_limit=[-20, 15], p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.4), contrast_limit=(0.7, 1), brightness_by_max=True, p=0.5),
            A.CLAHE(clip_limit=3, p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ], p=0.95),
        A.Compose([
            A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ], p=0.05)
    ], p=1)