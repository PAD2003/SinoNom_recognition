import numpy as np
from imgaug import augmenters as iaa

class ImgAugTransform:
  def __init__(self, p):
    sometimes = lambda aug: iaa.Sometimes(p, aug)

    self.aug = iaa.Sequential(
        iaa.SomeOf(
            (1, 5), 
            [
                # blur

                sometimes(iaa.OneOf([iaa.GaussianBlur(sigma=(0, 1.0)),
                                    iaa.MotionBlur(k=3)])),
                
                # color
                sometimes(iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)),
                sometimes(iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True)),
                sometimes(iaa.Invert(0.25, per_channel=0.5)),
                sometimes(iaa.Solarize(0.5, threshold=(32, 128))),
                sometimes(iaa.Dropout2d(p=0.5)),
                sometimes(iaa.Multiply((0.5, 1.5), per_channel=0.5)),
                sometimes(iaa.Add((-40, 40), per_channel=0.5)),

                sometimes(iaa.JpegCompression(compression=(5, 80))),
            ],
            random_order=True
        ),
        random_order=True
    )
      
  def __call__(self, img):
    img = np.array(img)
    img = self.aug.augment_image(img)
    img = np.array(img)
    return img
