import numpy as np
from datumaro.util.image import save_image

img = np.ones((5, 5, 3))
img0 = save_image('val/images/img0.jpg', img, create_dir=True)
img1 = save_image('train/images/img1.jpg', img, create_dir=True)
img2 = save_image('train/images/img2.jpg', img, create_dir=True)