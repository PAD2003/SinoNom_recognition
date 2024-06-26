import random
import os 
import json
import pickle

import rootutils
rootutils.setup_root(search_from=__file__, indicator='setup.py', pythonpath=True)
from src.data.components.aug.augment_v2 import *
from src.data.components.aug.base import thinning

# Static variables
random.seed(None)
MORPH_WINDOW_CROSS = [cv2.getStructuringElement(cv2.MORPH_CROSS, (i * 2 + 1, i * 2 + 1)) for i in range(1, 6)]
MORPH_WINDOW_ELLIPSE = [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i * 2 + 1, i * 2 + 1)) for i in range(1, 6)]

# Main augmentation module
class Augmenter:
    def __init__(self, 
                texture_path="", 
                bg_checkpoint="", 
                mask_checkpoint="",
                task="train"):
        self.texture = None 
        self.bg_checkpoint = bg_checkpoint
        self.mask_checkpoint = mask_checkpoint
        self.task = task

        self.prepare()
        # Since line pattern is gone, we gonna use this as alternative background
        self.load_texture(texture_path)
        
    

    def prepare(self):
        # In case the host has not build directory system for augmentation
        try:
            if not os.path.exists(self.mask_checkpoint):
                os.mkdir(self.mask_checkpoint)

            if not os.path.exists(self.bg_checkpoint):
                os.mkdir(self.bg_checkpoint)


            if not os.path.exists(os.path.join(self.mask_checkpoint, self.task)):
                os.mkdir(os.path.join(self.mask_checkpoint, self.task))

            if not os.path.exists(os.path.join(self.bg_checkpoint, self.task)):
                os.mkdir(os.path.join(self.bg_checkpoint, self.task))
        except: 
            print("Error Loading")
            return 
            

    def load_texture(self, path):
        # Load loaded texture if there exist one 
        if os.path.isdir(path):
            self.texture = []
            for fname in os.listdir(path):
                texture_path = os.path.join(path, fname)
                assert os.path.isfile(texture_path)
                
                # Is it all BGR or maybe RGB?
                self.texture.append(cv2.imread(texture_path))
        else:
            self.texture = cv2.imread(path)

    def bake_mask(self, image, path=None):
        if not path:
            mask = sharp_mask(image, 0.8, 0.8, line_erode=False)
            skeleton = thinning(mask)
            return  np.stack([mask, skeleton], axis=2)
        data = None
        pwd = os.path.join(self.mask_checkpoint, self.task)
        # Check mask folder availability
        create = False
        parts = path.split("/")
        # Sequentially reaching the file
        for part in parts:
            pwd = os.path.join(pwd, part)
            if '.' in part:
                if os.path.exists(pwd):
                    mask_img = cv2.imread(pwd) // 250
                    return mask_img[:, :, :2]
                else: 
                    mask = sharp_mask(image, 0.8, 0.8, line_erode=False)
                    word_skeleton = thinning(mask)
                    filling = np.zeros_like(mask)
                    mask_img = np.stack([mask, word_skeleton, filling], axis=2) 
                    cv2.imwrite(pwd, (mask_img * 255).astype(int).astype(np.uint8))
                    return mask_img[:, :, :2]
            else:
                if not os.path.exists(pwd):
                    os.mkdir(pwd)
                    create = True

        #  Mask extraction
        mask = sharp_mask(image, 0.8, 0.8, line_erode=False)
        #  Line pose extraction
        skeleton = thinning(mask)

        return  np.stack([mask, skeleton], axis=2)

    def bake_bg(self, image, path=None): 

        if not path: 
            thresh, bg_color, bg, freq = extract_background(image.copy())
            return {"thresh": thresh, "bg_color": bg_color, "bg": bg, "freq": freq}

        data = None
        pwd = os.path.join(self.bg_checkpoint, self.task)
        # Check background folder availability
        create = False
        parts = path.split("/")
        # Sequentially reaching the file
        for part in parts:
            pwd = os.path.join(pwd, part)
            if '.' in part:
                if os.path.exists(pwd):
                    data = np.load(pwd, allow_pickle=True)
                    return data
                else: 
                    thresh, bg_color, bg, freq = extract_background(image.copy())
                    np.savez(pwd, thresh=thresh, bg_color=bg_color, bg=bg, freq=freq)
                    return {"thresh": thresh, "bg_color": bg_color, "bg": bg, "freq": freq}
            else:
                if not os.path.exists(pwd):
                    os.mkdir(pwd)
                    create = True
        
        # Warning if such sample set does not exist 
        print("Miss some cases ?")

        thresh, bg_color, bg, freq = extract_background(image.copy())
        
        return {"thresh": thresh, "bg_color": bg_color, "bg": bg, "freq": freq} if data is None else data 
    

    # Randomly pick value in range
    @staticmethod
    def randomRange(range):
        if len(range) == 1:
            return range[0]
        else:
            return random.random() * (range[1] - range[0]) + range[0]
    
    # Choosing an option given probability of each option
    @staticmethod
    def randomPick(select, p):
        return np.random.choice(select, p)

    def transform_img(self, img,  
                            mask=None,
                            rotate=(-15, 15), 
                            shear_x=(-0.1, 0.1), 
                            shear_y=(-0.1, 0.1), 
                            opacity=(0.6, 1),
                            scale=(0.5, 1.2),
                            bx=(0, 3),
                            by=(0, 3),
                            ex=(0, 3),
                            ey=(0, 3),
                            logger=None,
                            debug= None,
                            keep_mask=False,
                            export=False,
                            fname=None,
                            borderMode='constant',
                            morph_size=(-2, 2.5),
                            morph=False,
                            inv=(0., 1.)):
        
        # Arguments configuration
        rotate = int(Augmenter.randomRange(rotate))
        shear_x = Augmenter.randomRange(shear_x)
        shear_y = Augmenter.randomRange(shear_y)
        alpha = borderMode != 'constant'

        inv = np.random.choice([True, False], p=inv) 

        scale = Augmenter.randomRange(scale)
        # Recenter image
        bx = int(Augmenter.randomRange(bx)) + int(max(1 - scale, 0) * img.shape[1] / 2)
        by = int(Augmenter.randomRange(by)) + int(max(1 - scale, 0) * img.shape[0] / 2)
        ex = int(Augmenter.randomRange(ex)) + int(max(1 - scale, 0) * img.shape[1] / 2)
        ey = int(Augmenter.randomRange(ey)) + int(max(1 - scale, 0) * img.shape[0] / 2)
        # Load background details
        if fname is not None:
            # exclude file extension
            
            fname = fname.split(".")[0]
            # read data
            data = self.bake_bg(img, path=fname + ".npz")
            thresh = data['thresh']
            bg_color = data['bg_color'].tolist()
            bg = data['bg']
            freq = data['freq']
            bg_color.append(0)  
            mask = self.bake_mask(img, path=fname + ".png")
        else:
            data = self.bake_bg(img, path=None)
            thresh = data['thresh']
            bg_color = data['bg_color'].tolist()
            bg = data['bg']
            freq = data['freq']
            bg_color.append(0)  
            mask = self.bake_mask(img, path=None)

        # Mask extraction
        if mask is None:
            mask = self.bake_mask(img, path=None)

        # Transform image
        transformed, transformed_mask, transform_matrix, alpha_mask = warp_transform(img, 
                                                                                    mask=mask, 
                                                                                    scale=scale,
                                                                                    angle=rotate, 
                                                                                    shear_x=shear_x, 
                                                                                    shear_y=shear_y, 
                                                                                    translate_x=bx,
                                                                                    translate_y=by,
                                                                                    pad_x=ex,
                                                                                    pad_y=ey,
                                                                                    bg_color=bg_color, 
                                                                                    export=export,
                                                                                    alpha=alpha,
                                                                                    border_replicate= borderMode == 'replicate')
        
        # Crop content and padding
        points = np.array(np.where(transformed_mask[:, :, 0] > 0)).T
        y, x, h, w = cv2.boundingRect(points)
        cropped = transformed[max(0,y-by):min(y+h+ey, transformed.shape[0]),max(0,x-bx):min(x+w+ex,transformed.shape[1])]
        cropped_mask = transformed_mask[max(0,y-by):min(y+h+ey, transformed.shape[0]),max(0,x-bx):min(x+w+ex,transformed.shape[1])]
        
        # Mask now has 2 channels
        skeleton = cropped_mask[:, :, 1].copy()
        cropped_mask = cropped_mask[:, :, 0]
        
        if alpha_mask is not None:
            # Create a canvas for full mask and morphology mask to operate on 
            alpha_mask = alpha_mask[max(0,y-by):min(y+h+ey, transformed.shape[0]),
                                    max(0,x-bx):min(x+w+ex, transformed.shape[1])]
            # Morphology augmentation
            if morph is True:
                morph_size = int(Augmenter.randomRange(morph_size) * scale) 
                
                if morph_size > 0: 
                    word_mask = cv2.dilate(skeleton, MORPH_WINDOW_CROSS[5 - morph_size])
                    alpha_mask[word_mask == 0] = 0
                elif morph_size < 0:
                    word_mask = cv2.dilate(skeleton, MORPH_WINDOW_ELLIPSE[int(- morph_size) ])
                    thicken_mask = np.stack([word_mask] * 3, axis = 2).astype(float)
                    thicken_mask = cv2.GaussianBlur(thicken_mask, (3, 3), 0)

            alpha_mask = cv2.GaussianBlur(alpha_mask, (3, 3), 0)
            alpha_mask = np.stack((alpha_mask, alpha_mask, alpha_mask), axis=2)

            
            # Initialize background 
            # Setting up ramdomnizer for background generation 
            randomnizer = np.random.default_rng()
            if np.max(freq) < 0.7: 
                background = randomnizer.choice(bg, 
                                                (cropped.shape[0], cropped.shape[1]), 
                                                axis=0, 
                                                p=freq).astype(float)
            else:
                assert self.texture is not None
                if isinstance(self.texture, list):
                    pos = np.random.randint(0, len(self.texture))
                    texture = self.texture[pos]
                else:
                    texture = self.texture
                y, x = randomnizer.integers(0, texture.shape[0] - cropped.shape[0]), randomnizer.integers(0, texture.shape[1] - cropped.shape[1])
                background = cv2.resize(cv2.GaussianBlur(texture, (3, 3), 0), 
                                        [cropped.shape[1], cropped.shape[0]],
                                        interpolation = cv2.INTER_LINEAR).astype(float)
            
            # Matching background & foreground color
            ratio = np.mean(background, axis=(0, 1)) / bg_color[:3]
            background /= ratio

            # Fusing image
            background = cv2.GaussianBlur(background, (3, 3), 0)
            alpha_mask *= Augmenter.randomRange(opacity)

            output = background * (1 - alpha_mask) + cropped * alpha_mask
            
            if morph is True and morph_size < 0:
                if inv is True: 
                    print("Inverse")
                    output[thicken_mask > 0] = 255 -  output[thicken_mask > 0]
                else: 
                    output *= 1 - thicken_mask * 0.8

            output = np.clip(output, 0, 255)
            output = output.astype(int).astype(np.uint8)
        else:
            output = cropped
        
        del background, alpha_mask, randomnizer, skeleton, points

        # This code is only for debugging.
        if isinstance(debug, str):
            fname = fname.split("/")[-1].split(".")[0]
            cv2.imwrite(os.path.join(debug, "original_{}.jpg").format(fname), img)
            # content mask
            log_mask = (np.stack([mask] * 3, axis=2) * 255).astype(np.uint8)
            log_t_mask = (np.stack([transformed_mask] * 3, axis=2) * 255).astype(np.uint8)
            log_c_mask = (np.stack([cropped_mask] * 3, axis=2) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(debug, "mask_{}.jpg").format(fname), log_mask)
            cv2.imwrite(os.path.join(debug, "transformed_mask_{}.jpg").format(fname), log_t_mask)
            cv2.imwrite(os.path.join(debug, "transformed_{}.jpg").format(fname), transformed) 
            cv2.imwrite(os.path.join(debug, "alpha_mask_{}.jpg").format(fname), alpha_mask * 255)
            cv2.imwrite(os.path.join(debug, "background_{}.jpg").format(fname), background)
            cv2.imwrite(os.path.join(debug, "cropped_{}.jpg").format(fname), cropped)
            cv2.imwrite(os.path.join(debug, "cropped_mask_{}.jpg").format(fname), log_c_mask)
            cv2.imwrite(os.path.join(debug, "background_alpha_{}.jpg").format(fname), background * (1 - alpha_mask))
            cv2.imwrite(os.path.join(debug, "content_alpha_{}.jpg").format(fname), cropped * alpha_mask)

        return output.astype(np.uint8), cropped_mask, transform_matrix 
            
    def full_augment(self, 
                    img,
                    choice,
                    fname=None, 
                    borderMode='native'):
        """ Automatic apply transformation on image with given probabilities of 4 choices:
            - Only crop ROI 
            - Basic Transform
            - Morphology Transform 
            - Inverse Morphology Transform
        """
        pose = np.random.choice((1, 2, 3, 4), p=choice)
        
        # Specific augment for each choice
        if pose == 1:
            return  self.transform_img( img, 
                                        fname=fname, 
                                        scale=(1., ),
                                        rotate=(0, ), 
                                        shear_x=(0, ), 
                                        shear_y=(0, ), 
                                        opacity=(1, ),
                                        bx=(1, ),
                                        by=(1, ),
                                        ex=(1, ),
                                        ey=(1, ),
                                        borderMode=borderMode)[0]
        elif pose == 2:
            return  self.transform_img( img, 
                                        fname=fname,
                                        scale=(1., ),  
                                        rotate=(0, ), 
                                        shear_x=(0, ), 
                                        shear_y=(0, ), 
                                        morph_size=(-3, 3.5),
                                        morph=True,
                                        borderMode=borderMode)[0]
        elif pose == 3:
            return  self.transform_img( img, 
                                        fname=fname,
                                        morph_size=(-3, 3.5),
                                        morph=True,
                                        borderMode=borderMode)[0]
        else: 
            return  self.transform_img( img, 
                                        fname=fname,
                                        morph_size=(-3, 3.5),
                                        morph=True,
                                        borderMode=borderMode,
                                        inv=(0.2, 0.8))[0]
    
    # Subprocess augmentation
    def threaded_augment(self, 
                            img, 
                            choice, 
                            fname=None, 
                            borderMode='native'):
        pass
    

    # Offline upsampling process 
    @staticmethod
    def process(img, 
                label, 
                sample, 
                p=None, 
                fname=None, 
                borderMode='constant'):
        output = []
        for i in range(sample):
            if p is None:
                output.append(Augmenter.full_augment(img, label, fname=fname, borderMode=borderMode))
            else:
                output.append(Augmenter.full_augment(img, label, choice=p, fname=fname, borderMode=borderMode))

        return output 
    
if __name__ =="__main__":
    def delete_contents_of_folder(folder_path):
        try:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            print(f"Deleted all contents of the folder: {folder_path}")
        except Exception as e:
            print(f"Error deleting contents of the folder: {folder_path}")

    augmenter = Augmenter(  texture_path="/work/hpc/potato/SinoNom_recognition/data/augment/texture/", 
                            bg_checkpoint="/work/hpc/potato/SinoNom_recognition/data/augment/background/",
                            mask_checkpoint="/work/hpc/potato/SinoNom_recognition/data/augment/mask/",
                            task="train")
    img_dir = "./data/wb_recognition_dataset/train/"
    # data/wb_recognition_dataset/train/1/nlvnpf-0137-01-022_crop_23.jpg
    filename = "0/6_0.png"
    label, fname = filename.split("/")
    img = cv2.imread(img_dir + filename)
    label = int(label)
    delete_contents_of_folder("/data/hpc/potato/sinonom/data/debug")
    # transformed, _, _ = augmenter.transform_img(img, 
    #                                             fname=filename,
    #                                             debug="/data/hpc/potato/sinonom/data/debug", 
    #                                             borderMode='native',
    #                                             scale=(0.7, ),
    #                                             morph_size=(4.,), 
    #                                             morph=True,
    #                                             inv=(1., 0.))
    # img_label = "/work/hpc/firedogs/data_/train_gt.txt"
    # parts = ["train_img_88652.png", "xeva"]
    # img = cv2.imread(img_dir + parts[0])
    # print(parts[0])
    # filename = parts[0].split(".")[0]
    # processed = augmenter.process(img, 
    #                               parts[1], 
    #                               1, 
    #                               p=(0, 1, 0, 0), 
    #                               fname=filename,
    #                               borderMode='replicate')[0]
    # print(processed.shape, processed.dtype)
    # cv2.imwrite("/data/hpc/potato/sinonom/data/debug/{}.jpg".format(fname.split(".")[0]), transformed)
    # import time 
    i = 0
    # current = time.time()
    # start = current
    with open("./data/manifest_full.json", "r") as file: 
        dataset = json.load(file)['train']
    for key in dataset.keys(): 
        samples = dataset[key]
        for sample in dataset[key]:
            img = cv2.imread(img_dir + sample)
            # print(sample)
            i += 1
            fname = sample.split(".")[0]
            # current = time.time()
            # augmenter.bake_bg(img, path=fname + ".npz")
            # augmenter.bake_mask(img, path=fname + ".png")
            transformed = augmenter.full_augment(img, [0, 0, 0, 1], fname=sample, borderMode='native')
            # print(i, "-th image shape:", time.time() - current)
    # print("Done all in", time.time() - start)
        