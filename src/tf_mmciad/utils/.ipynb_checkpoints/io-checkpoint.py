import numpy as np
import os
from os.path import join
from glob import glob
from skimage.io import imread, imsave
from keras.utils import to_categorical

def read_samples(path, colors, prefix='train', n_samples=None):
    """ Read all sample slides and subdivide into square tiles.
    
    Parameters
    ----------
    path : str
        Identifies the root data folder containing the whole slide images, e.g.
        .../path/train-N8b.tif
        .../path/gt/train-N8b.png
    colors : np.ndarray
        n x 3 array containing the RGB colors for each of the n classes
    prefix : str
        Filename prefix, e.g.
        'train', 'test', 'val'
    n_samples : int or None
        Number of sample tiles to extract from each whole slide image.
        
        If n_samples == None: the images will be subdivided into a regular grid
        with overlapping tiles.
        
        if n_samples > 0: n_slides*n_samples tiles will be created by creating
        coordinate paris at random and extracting the tiles from these locations.
    """
    size = (208, 208)
    X_files = sorted(glob(join(path, prefix+"*.tif")), key=str.lower)
    Y_files = sorted(glob(join(path,'gt', prefix+"*.png")), key=str.lower)
    num_img = len(X_files)
    assert(len(X_files) == len(Y_files))
    X_samples = [np.asarray(imread(X_files[i]).astype('float')/255.) for i in range(num_img)]
    means, stds = calculate_stats(X_samples)
    for i in range(num_img):
        for c in range(3):
            X_samples[i][...,c] = (X_samples[i][...,c] - means[c])/stds[c]
    Y_samples = np.asarray([imread(Y_files[i])[:,:,:3] for i in range(num_img)])
    Y_class = [np.expand_dims(np.zeros(Y_samples[i].shape[:2]),axis=-1) for i in range(num_img)]
    for i in range(num_img):
        for cls in range(colors.shape[0]):
            color = colors[cls,:]
            Y_class[i] += np.expand_dims(np.logical_and.reduce(Y_samples[i] == color,axis=-1)*cls, axis=-1)
        Y_class[i] = to_categorical(Y_class[i], num_classes=num_cls)
    X = []
    Y = []
    if n_samples is not None:
        for i in range(num_img):
            X_, Y_ = Y_samples[i].shape[:2]
            max_shape = np.array((np.asarray(Y_samples[i].shape[:2]) - np.array(size))/200)
            points = np.c_[np.random.randint(max_shape[0],size=n_samples),
                           np.random.randint(max_shape[1],size=n_samples)]*200
            for n in range(n_samples):
                x, y = points[n,:]
                X.append(X_samples[i][x:x+size[0], y:y+size[1],:])
                Y.append(  Y_class[i][x:x+size[0], y:y+size[1],:])
    else:
        for i in range(num_img):
            X_, Y_ = Y_samples[i].shape[:2]
            px, py = np.mgrid[0:X_:160, 0:Y_:160]
            points = np.c_[px.ravel(), py.ravel()]
            pr = points.shape[0]
            for n in range(pr):
                x, y = points[n,:]
                res_x = X_samples[i][x:x+size[0],y:y+size[1],:]
                res_y = Y_class[i][x:x+size[0],y:y+size[1],:]
                change = False
                if (x+size[0]) > X_:
                    x = X_ - size[0]
                    change = True
                if (y+size[1]) > Y_:
                    y = Y_ - size[1]
                    change = True
                if change:
                    res_x = X_samples[i][x:x+size[0],y:y+size[1],:]
                    res_y = Y_class[i][x:x+size[0],y:y+size[1],:]
                X.append(res_x)
                Y.append(res_y)
    X = np.asarray(X, dtype='float')
    Y = np.asarray(Y, dtype='uint8')
    return X, Y, means, stds, points


def create_samples(path, bg_color, ignore_color, prefix='train'):
    """ Read all sample slides and subdivide into square tiles. The resulting tiles are saved
    as .tif files in corresponding directories.
    
    Parameters
    ----------
    path : str
        Identifies the root data folder containing the whole slide images, e.g.
        .../path/train-N8b.tif
        .../path/gt/train-N8b.png
    bg_color : list of ints
        3-element list containing the RGB color code for the background class
    ignore_color : list of ints
        3-element list containing the RGB color code for the ignore class
    prefix : str
        Filename prefix, e.g.
        'train', 'test', 'val'
    """
    size = (208, 208)
    X_files = sorted(glob(join(path, prefix+"*.tif")), key=str.lower)
    Y_files = sorted(glob(join(path,'gt', prefix+"*.png")), key=str.lower)
    num_img = len(X_files)
    X_samples = np.asarray([imread(X_files[i]).astype('float')/255. for i in range(num_img)])
    Y_samples = np.asarray([imread(Y_files[i])[:,:,:3] for i in range(num_img)])
    X = [] #np.zeros(shape=(n_samples*num_img, size[0], size[1], 3), dtype="float")
    Y = [] #np.zeros(shape=(n_samples*num_img, size[0], size[1], num_cls), dtype="uint8")
    for i in range(num_img):
        X_, Y_ = Y_samples[i].shape[:2]
        px, py = np.mgrid[0:X_:160, 0:Y_:160]
        points = np.c_[px.ravel(), py.ravel()]
        pr = points.shape[0]
        for n in range(pr):
            x, y = points[n,:]
            res_x = X_samples[i][x:x+size[0],y:y+size[1],:]
            res_y = Y_samples[i][x:x+size[0],y:y+size[1],:]
            change = False
            if (x+size[0]) > X_:
                x = X_ - size[0]
                change = True
            if (y+size[1]) > Y_:
                y = Y_ - size[1]
                change = True
            if change:
                res_x = X_samples[i][x:x+size[0],y:y+size[1],:]
                res_y = Y_samples[i][x:x+size[0],y:y+size[1],:]
            # Check if res_y contains any pixels with the ignore label
            if not check_class(res_y, ignore_color, probability=1):
                # Check if res_y contains enough pixels with background label
                if not check_class(res_y, bg_color):
                    X.append(res_x)
                    Y.append(res_y)
    X = np.asarray(X, dtype='float')
    Y = np.asarray(Y, dtype='uint8')
    for i in range(len(X)): ## Check_contrast will be available from version 0.15
        imsave(path+prefix+'/X_{}.tif'.format(i), X[i, ...], check_contrast=False)
        imsave(path+prefix+'/gt/X_{}.tif'.format(i), Y[i, ...], check_contrast=False)


def check_class(segmap, class_color, probability=0.9, threshold=0.9):
    """ Filter input based on how much of a given class is present in the input image.
    Returns True if image should be filtered.
    
    Parameters
    ----------
    segmap : array-like
        The input image to be filtered
    class_color : list of ints
        list of length 3 with RGB color code matching the class color checking against
    probability : float
        Probability of image being filtered if above a certain threshold (optional, defaults to 0.9)
    threshold : float
        Threshold for fraction of segmap allowed to have class_color before being a candidate for
        filtering (optional, defaults to 0.9)
    """
    segmap = np.logical_and.reduce(segmap == class_color, axis=-1)
    if segmap.sum() >= segmap.size * threshold:
        return np.random.rand() > 1-probability
    return False

def load_slide(path, prefix='N8b', m=[0.,0.,0.], s=[1.,1.,1.], load_gt=True):
    ftype = "*.tif"
    X_files = glob(join(path, prefix+ftype))
    X = np.asarray([imread(X_files[i]).astype('float')/255. for i in range(len(X_files))])
    for i in range(len(X)):
        for c in range(3):
            X[i][ ..., c] = (X[i][..., c] - m[c])/s[c]
    if load_gt:
        Y_files = glob(join(path, prefix, "*.png"))
        Y_samples = imread(Y_files[0])[:,:,:3]
        Y_class = np.expand_dims(np.zeros(Y_samples.shape[:2]),axis=-1)
        for cls in range(num_cls):
            color = colorvec[cls,:]
            Y_class += np.expand_dims(np.logical_and.reduce(Y_samples == color,axis=-1)*cls,axis=-1)
        Y = to_categorical(Y_class, num_classes=num_cls)
        return X, Y
    return X