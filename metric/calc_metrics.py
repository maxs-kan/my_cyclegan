import os
from glob import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import imageio
from scipy import signal
import tqdm
from glob import glob
import multiprocessing
import functools

def fspecial_gauss(size, sigma):
    """
        Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def _ssim(img1, img2, L=1):
    """
        L = 1 for depth maps in [0, 1] range
    
        Return the Structural Similarity Map corresponding to input images img1 
        and img2 (images are assumed to be uint8)
        This function attempts to mimic precisely the functionality of ssim.m a 
        MATLAB provided by the author's of SSIM
        https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    return np.mean(((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)))
    

def _rmse(pred, target):
    return np.sqrt(np.mean(np.power(target - pred, 2)))

def _psnr(pred, target, max_value=1):
    mse = np.mean(np.power(target - pred, 2))
    if mse == 0:
        raise NotImplementedError('Same img')
    else:
        return 20. * np.log10(max_value) - 10 * np.log10(mse)

def _mae(pred, target):
    return np.mean(np.abs(target - pred))

def _rmse_h(pred, target, hole_map):
    if np.any(hole_map):
        diff2 = np.square(pred[hole_map] - target[hole_map])
        return np.sqrt(np.mean(diff2))
    else:
        return np.nan

def _rmse_d(pred, target, hole_map):
    diff2 = np.square(pred[~hole_map] - target[~hole_map])
    return np.sqrt(np.mean(diff2))

def calc_rmse(pred, target, hole_map):
    return _rmse(pred, target)

def calc_psnr(pred, target, hole_map):
    return _psnr(pred/8000, target/8000) # TODO: pass args.max_depth

def calc_mae(pred, target, hole_map):
    return _mae(pred, target)

def calc_ssim(pred, target, hole_map):
    return _ssim(pred/8000, target/8000) # TODO: pass args.max_depth

metric_by_name = {
    "mae": calc_mae, #_mae,
    "rmse": calc_rmse, #_rmse,
    "psnr": calc_psnr, #_psnr,
    "ssim": calc_ssim, #_ssim,
    "rmse_h": _rmse_h,
    "rmse_d": _rmse_d
}

def calc_metrics(pred, target, hole_map, metric_names):
    out = {}
    
    for metric_name in metric_names:
        metric_func = metric_by_name[metric_name]
        out[metric_name] = metric_func(pred, target, hole_map)

    return out

def calc_metrics_for_path(path_args, metric_names):
    input_path, pred_path, target_path = path_args
    input_orig = imageio.imread(input_path).astype(np.float64)
    pred = imageio.imread(pred_path).astype(np.float64).clip(0, 8000) # TODO: pass args.max_depth
    target = imageio.imread(target_path).astype(np.float64).clip(0, 8000) # TODO: pass args.max_depth
    hole_map = input_orig < 10
    
    return calc_metrics(pred, target, hole_map, metric_names)

def calculate_given_paths(input_dir_path, pred_dir_path, target_dir_path, metric_names, n_cpus):
    input_names = sorted(glob(os.path.join(input_dir_path,'*.png')))
    pred_names = sorted(glob(os.path.join(pred_dir_path,'*.png')))
    target_names = sorted(glob(os.path.join(target_dir_path,'*.png')))

    _calc_metrics_for_path = functools.partial(calc_metrics_for_path, metric_names=metric_names)
    paths = zip(input_names, pred_names, target_names)
    with multiprocessing.Pool(n_cpus) as p:
        res = list(tqdm.tqdm(p.imap(func=_calc_metrics_for_path, iterable=paths), total=len(input_names)))

    out = {}
    for metric_name in metric_names:
        out[metric_name] = np.asarray([x[metric_name] for x in res])
        out[metric_name] = np.mean(out[metric_name][~np.isnan(out[metric_name])])
    print(out)
    return out

if __name__ == '__main__':
    print('start')
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path', type=str, help='Path to the input images')
    parser.add_argument('--pred_path', type=str, help='Path to the generated images')
    parser.add_argument('--target_path', type=str, help='Path to the target images')
    parser.add_argument('--max_depth', type=int, default=8000, help='Maximum depth value')
    parser.add_argument('--n_cpus', type=int, default=10, help='Number of cpu cores to use')
    args = parser.parse_args()

    list_of_metrics = ["rmse", "mae", "rmse_h", "rmse_d", "psnr", "ssim"]
    calculate_given_paths(args.input_path, args.pred_path, args.target_path, list_of_metrics, args.n_cpus)