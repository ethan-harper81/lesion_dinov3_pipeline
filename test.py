import subprocess
import os
import sys
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import pandas as pd
import numpy as np
import pprint
import matplotlib.pyplot as plt

import cv2
import numpy as np
from lesion_dinov3.utils import extract_peaks

from inference.infer_validation_files import infer_validation_files, create_eval

from evaluate_points import match_points
def save_eval_image():
    csv = "validation_outputs/multiscale_diamter_no-final_prior/wide_00_001.jpg__candidates.csv"

    df = pd.read_csv(csv)

    diameter = True

    #amin_prescan_images_1_front_wide_00_001.jpg
    im = "C:/Users/davoo/Development/data_argenziano/settimo lavoro/Foto Cliniche/P D - 1966/2/19348__11092024 152508165404.jpg"


    #patient
    im = "C:/Users/davoo/Development/LumoLesionClassifier/patient_images/front/wide_00_001.jpg"

    og_points = [[5328.88,5875.84,37.36],[5162.08,5547.44,36],[5495.68,5428.72,33.52],[4857.28,5441.76,26.64],[5086.96,4971.04,38.8],[5484.32,4702.96,46.96],[5545.44,4520.4,35.84],[5390.88,4626.96,33.36],[4999.6,4441.76,27.2],[5202.24,4323.04,43.12],[5546.32,4260.16,32.24],[4985.6,4104.4,32],[4980,3877.2,50],[4935.2,3591.6,36.24],[5452,3536.4,35.12],[5385.6,3522.8,48.8],[5202.4,3343.6,40],[5292,3074.8,40],[2662.16,2669.64,39.2],[4962.4,2853.2,43.12],[5152.8,2798,35.76],[5134.4,2267.6,44.56],[4966.4,1707.6,41.6],[4547.2,1827.6,39.6],[4620.8,1989.36,29.12],[4653.12,2012,50.56],[4904.96,1996.64,36.32],[5307.46,1594.46,26.72],[4296.32,1642.24,44.88],[4075.12,1402.48,64.32],[4454.56,1080.4,58.32],[4356,1230.56,39.28],[4275.28,1344.4,39.12],[4607.92,1274.96,39.6],[5014.8,1735.92,37.52],[95.52,4945.26,154.44],[4934.88,1527.6,26.08],[4480.4,863.2,86.4],[4446.56,676.96,48.56],[4188.32,430.08,60],[3509.73,648.45,35.4],[3587.28,947.92,34.48],[1575.06,2914.75,62.92],[3812.16,256.64,62.8],[3582.88,105.36,58.72],[3546.32,326.56,63.52],[3466.64,412.72,44.24],[3004,365.6,45.36],[3994.24,309.44,37.92],[3660.16,820.88,34.88],[2767.76,112.96,49.28],[2734.56,257.04,46.96],[1819.12,850.64,47.52],[1975.6,762.08,39.36],[1571.2,789.68,35.04],[2010.08,213.04,45.28],[1665.68,246.48,44.56],[1625.28,251.36,32],[1793.6,789.68,49.2],[1839.2,53.2,51.6],[1680,390.8,37.76],[1704,481.2,40.8],[2155.2,584.4,37.36],[1404,100.4,36.56],[906.72,889.84,53.6],[808.64,1090.4,57.84],[628.56,1182,47.28],[727.47,532.99,42.8],[293.2,485.52,45.84],[1146.08,775.52,29.36],[656.56,972.88,45.36],[946.64,1012.72,54.88],[1056.56,954.56,43.68],[1140.64,221.36,66.64],[147.68,48.88,47.84],[614.7,516.43,66.68],[164.96,268.8,50.32],[1964.48,4255.04,67.2],[2519.52,4706,53.84],[2464.48,4706,72.24],[1781.52,5624.96,58.16],[1590.16,5128.64,48.24],[1363.28,4289.44,92.96],[1025.76,4952.88,65.84],[2746.91,3643.94,27.28],[1137.28,5677.44,64.88],[1040.8,3202.8,47.04],[359.84,4309.76,51.36],[200.64,4317.04,53.04],[160.08,4732.08,61.44],[483.84,5008.4,50.96],[379.28,5019.84,100.96]]

    pred_csv = "validation_outputs/psuedo_training/epoch_5/patient_images_front_wide_00_001.jpg_candidates.csv"

    df = pd.read_csv(pred_csv)

    r = 80
    e = 5 #error pixles

    img = Image.open(im)

    draw = ImageDraw.Draw(img)

    light = [95, 121, 89, 94]
    good = [97, 132, 105, 130, 48, 116, 27, 62]
    bad = [122, 143, 140]

    
    font = ImageFont.load_default()

    for i, row in df.iterrows():
        x = row['x']
        y = row['y']

        if i in light:
            crop = img.crop((x-r, y-r, x+r, y+r))
            crop.save(f"light_{i}.png")
        
        if i in good:
            crop = img.crop((x-r, y-r, x+r, y+r))
            crop.save(f"good_{i}.png")
        
        if i in bad:
            crop = img.crop((x-r, y-r, x+r, y+r))
            crop.save(f"bad_{i}.png")
        

    # for i, row in df.iterrows():
    #     x = row['x']
    #     y = row["y"]
    #     score = row["score"]
    #     draw.ellipse((x-r, y-r, x+r, y+r), outline='blue', width = 10)
    #     draw.text((x, y), str(i), fill='black', font=font)
    

    # r += 20
    # i = 0
    # for x,y, d in og_points:
    #     draw.ellipse((x-r, y-r, x+r, y+r), outline='red', width = 10)
        
    #     i+=1

    # img.save("test_out_2.png")
    
    # # r = 40

    # # for x, y, d in og_points:
    # #     draw.ellipse((x-r, y-r, x+r, y+r), outline="green", width=10)

    # r = 40

    # # for x, y in fns:
    # #     draw.ellipse((x-r, y-r, x+r, y+r), outline="blue", width=5)

    # img.save("fn_outputt.png")
    
def run_eval():
    for epoch in range(1, 10):    
        subprocess.run([sys.executable, "evaluate_points.py", "--val_out_dir", f"multiscale_test_output/epoch_{epoch}"], check=True)

def distribution():
    raw_candidates = "validation_outputs/multiscale_epoch_1/patient_images_front_wide_00_001.jpg_candidates.csv"
    multiscale_candidates = "multiscale_test_output/epoch_1/patient_images_front_wide_00_001.jpg_candidates.csv"

    raw_df = pd.read_csv(raw_candidates)
    multi_df = pd.read_csv(multiscale_candidates)
    
    raw_scores = raw_df["score"].tolist()
    multi_scores = multi_df["score"].tolist()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist(raw_scores, bins=20, range=(0, 1), color='blue', alpha=0.7)
    ax[0].set_title("Single Scale DINOv3 Candidate Scores")
    ax[0].set_xlabel("Score")
    ax[0].set_ylabel("Frequency")

    ax[1].hist(multi_scores, bins=20, range=(0, 1), color='orange', alpha=0.7)
    ax[1].set_title("Single Scale DINOv3 Candidate Scores")
    ax[1].set_xlabel("Score")
    ax[1].set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_sumn():
    x = [.894, .903, .905, .915, .921, .909, .909, .126, .103, 0.0]
    x = [-np.log(max(.01, xi)) for xi in x]
    plt.plot(x)
    plt.xlabel("Epoch")
    plt.ylabel("MSS")
    plt.show()

def eval_points_all_epochs():
    for epoch in range(1, 11):
        folder = f"multiscale_no_final_prior_output/epoch_{epoch}"
        if not os.path.exists(os.path.join(folder, "eval.csv")):
            continue
        subprocess.run([sys.executable, "evaluate_points.py", "--val_out_dir", folder])

def show_overlay():
    """Overlay a grayscale heatmap (numpy array or .npy file) on an image.

    Parameters
    - image: path to an image file or a PIL Image instance.
    - heatmap: numpy array or path to a .npy file or grayscale image file.
    - cmap: matplotlib colormap name (default 'jet').
    - alpha: max overlay alpha (0..1).
    - save_path: optional path to save the resulting image. If None, returns the PIL Image.
    """
    def _load_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGBA")
        if isinstance(img, Image.Image):
            return img.convert("RGBA")
        raise ValueError("image must be a file path or PIL.Image.Image")

    def _load_heatmap(hm):
        if isinstance(hm, str):
            # try numpy file first
            if hm.lower().endswith(".npy"):
                arr = np.load(hm)
            else:
                arr = np.array(Image.open(hm).convert("L"))
        elif isinstance(hm, np.ndarray):
            arr = hm
        else:
            raise ValueError("heatmap must be a numpy array or a path to .npy/.png/.jpg")
        return arr.astype(float)

    # convenience defaults for single-arg style
    # If caller passes two args, support that. Otherwise expect nothing and return.
    import inspect
    sig = inspect.signature(show_overlay)
    # We allow calling show_overlay(image, heatmap, cmap='jet', alpha=0.5, save_path=None)
    # But since original signature took no args, we provide a nested callable below.
    def _overlay(image, heatmap, cmap='jet', alpha=0.5, blur_radius=15, baseline=0.05, gamma=1.0, save_path=None):
        img = _load_image(image)
        hm = _load_heatmap(heatmap)

        # normalize heatmap to 0..1
        hm = hm - np.nanmin(hm)
        hm_max = np.nanmax(hm)
        if hm_max > 0:
            hm = hm / hm_max
        else:
            hm = np.zeros_like(hm)

        # resize heatmap to image size
        hm_img = Image.fromarray((hm * 255).astype(np.uint8)).resize(img.size, resample=Image.BILINEAR)
        # apply gaussian blur to spread small bright pixels into gradual blobs
        if blur_radius and blur_radius > 0:
            hm_img = hm_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        hm_resized = np.array(hm_img).astype(float) / 255.0

        # apply gamma correction to emphasize peaks
        if gamma != 1.0 and gamma > 0:
            hm_resized = np.power(hm_resized, float(gamma))

        # ensure even zero-valued pixels have a small baseline so they receive color
        baseline = float(baseline)
        if baseline > 0:
            hm_resized = hm_resized * (1.0 - baseline) + baseline

        # map through colormap
        cmap_fn = plt.get_cmap(cmap)
        colored = cmap_fn(hm_resized)  # returns NxMx4 floats 0..1
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

        overlay = Image.fromarray(colored_rgb).convert('RGBA')

        # create alpha mask from heatmap intensity scaled by alpha
        alpha_mask = (hm_resized * 255.0 * float(alpha)).clip(0, 255).astype(np.uint8)
        overlay.putalpha(Image.fromarray(alpha_mask))

        # composite over original
        blended = Image.alpha_composite(img, overlay)

        if save_path:
            blended.save(save_path)
            return blended

        return blended

    return _overlay


def create_training_heatmap(
    image,
    points=None,
    points_csv=None,
    points_json=None,
    diameter_aware=True,
    sigma=6.0,
    min_sigma=3.0,
    max_sigma=12.0,
    output_shape=None,
    save_npy=None,
    save_vis=None,
):
    """Create a training heatmap for `image` using existing utilities.

    Parameters
    - image: path to image or PIL Image or numpy array. Used to infer output size if `output_shape` is None.
    - points: sequence of (x,y) or (x,y,d) tuples or numpy array.
    - points_csv: optional path to a CSV containing point columns (tries common column names).
    - points_json: optional JSON string of points compatible with `parse_points`.
    - diameter_aware: use `make_diameter_aware_heatmap` when points contain diameters.
    - sigma: fallback sigma for `make_heatmap`.
    - min_sigma,max_sigma: passed to diameter-aware heatmap.
    - output_shape: (h,w) tuple to generate heatmap at that resolution. If None uses image size.
    - save_npy: optional path to save the resulting heatmap (.npy).
    - save_vis: optional path to save a visualization PNG (uses `show_overlay`).

    Returns numpy heatmap (H, W) float32 in range ~0..1
    """
    from lesion_dinov3.utils import read_rgb, make_heatmap, make_diameter_aware_heatmap, parse_points

    # determine points list
    pts = None
    if points is not None:
        pts = list(points)
    elif points_csv is not None:
        df = pd.read_csv(points_csv)
        # try common column names
        if set(["x", "y", "d"]).issubset(df.columns):
            pts = [(float(r.x), float(r.y), float(r.d)) for r in df.itertuples()]
        elif set(["x", "y", "diameter_px"]).issubset(df.columns):
            pts = [(float(r.x), float(r.y), float(r.diameter_px)) for r in df.itertuples()]
        elif set(["x", "y"]).issubset(df.columns):
            pts = [(float(r.x), float(r.y)) for r in df.itertuples()]
        else:
            # try first two numeric columns
            numcols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if len(numcols) >= 2:
                pts = [(float(r[0]), float(r[1])) for r in df[numcols[:2]].values]
            else:
                raise ValueError("Could not parse points from CSV")
    elif points_json is not None:
        pts = parse_points(points_json)
    else:
        raise ValueError("points, points_csv or points_json must be provided")

    # determine output shape
    if output_shape is None:
        if isinstance(image, str):
            img = read_rgb(image)
            h, w = img.shape[:2]
        elif isinstance(image, Image.Image):
            w, h = image.size
        elif isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            raise ValueError("image must be a path, PIL Image or numpy array when output_shape is None")
    else:
        h, w = int(output_shape[0]), int(output_shape[1])

    # call appropriate heatmap generator
    if diameter_aware and len(pts) > 0 and len(pts[0]) == 3:
        hm = make_diameter_aware_heatmap(h, w, pts, image_scale=1, min_sigma=min_sigma, max_sigma=max_sigma)
    else:
        hm = make_heatmap(h, w, pts, sigma=float(sigma))

    hm = hm.astype(np.float32)
    # clip/normalize to 0..1
    hm = np.clip(hm, 0.0, None)
    m = hm.max()
    if m > 0:
        hm = hm / m

    if save_npy:
        np.save(save_npy, hm)

    if save_vis:
        overlay_fn = show_overlay()
        # save visualization with modest alpha and blur so points are visible
        overlay_fn(image, hm, cmap='jet', alpha=0.6, blur_radius=12, baseline=0.02, gamma=0.6, save_path=save_vis)

    return hm


def binarize_heatmap(npy_path, threshold=None, percentile=99.0, method='percentile', blur_radius=0, save_path=None, invert=False, output_shape=None):
    """Load a heatmap from a .npy file (or array) and produce a binary B/W image.

    Parameters
    - npy_path: path to .npy file or a numpy array containing the heatmap.
    - threshold: numeric threshold; if provided, used directly.
    - percentile: percentile used to compute threshold when method=='percentile'.
    - method: 'percentile' or 'fixed' or 'meanstd' (mean+std).
    - blur_radius: optional Gaussian blur radius to apply before thresholding.
    - save_path: optional path to save output PNG (0=black,255=white).
    - invert: if True, invert colors (heat becomes white->black).
    - output_shape: optional (h,w) to resize output image.

    Returns a uint8 numpy array of shape (H,W) with values 0 or 255.
    """
    # load array if path given
    if isinstance(npy_path, str):
        hm = np.load(npy_path)
    elif isinstance(npy_path, np.ndarray):
        hm = npy_path
    else:
        raise ValueError("npy_path must be a path or numpy array")

    hm = hm.astype(np.float32)

    # optionally resize first if output_shape provided
    if output_shape is not None:
        h_out, w_out = int(output_shape[0]), int(output_shape[1])
        hm_img = Image.fromarray((np.clip(hm, 0, None) / (hm.max() + 1e-9) * 255).astype(np.uint8))
        hm_img = hm_img.resize((w_out, h_out), resample=Image.BILINEAR)
        hm = np.array(hm_img).astype(np.float32) / 255.0

    # normalize
    hm = hm - np.nanmin(hm)
    hm_max = np.nanmax(hm)
    if hm_max > 0:
        hm = hm / hm_max
    else:
        hm = np.zeros_like(hm)

    # optional blur
    if blur_radius and blur_radius > 0:
        hm_img = Image.fromarray((hm * 255).astype(np.uint8))
        hm_img = hm_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        hm = np.array(hm_img).astype(np.float32) / 255.0

    # determine threshold
    if threshold is None:
        if method == 'percentile':
            thresh = float(np.percentile(hm, float(percentile)))
        elif method == 'meanstd':
            thresh = float(np.mean(hm) + np.std(hm))
        else:
            thresh = float(np.mean(hm))
    else:
        thresh = float(threshold)

    bw = (hm >= thresh).astype(np.uint8) * 255
    if invert:
        bw = 255 - bw

    if save_path:
        Image.fromarray(bw).save(save_path)

    return bw

def save_overlay():
    overlay_fn = show_overlay()
    im = "C:/Users/davoo/Development/LumoLesionClassifier/patient_images/front/wide_00_001.jpg"
    npy = "validation_outputs/multiscale_no_final_prior_best_epoch/patient_images_front_wide_00_001.jpg_lesion_heatmap.npy"

    

    img = cv2.imread(im)
    hm = np.load(npy).astype(np.float32)

    # display-only enhancement
    hm_vis = hm / (hm.max() + 1e-6)
    hm_vis = cv2.GaussianBlur(hm_vis, (0, 0), sigmaX=8, sigmaY=8)
    p99 = np.percentile(hm_vis, 99.8)
    hm_vis = np.clip(hm_vis / (p99 + 1e-6), 0, 1)
    hm_vis = hm_vis ** 0.4

    heat_color = cv2.applyColorMap((hm_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = img#cv2.addWeighted(img, 0.75, heat_color, 0.25, 0)

    pts, scores = extract_peaks(hm, threshold=0.30, radius=3, max_candidates=5000)

    for (x, y), s in zip(pts, scores):
        cv2.circle(overlay, (int(x), int(y)), 30, (191, 27, 54), 5)
        cv2.circle(overlay, (int(x), int(y)), 3, (191, 27, 54), -1)

    cv2.imwrite("widefield_points(1).png", overlay)

def highpass(path):
    img = cv2.imread(path)

    hpf = img - cv2.GaussianBlur(img, (21, 21), 3)
    cv2.imshow("HighPass Filete", hpf)

    cv2.imwrite("hpf.png", hpf)


if __name__ == "__main__":
    # p = [[5328.88,5875.84,37.36],[5162.08,5547.44,36],[5495.68,5428.72,33.52],[4857.28,5441.76,26.64],[5086.96,4971.04,38.8],[5484.32,4702.96,46.96],[5545.44,4520.4,35.84],[5390.88,4626.96,33.36],[4999.6,4441.76,27.2],[5202.24,4323.04,43.12],[5546.32,4260.16,32.24],[4985.6,4104.4,32],[4980,3877.2,50],[4935.2,3591.6,36.24],[5452,3536.4,35.12],[5385.6,3522.8,48.8],[5202.4,3343.6,40],[5292,3074.8,40],[2662.16,2669.64,39.2],[4962.4,2853.2,43.12],[5152.8,2798,35.76],[5134.4,2267.6,44.56],[4966.4,1707.6,41.6],[4547.2,1827.6,39.6],[4620.8,1989.36,29.12],[4653.12,2012,50.56],[4904.96,1996.64,36.32],[5307.46,1594.46,26.72],[4296.32,1642.24,44.88],[4075.12,1402.48,64.32],[4454.56,1080.4,58.32],[4356,1230.56,39.28],[4275.28,1344.4,39.12],[4607.92,1274.96,39.6],[5014.8,1735.92,37.52],[95.52,4945.26,154.44],[4934.88,1527.6,26.08],[4480.4,863.2,86.4],[4446.56,676.96,48.56],[4188.32,430.08,60],[3509.73,648.45,35.4],[3587.28,947.92,34.48],[1575.06,2914.75,62.92],[3812.16,256.64,62.8],[3582.88,105.36,58.72],[3546.32,326.56,63.52],[3466.64,412.72,44.24],[3004,365.6,45.36],[3994.24,309.44,37.92],[3660.16,820.88,34.88],[2767.76,112.96,49.28],[2734.56,257.04,46.96],[1819.12,850.64,47.52],[1975.6,762.08,39.36],[1571.2,789.68,35.04],[2010.08,213.04,45.28],[1665.68,246.48,44.56],[1625.28,251.36,32],[1793.6,789.68,49.2],[1839.2,53.2,51.6],[1680,390.8,37.76],[1704,481.2,40.8],[2155.2,584.4,37.36],[1404,100.4,36.56],[906.72,889.84,53.6],[808.64,1090.4,57.84],[628.56,1182,47.28],[727.47,532.99,42.8],[293.2,485.52,45.84],[1146.08,775.52,29.36],[656.56,972.88,45.36],[946.64,1012.72,54.88],[1056.56,954.56,43.68],[1140.64,221.36,66.64],[147.68,48.88,47.84],[614.7,516.43,66.68],[164.96,268.8,50.32],[1964.48,4255.04,67.2],[2519.52,4706,53.84],[2464.48,4706,72.24],[1781.52,5624.96,58.16],[1590.16,5128.64,48.24],[1363.28,4289.44,92.96],[1025.76,4952.88,65.84],[2746.91,3643.94,27.28],[1137.28,5677.44,64.88],[1040.8,3202.8,47.04],[359.84,4309.76,51.36],[200.64,4317.04,53.04],[160.08,4732.08,61.44],[483.84,5008.4,50.96],[379.28,5019.84,100.96]]
    # hm = create_training_heatmap(
    #     "C:/Users/davoo/Development/LumoLesionClassifier/patient_images/front/wide_00_001.jpg",
    #     points=p,
    #     diameter_aware=True,      # use diameter-aware Gaussians if points include diameters
    #     sigma=6.0,                # fallback sigma for simple points
    #     min_sigma=3.0, max_sigma=12.0,
    #     save_npy="out_heatmap.npy",
    #     save_vis="out_heatmap_vis.png"
    # )
    save_eval_image()