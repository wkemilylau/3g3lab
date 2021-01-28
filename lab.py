import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import requests
import scipy.ndimage as ndimage
from scipy.optimize import minimize, approx_fprime
from IPython import display
import time


def load_data(mode):
    """
    Returns
    -------
    imgs
        A dictionary of images (keys: I1, I2, I2w, and I3)
    """
    if mode == "local":
        imgs = np.load("imgs/imgs.npz")
    elif mode == "colab":
        url = 'https://github.com/tachukao/3g3lab-imgs/blob/master/imgs.npz?raw=true'
        r = requests.get(url, stream=True)
        imgs = np.load(BytesIO(r.raw.read()))
    else:
        raise Exception("mode must be either local or colab")
    return imgs


def plot_image(x, i):
    """
    Parameters
    ----------
    x
        images (e.g. imgs["I2"])
    i
        ith image of x (i.e., x[i])

    Returns
    -------
    None
    """
    if i > x.shape[0] - 1:
        raise Exception("index %i out of bounds, total number of images: %i" %
                        (i, x.shape[0]))
    n_pix = x.shape[1]
    x = x[i]
    sz = int(np.sqrt(n_pix))
    x = x.reshape(sz, sz)
    plt.figure()
    plt.imshow(x, interpolation="nearest")
    plt.axis("off")


def plot_all_images(x):
    """
    Parameters
    ----------
    x 
        images with shape (n_img, n_pix)
        e.g., imgs["I2"]

    Returns
    -------
    None
    """
    n_img, n_pix = x.shape
    if n_img > 1:
        n_cols = int(np.ceil(np.sqrt(n_img)))
        n_rows = n_cols
        _, axes = plt.subplots(n_rows,
                               n_cols,
                               figsize=(10, 10),
                               sharex=True,
                               sharey=True)
        sz = int(np.sqrt(n_pix))
        for i in range(n_img):
            acol = i % n_cols
            arow = (i - acol) // n_cols
            axes[arow, acol].imshow(x[i].reshape((sz, sz)), interpolation="nearest")
            axes[arow, acol].axis("off")
    else:
        plot_image(x, 0)


def overlayimages(img, basis, oimg=None):
    """
    Parameters
    ----------
    img
        If oimg is None, then it is treated as the original image.
        Otherwise, it is only used for computing the filtered image
    basis
        basis functions
    oimg
        Original image

    Returns
    -------
    """
    osz = int(np.sqrt(len(img)))
    sz = int(np.sqrt(len(basis)))
    img = img.reshape((osz, osz))
    basis = basis.reshape((sz, sz))
    filtered = ndimage.convolve(img, basis)
    img = img if oimg is None else img  # used for I2w
    _, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(img, interpolation="nearest")
    axes[0, 0].axis("off")
    axes[0, 0].title.set_text("original image")
    axes[0, 1].imshow(filtered, cmap="RdBu_r", interpolation="nearest")
    axes[0, 1].axis("off")
    axes[0, 1].title.set_text("filtered image")
    axes[1, 0].imshow(img, cmap="gray", alpha=0.55, interpolation="nearest")
    axes[1, 0].imshow(filtered, cmap="RdBu_r", alpha=0.45, interpolation="nearest")
    axes[1, 0].axis("off")
    axes[1, 0].title.set_text("filtered overlayed on original")
    axes[1, 1].imshow(basis, interpolation="nearest")
    axes[1, 1].axis("off")
    axes[1, 1].title.set_text("filter (basis)")


def sample_patches(imgs, sz, n_sub):
    """
    Sample image patches
    
    Parameters
    ----------
    imgs
        images with shape (N_img, N_pix)
    sz 
        side-length (in pixels) of the image patches
    n_sub
        number of image patches to sample
    
    Returns
    -------
    S 
        matrix of image patches with shape (n_sub, n_pix), where n_pix =  sz * sz
    """
    N_img = imgs.shape[0]
    samples = []
    osz = int(np.sqrt(len(imgs[0])))
    for _ in range(n_sub):
        img = imgs[np.random.randint(N_img)].reshape((osz, osz))
        ii = np.random.randint(osz - sz + 1)
        jj = np.random.randint(osz - sz + 1)
        c = img[ii:ii + sz, jj:jj + sz].reshape(1, -1)
        samples.append(c)
    return np.concatenate(samples, 0)


def pca(x):
    """
    PCA
    ---
    
    Parameters
    ----------
    x: matrix of subimages with shape (n_sub, n_pix)
    
    Returns
    -------
    bases: basis functions with shape (n_pix, n_pix)
    pct: percentage of variance captured by each basis functions (each row of bases)
    """
    n_sub, _ = x.shape
    x = x.reshape((n_sub, -1))
    x = x - np.mean(x, 0)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    var = np.square(s)
    pct = var / np.sum(var)
    bases = vt
    return bases, pct


def pca3g3(imgs, sz, n_sub):
    """
    Extract subimages from an image set and perform PCA them.

    Parameters
    ----------
    imgs
        images with shape (N_img, N_pix), where N_pix is the number of pixels in each full image
    sz
        size of the image patches
    n_sub
        number of image patches to sample
    
    Returns
    -------
    bases
        basis functions with shape (n_pix, n_pix), where n_pix is the number of pixels in each subimage
    pct
        percentage of variance captured by each basis functions (each row of bases)
    """
    S = sample_patches(imgs, sz, n_sub)
    bases, pct = pca(S)
    return bases, pct


def grad_error(cost, dcost, A_or_B):
    """
    Checks implementation of dcost_A or dcost_B with a finite-difference approximation of the gradient
    
    Parameters
    ----------
    cost
        cost function
    dcost
        gradient of cost function with respect to A or B
    A_or_B
        must be either "A" or "B", telling the function to check the cost gradient w.r.t. A or B.
    
    Returns
    -------
    dCerr
        error in dcost
    
    Notes
    -----
    If cost, dcost_A, and dcost_B are implemented correctly, gradient error should be less than 1E-5
    
    """
    lambd = 0.1
    sigma = 0.3
    sz = 8
    n_sub = 3
    n_bas = 64
    A = np.random.randn(n_sub, n_bas)
    B = np.random.randn(n_bas, sz * sz)
    S = np.random.randn(n_sub, sz * sz)
    # In Python, functions can be defined as "lambda expressions"
    # (do not confuse lambda in the next lines with lambd -- see https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions)
    if A_or_B == "A":
        x_dim = A.size
        C = lambda A: cost(A.reshape(n_sub, -1), B, S, lambd, sigma)[0]
        dC = lambda A: dcost(A.reshape(n_sub, -1), B, S, lambd, sigma).ravel()
    elif A_or_B == "B":
        x_dim = B.size
        C = lambda B: cost(A, B.reshape(n_bas, -1), S, lambd, sigma)[0]
        dC = lambda B: dcost(A, B.reshape(n_bas, -1), S, lambd, sigma).ravel()
    x = np.random.randn(x_dim)
    dCapprox = approx_fprime(x, C, np.sqrt(np.finfo(float).eps))
    dC = dC(x)
    return np.linalg.norm(dCapprox - dC) / np.linalg.norm(dCapprox + dC)


def optimal_A(cost, dcost_A, dcost_B, A0, B, S, lambd, sigma):
    """
    Finds the optimal activations A for image patches S and basis functions B with initial guess A0
    
    Parameters
    ----------
    cost
        cost function
    dcost_A
        gradient of cost function with respect to A i.e. dcost/dA 
    dcost_B
        gradient of cost function with respect to B i.e. dcost/dB 
    A0
        initial guess of optimal activations A
    B
        bases functions with shape (n_bas, n_pix)
    S
        matrix of subimages with shape (n_sub, n_pix)
    lambd
        scalar parameter that weighs the importance of reconstruction error and sparsity
    sigma
        activation scale hyper-parameter in the cost function
    
    Returns
    -------
    A
        optimal A
    """
    n_sub = S.shape[0]
    f = lambda A: cost(A.reshape(n_sub, -1), B, S, lambd, sigma)[0]
    df = lambda A: dcost_A(A.reshape(n_sub, -1), B, S, lambd, sigma).ravel()
    res = minimize(f,
                   A0.ravel(),
                   method='L-BFGS-B',
                   jac=df,
                   options={'gtol': 1e-3})
    if not res.success:
        print(res.message)
    opt_A = res.x.reshape(n_sub, -1)
    return opt_A


class Normaliser:
    def __init__(self, n_bas, B, var_goal=0.1, var_eta=0.1, alpha=0.02):
        self.var_goal = var_goal
        self.A_var = np.ones((1, n_bas)) * self.var_goal
        self.alpha = alpha
        self.var_eta = var_eta
        self.gain = np.linalg.norm(B, axis=1, keepdims=True).T

    def __call__(self, B, A):
        self.A_var = ((1 - self.var_eta) * self.A_var) + (
            self.var_eta * np.mean(A**2, axis=0, keepdims=True))
        self.gain = self.gain * ((self.A_var / self.var_goal)**self.alpha)
        normB = np.sqrt(np.sum(B**2, axis=1, keepdims=True))
        B = self.gain.T * B / normB
        return B


def normalize_B(B):
    """
    Normalises each basis function in the input.

    Parameters
    ----------
    B
        basis functions of size (n_bases, n_pix)
    """
    return B / np.sqrt(np.sum(B**2, axis=1, keepdims=True))


def sparseopt(cost,
              dcost_A,
              dcost_B,
              imgs,
              n_iter,
              eta=0.2,
              lambd=2 * 0.01 * 5,
              sigma=0.316,
              sz=8,
              n_bas=64,
              n_sub=100,
              verbose=False,
              display_every=50,
              B0=None):
    """
    Learns the basis functions for imgs
    
    Parameters
    ----------
    cost
        cost function
    dcost_A
        gradient of cost function with respect to A i.e. dcost/dA 
    dcost_B
        gradient of cost function with respect to B i.e. dcost/dB 
    imgs
        the set images for which we wish to find the optimal basis functions, with shape (N_img, N_pix)
    n_iter : int
        number of iterations of optimization
    eta : float
        learning rate
    lambd : float
        scalar parameter that weighs the importance of reconstruction error and sparsity
    sigma : float
        activation scale hyper-parameter in the cost function
    sz : int
        size of the image patches to sample form imgs
    n_bas : int
        number of bases functions to learn
    n_sub : int
        number of image patches to use at each iteration
    verbose : bool
        if False, print only iteration, cost, err sparsity 
        if True, print also avg A var norm, avg B norm, and avg gain
    display_every : int
        determines how often the progress is printed and the bases plotted
    B0
        initial guess of the bases functions (optional)

    Returns
    -------
    B
        optimal basis functions
    cost_hist
        a list of the cost over the course of the optimization (computed every display_every iteration)
    """
    if B0 is not None:
        # check B0 has the right hspae
        if B0.shape != (n_bas, sz * sz):
            raise Exception(
                "B0 should have shape (%i, %i), instead it has shape (%i, %i)"
                % (n_bas, sz * sz, B0.shape[0], B0.shape[1]))

        B = B0
    else:
        B = np.random.rand(n_bas, sz * sz) - 0.5
    normalise = Normaliser(n_bas, B)
    cost_hist = []
    n_rows = int(np.ceil(np.sqrt(n_bas)))
    n_cols = n_rows
    fig, axes = plt.subplots(n_rows,
                             n_cols,
                             figsize=(10, 10),
                             sharey=True,
                             sharex=True)
    for i in range(n_iter):
        ###############################################################
        ###  OPTIMIZATION INNER BEGINS                              ###
        ###############################################################
        S = sample_patches(imgs, sz, n_sub)
        A0 = S.dot(B.T) / np.sum(B**2, axis=1,
                                 keepdims=True).T  # initial guess
        #A0 = np.random.randn(n_sub, n_bas) * 0.01
        A = optimal_A(cost, dcost_A, dcost_B, A0, B, S, lambd, sigma)
        if (i + 1) % display_every == 0:
            l, err, sparsity = cost(A, B, S, lambd, sigma)
            cost_hist.append(l)
        dB = dcost_B(A, B, S, lambd, sigma)
        B = B - eta * dB
        B = normalise(B, A)
        if (i + 1) % display_every == 0:
            if verbose:
                msg = "iter %i cost %.4f, err %.4f, sparsity %.4f, avg A var %.5f, avg norm of B %.5f, norm of gain %.3f" % (
                    i + 1, l, err, sparsity, np.mean(normalise.A_var),
                    np.linalg.norm(B) / n_bas, np.linalg.norm(normalise.gain))
            else:
                msg = "iter %i cost %.4f, err %.4f, sparsity %.4f" % (
                    i + 1, l, err, sparsity)
            fig.suptitle(msg)
            for ii in range(n_bas):
                axcol = ii % n_cols
                axrow = (ii - axcol) // n_cols
                Bii = B[ii]
                vm = np.max(np.abs(Bii))
                axes[axrow, axcol].imshow(Bii.reshape((sz, sz)), interpolation="nearest", vmin=-vm, vmax=vm)
                axes[axrow, axcol].axis("off")
            display.display(fig)
            display.clear_output(wait=True)
            time.sleep(0.01)
        ###############################################################
        ###  OPTIMIZATION INNER LOOP ENDS                           ###
        ###############################################################
    return normalize_B(B), cost_hist
