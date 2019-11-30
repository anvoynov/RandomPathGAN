#!/usr/bin/env python3
# based on //github.com/mseitzer/pytorch-fid.git
import numpy as np
from scipy import linalg
from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool2d
from score.inception import InceptionV3


__potential_dims__ = list(InceptionV3.BLOCK_INDEX_BY_DIM)
__default_dim__ = 2048


def get_activations(dataloader, model, dims=2048,
                    cuda=False, verbose=False, samples_to_take=None):
    model.eval()

    batch_size = dataloader.batch_size
    expected_out_samples = samples_to_take + batch_size if samples_to_take is not None \
            else len(dataloader) * batch_size
    pred_arr = np.empty([expected_out_samples, dims])

    samples_count = 0
    for i, images in tqdm(enumerate(dataloader)):
        if cuda:
            images = images.cuda()
        images = 0.5 * (images + 1.0)

        pred = model(images)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[i * batch_size: (i + 1) * batch_size] = \
            pred.cpu().data.numpy().reshape(batch_size, -1)

        samples_count += batch_size
        if samples_to_take is not None and samples_count > samples_to_take:
            break

    pred_arr = pred_arr[:samples_count]

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(dataloader, model,
                                    dims=2048, cuda=False, verbose=False, samples_to_take=None):
    """Calculation of the statistics used by the FID.
    Params:
    -- dataloader  : images tensors dataloader
    -- model       : Instance of inception model
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(dataloader, model, dims, cuda, verbose, samples_to_take)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def make_model_for_fid(cuda=True, dims=__default_dim__, model_path=None):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx], model_path=model_path)
    if cuda:
        model.cuda()
    return model


def calculate_fid_for_generators(original_data_loader, generators_dataloaders, cuda=True,
                                 dims=__default_dim__, verbose=False, model_path=None,
                                 samples_to_take=None):
    model = make_model_for_fid(cuda, dims, model_path)
    m_orig, s_orig = calculate_activation_statistics(
        original_data_loader, model, dims, cuda, verbose, samples_to_take)

    fids = []
    for gen in generators_dataloaders:
        gen.generator.cuda()
        m, s = calculate_activation_statistics(gen, model, dims, cuda, verbose, samples_to_take)
        fids.append(calculate_frechet_distance(m, s, m_orig, s_orig))
        gen.generator.cpu()

    return fids


def calculate_fid_given_dataloaders(dataloaders, cuda, dims, verbose=False):
    model = make_model_for_fid(cuda, dims)
    stat = []
    for dl in dataloaders:
        stat.append(calculate_activation_statistics(dl, model, dims, cuda, verbose))

    return calculate_frechet_distance(stat[0][0], stat[0][1], stat[1][0], stat[1][1])
