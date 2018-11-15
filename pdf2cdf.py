import numpy as np


def pdf_to_cdf(pdf):
    last = 0
    cdf = np.zeros(len(pdf[1]))
    for i in range(len(pdf[1])):
        last = last + pdf[i]
        cdf[i] = last
    return cdf
