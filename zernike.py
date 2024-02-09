import torch
from math import factorial, sqrt

def diracd(v):
    if v == 0:
        return 1
    else:
        return 0

def generateZ(m, n, pixelNumber, coeff, device):
    #implementation of eq (4.37) from [2], normalization factor from [5]
    #kudos to [5] for some hints on this

    sigmaSpan = 2
    x = torch.arange(-sigmaSpan, sigmaSpan, sigmaSpan*2/pixelNumber, dtype=torch.float32, device=device)
    X, Y = torch.meshgrid((x, x), indexing='xy')

    r = (X**2+Y**2)
    theta = torch.arctan2(Y, X)

    lLim = int((n-abs(m))/2)
    ilLim = int((n+abs(m))/2)

    Rmn = torch.zeros((lLim+1, pixelNumber, pixelNumber))
    for k in range(lLim+1):
        staticCoeff = ((-1)**k * factorial(n-k)) / (factorial(k)*factorial(ilLim-k)*factorial(lLim-k))
        intm = staticCoeff * r**(n-2*k)
        Rmn[k] = intm

    R = torch.sum(Rmn, dim=0)
    Nmn = sqrt((2*n+1)/(1+diracd(m)))

    if m >= 0:
        Z = coeff * Nmn * R * torch.cos(m*theta)
    else:
        Z = coeff * -Nmn * R * torch.sin(m*theta)

    mask = torch.where(r>=1, 0, 1)
    return Z*mask

def OSAindexToMN(ji): #TODO: add the annoying fringe indexing system
    #eq (4.39) and (4.40) in [2]
    n = int((-3 + sqrt(9 + 8*ji))/2) 
    m = 2*ji - n*(n + 2)
    return m, n

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    Z = generateZ(-3, 3, 64, 20, torch.device('cpu')) #TODO: Add multi-abberation support with given order and coefficients
    plt.imshow(Z)
    plt.show()