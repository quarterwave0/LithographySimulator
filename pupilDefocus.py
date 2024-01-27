import numpy as np
from matplotlib import pyplot as plt

pixel = 64
nmaperture = 0.7
lamda = 193
defocus = 20

delta = 4/pixel
kmax = 2

Z3 = defocus*np.square(nmaperture)/(4*lamda)

pupilf = np.zeros((pixel, pixel), dtype=np.complex_)

kx = -kmax
for p in range(pixel):
    ky = -kmax
    for q in range(pixel):

        o = np.sqrt(np.square(kx)+np.square(ky))
        Rk = o/nmaperture
        WR = Z3*(2*np.square(Rk)-1)

        if(o < nmaperture):
            phi = 1*np.exp(1j*2*np.pi*WR)
            pupilf[p, q] = phi
        else:
            pupilf[p,q] = 0
        
        ky = ky + delta
    kx = kx + delta

if __name__ == '__main__':
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(np.real(pupilf))
    ax2.imshow(np.imag(pupilf))

    plt.show()