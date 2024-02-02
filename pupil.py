from matplotlib import pyplot as plt
import torch

pixelNumber = 64

NA = 0.7
wavelength = 193

defocus = 0

sigmaSpan = 2
deltaSigma = sigmaSpan*2/pixelNumber

Z3 = defocus*NA**2/(4*wavelength)
#Equation 3.24 of mack

def addPhaseError(o):

    WR = Z3*(2*o**2-1) #This is performed in sigma space, that is, on the unit disk in spatial fq domain
    phi = torch.exp(1j*2*torch.pi*WR)
    return phi

pupilFunction = torch.zeros((pixelNumber, pixelNumber), dtype=torch.complex64)
zernikeField = pupilFunction

sigmaX = torch.arange(-sigmaSpan, sigmaSpan, deltaSigma, dtype=torch.float32)
sigmaY = torch.arange(-sigmaSpan, sigmaSpan, deltaSigma, dtype=torch.float32)
sX, sY = torch.meshgrid((sigmaX, sigmaY), indexing='xy')
o = torch.sqrt(sX**2 + sY**2) # In the original, there was a /NA here for reasons I cannot determine

zernikeField = addPhaseError(o)

pupilFunction = torch.where(o<=1, zernikeField, 0)

if __name__ == '__main__':
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(torch.real(pupilFunction))
    ax2.imshow(torch.imag(pupilFunction))

    plt.show()
    print("hewwow")