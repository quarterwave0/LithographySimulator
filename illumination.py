import numpy as np
from matplotlib import pyplot as plt
import torch

#annular light source, where sigmain is the internal radius as a function of NA and sigma out is the external radius
sigmain = 0
sigmaout = 0.8

#off axis illumination weirdness
sigmashiftx = 0
sigmashifty = 0

pixelNumber = 64
nmaperture=0.7 #lightsource NA
wavelength=193 #ArF excimer

def calculateLightSource(device):
    sourcer = torch.zeros((pixelNumber, pixelNumber), dtype=int, device=device)

    deltaSigma = 4/pixelNumber #step size for illumination distribution we are stepping through
    sigmaBound = (pixelNumber/2)*deltaSigma 
    SEPSD = sigmaBound/nmaperture

    #sigmaX = torch.arange(-sigmaBound, sigmaBound, pixelNumber, dtype=torch.float32, device=device)
    #sigmaY = torch.arange(-sigmaBound, sigmaBound, pixelNumber, dtype=torch.float32, device=device)
    #sX, sY = torch.meshgrid((sigmaX, sigmaY), indexing='xy')
    #sXY = torch.stack((sX, sY), dim=-1)
 
    sigmaX = torch.arange(-SEPSD-sigmashiftx, SEPSD-sigmashiftx, deltaSigma/nmaperture, dtype=torch.float32, device=device)
    sigmaY = torch.arange(-SEPSD-sigmashifty, SEPSD-sigmashifty, deltaSigma/nmaperture, dtype=torch.float32, device=device)
    sX, sY = torch.meshgrid((sigmaX, sigmaY), indexing='xy')
    #sXY = torch.stack((sX, sY), dim=-1)
    O = torch.sqrt(sX**2 + sY**2)

    #sigmaXDE = np.tile(np.arange(-SEPSD-sigmashiftx, SEPSD-sigmashiftx, deltaSigma/nmaperture), pixelNumber)
    #sigmaYDE = np.repeat(np.arange(-SEPSD-sigmashifty, SEPSD-sigmashifty, deltaSigma/nmaperture), pixelNumber)
    #O = np.reshape(np.sqrt(sigmaYDE**2 + sigmaXDE**2), (pixelNumber, pixelNumber))

    O = torch.where((O >= sigmain) & (O <= sigmaout), 1, 0)

    return O, O.cpu()

if __name__ == '__main__':

    if torch.cuda.is_available:
        device = torch.device('cuda')
        print(f"Using {torch.cuda.get_device_name(device)} for illumination computation")
    else:
        device = torch.device('cpu')
        print(f"Using CPU for illumination computation")

    O, OCPU = calculateLightSource(device)

    plt.imshow(OCPU)
    plt.show()