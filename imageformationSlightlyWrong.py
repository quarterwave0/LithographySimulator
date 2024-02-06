import torch
from pupil import pupilFunction
from matplotlib import pyplot as plt
from lightsource import LightSource
from mask import Mask

def calculateAerial(pupil, maskFT, fraunhoferConstant, pixelNumber, pixelSize, device):
    pixelNumber = maskFT.size()[0]
    deltaK=4/pixelNumber #fn step size
    
    Kbound = pixelNumber / 2 * deltaK
    pixelBound = pixelNumber / 2 * pixelSize

    #create all needed tensors

    kx = torch.arange(-Kbound, Kbound, deltaK, dtype = torch.float32, device=device) #- deltaK
    ky = torch.arange(-Kbound, Kbound, deltaK, dtype = torch.float32, device=device)
    KX, KY = torch.meshgrid(kx,ky, indexing='xy')
    k_grid = torch.stack((KX, KY), dim=-1)

    xs = torch.arange(-pixelBound, pixelBound, pixelSize, dtype = torch.float32, device=device) #-pixelSize
    ys = torch.arange(-pixelBound, pixelBound, pixelSize, dtype = torch.float32, device=device)
    XS, YS = torch.meshgrid(xs,ys,indexing='xy')
    xy_grid = torch.stack((XS, YS), axis=-1)

    k_grid = k_grid.unsqueeze(2).unsqueeze(2)
    xy_grid = xy_grid.unsqueeze(0).unsqueeze(0)

    solution = torch.zeros((pixelNumber, pixelNumber), dtype=torch.complex64, device=device)
    exponent = torch.sum((k_grid * xy_grid), dim=-1) * fraunhoferConstant

    intermediate = pupil * maskFT * torch.exp(exponent)

    solution = torch.sum(intermediate, axis=(2,3))

    return solution

def abbeImage(maskFT, pupilF, lightsource, pixelSize, wavelength, device):

    pixelNumber = maskFT.size()[0]
    fraunhoferConstant = (-2*1j*torch.pi)/wavelength

    image = torch.zeros((pixelNumber, pixelNumber), dtype=torch.complex64, device=device)

    pupilOnDevice = pupilF.to(device)
    pupilshift = torch.zeros((pixelNumber*2, pixelNumber*2, pixelNumber, pixelNumber), dtype=torch.complex64, device=device)

    a = torch.arange(0, pixelNumber, 1, dtype=torch.int32, device=device)
    b = torch.arange(0, pixelNumber, 1, dtype=torch.int32, device=device)
    A, B = torch.meshgrid((a, b), indexing='xy')

    i = torch.arange(0, pixelNumber, 1, dtype=torch.int32, device=device)
    j = torch.arange(0, pixelNumber, 1, dtype=torch.int32, device=device)
    I, J = torch.meshgrid((i, j), indexing='xy')
    Iu = I.unsqueeze(-1).unsqueeze(-1)
    Ju = J.unsqueeze(-1).unsqueeze(-1)
    
    pupilshift[(A+Iu), (B+Ju), Iu, Ju] = pupilOnDevice
    #there are Px x Px fields of Px x Px (1) where each (1) field has the pupil function where it is illuminated by the light source at a different position within it.
    # A and B represent every position in our un-padded field, and I and J respectively slide the pupil around our padded pupilshift space by broadcasting the AB grid across itself grid through addition
    # such that A begins at 1 for I = 1, etc.
    psTrim = pupilshift.narrow(0, 32, 64).narrow(1, 32, 64)

    for i in range(pixelNumber):
        for j in range(pixelNumber):
            if (lightsource[i, j] > 0): #Suprisingly, this is appreciably faster than the equivalent multiplication process
                image = image + calculateAerial(psTrim[:, :, i, j], maskFT, fraunhoferConstant, pixelNumber, pixelSize, device)

    #imagers = torch.real(imagerr * torch.conj(imagerr)) #In the original matlab, this is a .* (elementwise), so the * is intentional, but I think they meant:
    return torch.abs(image)

device = torch.device('cuda')

mask = Mask(device=device)
F = mask.fraunhofer(wavelength=193)

lightsource = LightSource(device=device)
ls = lightsource.generateSource()

aerialImage = abbeImage(F, pupilFunction, ls, 25, 193, device)
plt.contour(aerialImage.cpu())
#plt.plot(aerialImage.cpu()[32, :])
plt.show()