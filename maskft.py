import numpy as np
from matplotlib import pyplot as plt
import torch

pixelNumber = 64
pixelSize = 25 #nanometers
pixelBound = pixelNumber / 2 * pixelSize

deltaK = 4 / pixelNumber
Kbound = pixelNumber / 2 * deltaK

wavelength = 193 #ArF excimer
fraunhoferConstant = (2*1j*np.pi)/wavelength

def calculateFullFraunhofer(mask, device):
    #create all needed tensors
    mask = torch.asarray(mask, device=device)

    kx = torch.arange(-Kbound, Kbound, deltaK, dtype = torch.float32, device=device) #- deltaK
    ky = torch.arange(-Kbound, Kbound, deltaK, dtype = torch.float32, device=device)
    KX, KY = torch.meshgrid(kx,ky, indexing='xy')
    k_grid = torch.stack((KX, KY), dim=-1)
    
    xs = torch.arange(-pixelBound, pixelBound, pixelSize, dtype = torch.float32, device=device) #-pixelSize
    ys = torch.arange(-pixelBound, pixelBound, pixelSize, dtype = torch.float32, device=device)
    XS, YS = torch.meshgrid(xs,ys,indexing='xy')
    xy_grid = torch.stack((XS, YS), dim=-1)

    k_grid = k_grid.unsqueeze(2).unsqueeze(2)
    xy_grid = xy_grid.unsqueeze(0).unsqueeze(0)

    #solve the fraunhofer
    solution = torch.zeros((pixelNumber, pixelNumber), dtype=torch.complex64, device=device)
    exponent = torch.sum((k_grid * xy_grid), dim=-1) * fraunhoferConstant

    intermediate = mask * torch.exp(exponent)
    solution = torch.sum(intermediate, dim=(2,3))

    return solution, solution.cpu()
     
if __name__ == '__main__':

    if torch.cuda.is_available:
        device = torch.device('cuda')
        print(f"Using {torch.cuda.get_device_name(device)} for Fraunhofer computation")
    else:
        device = torch.device('cpu')
        print(f"Using CPU for Fraunhofer computation")

    mask = np.zeros((pixelNumber, pixelNumber), dtype=int)
    mask[0, 0:64] = 1
    mask[2, 0:64] = 1
    mask[5, 0:64] = 1
    mask[9, 0:64] = 1

    F_gpu, F = calculateFullFraunhofer(mask, device)
    F_norm_gpu = torch.real(F_gpu @ torch.conj(F_gpu))
    
    #F = F_gpu.cpu()
    F_norm = F_norm_gpu.cpu()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(mask)
    ax1.set_title('Mask')
    ax1.set_xlabel('X Position (nm)')
    ax1.set_ylabel('Y Position (nm)')

    ax2.imshow(F_norm)
    ax2.set_title('Diffraction Pattern')
    plt.show()