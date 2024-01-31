from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import torch

from mask import Mask
from pupilDefocus import pupilf
from illumination import calculateLightSource

if torch.cuda.is_available:
    device = torch.device('cuda')
    print(f"Using {torch.cuda.get_device_name(device)}")
    print()
else:
    device = torch.device('cpu')
    print(f"Using CPU")
    print()

pupilf = torch.asarray(pupilf)

pixelNumber=64
centerpoint=pixelNumber/2-1
pixelSize = 25 #nanometers

deltaK=4/pixelNumber #fn step size

wavelength=193 #ArF excimer
fraunhoferConstant = (-2*1j*torch.pi)/wavelength
#knorm=1/wavelength                                       #checkme, what's going on here?

nmaperture=0.7
defocus=0

Kbound = pixelNumber / 2 * deltaK
pixelBound = pixelNumber / 2 * pixelSize

def calculateFullAerial(pupil, diffractionPattern):
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

    intermediate = pupil * diffractionPattern * torch.exp(exponent)
    solution = torch.sum(intermediate, axis=(2,3))

    return solution

def aerial(F, O):
    imagero = torch.zeros((pixelNumber, pixelNumber), device=device)
    imagers = torch.zeros((pixelNumber, pixelNumber), device=device)

    for j in tqdm(range(pixelNumber), desc="Abbe Image Formation", leave=True, colour='blue'):
        for k in range(pixelNumber):
            pupilshift = torch.zeros((pixelNumber, pixelNumber), dtype=torch.complex64, device=device)

            if (O[j, k] > 0):

                shiftx = -(j-centerpoint)
                shifty = -(k-centerpoint)

                if (shiftx >= 0):
                    pua = 1 + shiftx
                    pub = pixelNumber
                    puc = 1
                    pud = pixelNumber-shiftx
                else:
                    pua = 1
                    pub = pixelNumber + shiftx
                    puc = 1-shiftx
                    pud = pixelNumber

                if (shifty >= 0):
                    qua = 1 + shifty
                    qub = pixelNumber + shifty
                    quc = 1 
                    qud = pixelNumber-shifty
                else:
                    qua = 1
                    qub = pixelNumber + shifty
                    quc = 1 - shifty
                    qud = pixelNumber

                pupilshift[int(pua):int(pub), int(qua):int(qub)] = pupilf[int(puc):int(pud), int(quc):int(qud)] #from pupilfunction64_defocus

                imagerr = calculateFullAerial(pupilshift, F)

                #imagers = torch.real(imagerr * torch.conj(imagerr)) #In the original matlab, this is a .* (elementwise), so the * is intentional, but I think they meant:
                imagers = torch.abs(imagerr)
                imagero = imagero + imagers

    return imagero, imagero.cpu()

if __name__ == '__main__':

    print("Beginning simulation")
    t = time.time()

    mask = Mask()
    F = mask.fraunhofer(wavelength)
    fFraunhofer = time.time()
    print(f"Fraunhofer computation complete in: {round(fFraunhofer-t, 2)} seconds")

    O, O_cpu = calculateLightSource(device)
    fLightSource = time.time()
    print(f"Light source computation complete in: {round(fLightSource - fFraunhofer, 2)} seconds")

    aerialImage, aerialImageCPU = aerial(F, O)
    finish = time.time()
    print(f"Aerial iamge computed in {round(finish-t, 2)} seconds")

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

    ax1.imshow(torch.kron((aerialImageCPU), torch.ones((pixelSize, pixelSize))))
    ax1.set_title('Simulated Aerial Image')
    ax1.set_xlabel('X Position (nm)')
    ax1.set_ylabel('Y Position (nm)')

    ax2.imshow(torch.real(F.cpu() @ torch.conj(F.cpu())))
    ax2.set_title('Diffraction Pattern (Mag)')

    ax3.imshow(torch.kron(mask.geometry.cpu(), torch.ones((pixelSize, pixelSize))))
    ax3.set_title('Mask')
    ax3.set_xlabel('X Position (nm)')
    ax3.set_ylabel('Y Position (nm)')

    ax4.imshow(torch.kron((O_cpu), torch.ones((pixelSize, pixelSize))))
    ax4.set_title('Light Source')

    ax5.imshow(torch.real(pupilf))
    ax5.set_title('Wavefront Error (Re)')

    ax6.imshow(torch.imag(pupilf))
    ax6.set_title('Wavefront Error (Imag)')
    

    plt.show()