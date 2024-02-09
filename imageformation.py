import torch

def calculateAerial(pupil, maskFT, fraunhoferConstant, pixelNumber, pixelSize, device):
    pixelNumber = maskFT.size()[0]
    deltaK=4/pixelNumber #k-grid step size
    
    Kbound = pixelNumber / 2 * deltaK
    pixelBound = pixelNumber / 2 * pixelSize

    kx = torch.arange(-Kbound, Kbound, deltaK, dtype = torch.float32, device=device) 
    ky = torch.arange(-Kbound, Kbound, deltaK, dtype = torch.float32, device=device)
    KX, KY = torch.meshgrid(kx,ky, indexing='xy')
    k_grid = torch.stack((KX, KY), dim=-1)

    xs = torch.arange(-pixelBound, pixelBound, pixelSize, dtype = torch.float32, device=device)
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
    psTrim = pupilshift.narrow(0, 31, 64).narrow(1, 31, 64) #trim off the padding

    for i in range(pixelNumber):
        for j in range(pixelNumber):
            if (lightsource[i, j] > 0): #Suprisingly, this is appreciably faster than the equivalent multiplication process. TODO: Rework later to be efficient w/ zeroes on the aerial instead of doing it here
                image = image + torch.abs(calculateAerial(psTrim[:, :, j, i], maskFT, fraunhoferConstant, pixelNumber, pixelSize, device))

    return torch.abs(image)

if __name__ == '__main__':
    import time
    from matplotlib import pyplot as plt
    from pupil import pupilFunction
    from lightsource import LightSource
    from mask import Mask

    if torch.cuda.is_available:
        device = torch.device('cuda')
        print(f"Using {torch.cuda.get_device_name(device)}")
        print()
    else:
        device = torch.device('cpu')
        print(f"Using CPU")
        print()
        
    wavelength = 193 #ArF

    print("Beginning simulation")
    t = time.time()

    mask = Mask(device=device)
    maskFT = mask.fraunhofer(wavelength)
    fFraunhofer = time.time()
    print(f"Fraunhofer computation complete in: {round(fFraunhofer-t, 2)} seconds")

    lightsource = LightSource(device=device)
    ls = lightsource.generateSource()
    fLightSource = time.time()
    print(f"Light source computation complete in: {round(fLightSource - fFraunhofer, 2)} seconds")

    aerialImage = abbeImage(maskFT, pupilFunction, ls, mask.pixelSize, wavelength, device)
    finish = time.time()
    print(f"Aerial iamge computed in {round(finish-t, 2)} seconds")

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

    ax1.imshow(torch.kron((aerialImage.cpu()), torch.ones((mask.pixelSize, mask.pixelSize))))
    ax1.set_title('Simulated Aerial Image')
    ax1.set_xlabel('X Position (nm)')
    ax1.set_ylabel('Y Position (nm)')

    ax2.imshow(torch.abs(maskFT.cpu()))
    ax2.set_title('Diffraction Pattern (Mag)')

    ax3.imshow(torch.kron(mask.geometry.cpu(), torch.ones((mask.pixelSize, mask.pixelSize))))
    ax3.set_title('Mask')
    ax3.set_xlabel('X Position (nm)')
    ax3.set_ylabel('Y Position (nm)')

    ax4.imshow(ls.cpu())
    ax4.set_title('Light Source')
    ax4.set_xlabel('σ in X (λ/NA)')
    ax4.set_ylabel('σ in Y (λ/NA)')

    ax5.imshow(torch.real(pupilFunction))
    ax5.set_title('Wavefront Error (Re)')

    ax6.imshow(torch.imag(pupilFunction))
    ax6.set_title('Wavefront Error (Imag)')
    

    plt.show()