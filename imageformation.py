import torch

def calculateAerial(pupil, maskFT, fraunhoferConstant, pixelNumber, pixelSize, device):
    pixelNumber = maskFT.size()[0]
    deltaK=4/pixelNumber #k-grid step size
    
    Kbound = pixelNumber / 2 * deltaK
    pixelBound = pixelNumber / 2 * pixelSize

    kx = torch.arange(-Kbound, Kbound, deltaK, dtype = torch.float16, device=device) 
    ky = torch.arange(-Kbound, Kbound, deltaK, dtype = torch.float16, device=device)
    KX, KY = torch.meshgrid(kx,ky, indexing='xy')
    k_grid = torch.stack((KX, KY), dim=-1)

    xs = torch.arange(-pixelBound, pixelBound, pixelSize, dtype = torch.float16, device=device)
    ys = torch.arange(-pixelBound, pixelBound, pixelSize, dtype = torch.float16, device=device)
    XS, YS = torch.meshgrid(xs,ys,indexing='xy')
    xy_grid = torch.stack((XS, YS), axis=-1)

    k_grid = k_grid.unsqueeze(2).unsqueeze(2)
    xy_grid = xy_grid.unsqueeze(0).unsqueeze(0)

    solution = torch.zeros((pixelNumber, pixelNumber), dtype=torch.complex64, device=device)
    exponent = torch.sum((k_grid * xy_grid), dim=-1, dtype=torch.complex64) * fraunhoferConstant

    intermediate = pupil * maskFT * torch.exp(exponent)

    solution = torch.trapz(torch.trapz(intermediate, dim=3), dim=2)

    return solution

def calculateFFTAerial(pf, maskFFFT, pixelNumber, epsilon, N, device):

    pfAmplitudeProduct = pf * maskFFFT

    paddingWidth = (N-pixelNumber) // 2
    padder = torch.nn.ConstantPad2d(paddingWidth, 0)
    paddedPFA = padder(pfAmplitudeProduct)

    standardFormPPFA = torch.fft.fftshift(paddedPFA) #back into fft order
    abbeFFT = torch.fft.ifft2(standardFormPPFA, norm='forward') #TODO: why is this ifft2 instead of fft2 like it is in the matlab source code? Bizzare offset otherwise.
    unrolledFFT = torch.fft.ifftshift(abbeFFT)
    usqAbbe = torch.abs(unrolledFFT.unsqueeze(0).unsqueeze(0)).to(torch.float32)

    aerial = torch.nn.functional.interpolate(usqAbbe, scale_factor=(1/epsilon), mode='bilinear').to(torch.float16).squeeze(0).squeeze(0)

    extraSize = (aerial.size()[0] - (pixelNumber+(2*paddingWidth))) // 2 + paddingWidth
    trimmedAerial = aerial[extraSize:extraSize+pixelNumber, extraSize:extraSize+pixelNumber] #TODO: This is busted

    return trimmedAerial

def abbeImage(mask, maskFT: torch.Tensor, pupilF: torch.Tensor, lightsource: torch.Tensor, pixelSize: int, deltaK: float, wavelength:int, fft: bool, device: torch.device):

    if fft:
        epsilon, N = Mask.calculateEpsilonN(self=mask, deltaK=deltaK, pixelSize=pixelSize, wavelength=wavelength)

    pixelNumber = maskFT.size()[0]
    fraunhoferConstant = (-2*1j*torch.pi)/wavelength

    image = torch.zeros((pixelNumber, pixelNumber), dtype=torch.complex64, device=device)

    pupilOnDevice = pupilF.to(device)
    pupilshift = torch.zeros((pixelNumber*2, pixelNumber*2, pixelNumber, pixelNumber), dtype=torch.complex64, device=device)

    a = torch.arange(0, pixelNumber, 1, dtype=int, device=device)
    b = torch.arange(0, pixelNumber, 1, dtype=int, device=device)
    A, B = torch.meshgrid((a, b), indexing='xy')

    i = torch.arange(0, pixelNumber, 1, dtype=int, device=device)
    j = torch.arange(0, pixelNumber, 1, dtype=int, device=device)
    I, J = torch.meshgrid((i, j), indexing='xy')
    Iu = I.unsqueeze(-1).unsqueeze(-1)
    Ju = J.unsqueeze(-1).unsqueeze(-1)
    
    pupilshift[(A+Iu), (B+Ju), Iu, Ju] = pupilOnDevice
    #there are Px x Px fields of Px x Px (1) where each (1) field has the pupil function where it is illuminated by the light source at a different position within it.
    # A and B represent every position in our un-padded field, and I and J respectively slide the pupil around our padded pupilshift space by broadcasting the AB grid across itself grid through addition
    # such that A begins at 1 for I = 1, etc.
    psTrim = pupilshift.narrow(0, 31, 64).narrow(1, 31, 64) #trim off the padding TODO: make this dynamic

    for i in range(pixelNumber):
        for j in range(pixelNumber):
            if (lightsource[i, j] > 0): #Suprisingly, this is appreciably faster than the equivalent multiplication process. TODO: Rework later to be efficient w/ zeroes on the aerial instead of doing it here

                if not fft:
                    image = image + torch.abs(calculateAerial(psTrim[:, :, j, i], maskFT, fraunhoferConstant, pixelNumber, pixelSize, device))
                else:
                    new = calculateFFTAerial(psTrim[:, :, j, i], maskFT, pixelNumber, epsilon, N, device)
                    image = image + new

    return torch.abs(image)

if __name__ == '__main__':
    import time
    from matplotlib import pyplot as plt
    from pupil import Pupil
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
    aberrations = [0, 0, 0.01, 0, 100, 0.01, 0, 0.01, 0.01, 0.01]

    print("Beginning simulation")
    t = time.time()

    mask = Mask(device=device)
    maskFFFT = mask.fraunhofer(wavelength, True)
    fFraunhofer = time.time()
    print(f"Fraunhofer computation complete in: {round(fFraunhofer-t, 2)} seconds")

    lightsource = LightSource(sigmaIn=0.4, sigmaOut=0.8, device=device)
    ls = lightsource.generateQuasar(4, -torch.pi/(4*2))
    fLightSource = time.time()
    print(f"Light source computation complete in: {round(fLightSource - fFraunhofer, 2)} seconds")

    pupil = Pupil(mask.pixelNumber, wavelength, lightsource.NA, aberrations, device=device)
    pupilFunction = pupil.generatePupilFunction()

    aerialImage = abbeImage(mask, maskFFFT, pupilFunction, ls, mask.pixelSize, mask.deltaK, wavelength, True, device) #- abbeImage(mask, maskFFFT, pupilFunction, ls, mask.pixelSize, mask.deltaK, wavelength, False, device)
    finish = time.time()
    print(f"Aerial iamge computed in {round(finish-t, 2)} seconds")

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

    ax1.imshow(torch.kron((aerialImage.cpu()), torch.ones((mask.pixelSize, mask.pixelSize))))
    ax1.set_title('Simulated Aerial Image')
    ax1.set_xlabel('X Position (nm)')
    ax1.set_ylabel('Y Position (nm)')

    ax2.imshow(torch.abs(maskFFFT.cpu()))
    ax2.set_title('Diffraction Pattern (Mag)')

    ax3.imshow(torch.kron(mask.geometry.cpu(), torch.ones((mask.pixelSize, mask.pixelSize))))
    ax3.set_title('Mask')
    ax3.set_xlabel('X Position (nm)')
    ax3.set_ylabel('Y Position (nm)')

    ax4.imshow(ls.cpu())
    ax4.set_title('Light Source')

    ax5.imshow(torch.real(pupilFunction.cpu()))
    ax5.set_title('Wavefront Error (Re)')

    ax6.imshow(torch.imag(pupilFunction.cpu()))
    ax6.set_title('Wavefront Error (Imag)')
    

    plt.show()