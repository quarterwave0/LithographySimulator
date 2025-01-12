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

def calculateFFTAerial(pf, maskFFFT, pixelNumber, N):

    pfAmplitudeProduct = pf * maskFFFT

    pW = (N-pixelNumber) // 2
    paddedPFA = torch.nn.functional.pad(pfAmplitudeProduct, (pW, pW, pW, pW), mode='constant', value=0)

    standardFormPPFA = torch.fft.fftshift(paddedPFA) #back into fft order
    abbeFFT = torch.fft.ifft2(standardFormPPFA, norm='forward') #TODO: why is this ifft2 instead of fft2 like it is in the matlab source code? Bizzare offset otherwise
    unrolledFFT = torch.fft.ifftshift(abbeFFT)

    trimmedFFT = unrolledFFT[pW:pW + pixelNumber, pW:pW + pixelNumber]

    return trimmedFFT

def abbeImage(mask, maskFT: torch.Tensor, pupilF: torch.Tensor, lightsource: torch.Tensor, pixelSize: int, deltaK: float, wavelength: torch.float16, fft: bool, device: torch.device):

    if fft:
        epsilon, N = Mask.calculateEpsilonN(self=mask, deltaK=deltaK, pixelSize=pixelSize, wavelength=wavelength)
    else:
        fraunhoferConstant = (-2 * 1j * torch.pi) / wavelength

    pixelNumber = maskFT.size()[0]

    image = torch.zeros((pixelNumber, pixelNumber), dtype=torch.complex64, device=device)

    pupilOnDevice = pupilF.to(device)
    x_y_shifts = (torch.argwhere(lightsource) - (pixelNumber // 2)).to(torch.int)
    ls_points = x_y_shifts.shape[0]

    for i in range(ls_points):
        pupil_shift = torch.roll(pupilOnDevice, shifts=(x_y_shifts[i, 0], x_y_shifts[i, 1]), dims=(0, 1))
        if not fft:
            image += torch.abs(calculateAerial(pupil_shift, maskFT, fraunhoferConstant, pixelNumber, pixelSize, device))**2
        else:
            image += torch.abs(calculateFFTAerial(pupil_shift, maskFT, pixelNumber, N))**2

    if fft:
        image = torch.abs(image) #bug in MPS
        image = torch.nn.functional.interpolate(image.unsqueeze(0).unsqueeze(0), scale_factor=(1 / epsilon),
                                                mode='bilinear').squeeze(0).squeeze(0)
        pW = (pixelNumber - round(pixelNumber / epsilon)) // 2
        corr = image.shape[0] % 2
        image = torch.nn.functional.pad(image, (pW, pW + corr, pW, pW + corr), mode='constant', value=0)

    return torch.real(image)

if __name__ == '__main__':
    import time
    from matplotlib import pyplot as plt
    from pupil import Pupil
    from lightsource import LightSource
    from mask import Mask

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using MPS")
        print()
    elif torch.cuda.is_available:
        device = torch.device('cuda')
        print(f"Using {torch.cuda.get_device_name(device)}")
        print()
    else:
        device = torch.device('cpu')
        print(f"Using CPU")
        print()

    wavelength = 193. #ArF
    aberrations = torch.tensor([0, 0, 0.01, 0, 100, 0.01, 0, 0.01, 0.01, 0.01], dtype=torch.float16, device=device)
    fft = True

    print("Beginning simulation")
    t = time.time()

    mask = Mask(device=device, pixelSize=25)
    maskFT = mask.fraunhofer(wavelength, fft)
    fFraunhofer = time.time()
    print(f"Fraunhofer computation complete in: {round(fFraunhofer-t, 2)} seconds")

    lightsource = LightSource(sigmaIn=0.4, sigmaOut=0.8, device=device)
    ls = lightsource.generateQuasar(4, -torch.pi/(4*2))
    fLightSource = time.time()
    print(f"Light source computation complete in: {round(fLightSource - fFraunhofer, 2)} seconds")

    pupil = Pupil(mask.pixelNumber, wavelength, lightsource.NA, aberrations, device=device)
    pupilFunction = pupil.generatePupilFunction()

    aerialImage = abbeImage(mask, maskFT, pupilFunction, ls, mask.pixelSize, mask.deltaK, wavelength, fft, device)

    finish = time.time()
    print(f"Aerial image computed in {round(finish-t, 2)} seconds")

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, dpi=300)

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

    ax5.imshow(torch.real(pupilFunction.cpu()))
    ax5.set_title('Wavefront Error (Re)')

    ax6.imshow(torch.imag(pupilFunction.cpu()))
    ax6.set_title('Wavefront Error (Imag)')

    fig.tight_layout()
    plt.show()