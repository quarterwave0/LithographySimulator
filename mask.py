import torch

class Mask:

    def __init__(self, geometry: torch.Tensor=None, pixelSize: int=25, device: torch.device=None):

        if type(device) is torch.device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"No device defined for mask! Using MPS.")
            print()
        elif torch.cuda.is_available:
            self.device = torch.device('cuda')
            print(f"No device defined for mask! Using {torch.cuda.get_device_name(self.device)}.")
        else:
            self.device = torch.device('cpu')
            print(f"No device defined for mask! Using CPU.")

        if (geometry is None or type(geometry) is not torch.Tensor) or (len(geometry.size()) != 2 or geometry.size()[0] != geometry.size()[1]): #First, does it exist? Second, is it the right shape
            print("Mask not defined or invalid. Check that it is a torch tensor and is square. Using demo instead.")
            self.pixelNumber = 64
            self.geometry = torch.zeros((self.pixelNumber, self.pixelNumber), dtype=torch.int16, device=self.device)
            self.geometry[9:55, 16:20] = 1
            self.geometry[9:55, 25:29] = 1
            self.geometry[9:55, 34:38] = 1
            self.geometry[9:55, 43:47] = 1
        else:
            self.geometry = geometry.to(dtype=torch.int16, device=self.device)
            self.pixelNumber = self.geometry.size()[0]

        self.pixelSize = pixelSize
        self._pixelBound = self.pixelNumber / 2 * self.pixelSize
        self.deltaK = 4 / self.pixelNumber
        self._Kbound = self.pixelNumber / 2 * self.deltaK
        
    def fraunhofer(self, wavelength: torch.float16, fft: bool) -> torch.Tensor:
        if fft:
            epsilon, N = self.calculateEpsilonN(self.deltaK, self.pixelSize, wavelength)
            return self._ffFraunhofer(epsilon, N)
        else:
            fraunhoferConstant = (2*1j*torch.pi)/wavelength

            kx = torch.arange(-self._Kbound, self._Kbound, self.deltaK, dtype = torch.float16, device=self.device)
            ky = torch.arange(-self._Kbound, self._Kbound, self.deltaK, dtype = torch.float16, device=self.device)
            KX, KY = torch.meshgrid(kx,ky, indexing='xy')
            k_grid = torch.stack((KX, KY), dim=-1)
                
            xs = torch.arange(-self._pixelBound, self._pixelBound, self.pixelSize, dtype = torch.float16, device=self.device)
            ys = torch.arange(-self._pixelBound, self._pixelBound, self.pixelSize, dtype = torch.float16, device=self.device)
            XS, YS = torch.meshgrid(xs,ys,indexing='xy')
            xy_grid = torch.stack((XS, YS), dim=-1)

            k_grid = k_grid.unsqueeze(2).unsqueeze(2)
            xy_grid = xy_grid.unsqueeze(0).unsqueeze(0)

            exponent = torch.sum((k_grid * xy_grid), dim=-1, dtype=torch.complex64) * fraunhoferConstant
            intermediate = self.geometry * torch.exp(exponent)      
            solution = torch.trapz(torch.trapz(intermediate, dim=3), dim=2)

            return solution
    
    def _nearest2SqInt(self, input: float): #find the nearest integer beta that is a power of two
        squares = torch.tensor([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384], dtype=torch.int16, device=self.device)
        return squares[torch.argmin(torch.abs(squares - input))].item()

    def calculateEpsilonN(self, deltaK, pixelSize: int, wavelength: torch.float16):
        beta = ((deltaK*pixelSize)/wavelength)**-1
        N = self._nearest2SqInt(beta)
        epsilon = N/beta

        return epsilon, N

    def _ffFraunhofer(self, epsilon, N: int) -> torch.Tensor: #this all comes from [1]

        usqMask = self.geometry.unsqueeze(0).unsqueeze(0).to(torch.float32)
        scaledMask = torch.nn.functional.interpolate(usqMask, scale_factor=epsilon, mode='bilinear').squeeze(0).squeeze(0)

        pW = ((N - self.pixelNumber) - (scaledMask.shape[0] - self.pixelNumber)) // 2
        corr = scaledMask.shape[0] % 2
        paddedMask = torch.nn.functional.pad(scaledMask, (pW, pW + corr, pW, pW + corr), mode='constant', value=0)

        standardForm = torch.fft.fftshift(paddedMask)
        fraunhoferFFT = torch.fft.fft2(standardForm, norm="backward")
        fft = torch.fft.ifftshift(fraunhoferFFT)

        trimFactor = (N - self.pixelNumber) // 2
        fft = torch.nn.functional.pad(fft, (-trimFactor, -trimFactor, -trimFactor, -trimFactor))

        return fft

    
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    demoMask = Mask()
    diffractionPattern = torch.abs((demoMask.fraunhofer(193, True).cpu())) #In the original matlab, this is a .* (elementwise), but I think they meant abs.

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(demoMask.geometry.cpu())
    ax1.set_title('Mask')
    ax1.set_xlabel('X Position (nm)')
    ax1.set_ylabel('Y Position (nm)')

    ax2.imshow(diffractionPattern)
    ax2.set_title('Diffraction Pattern (Mag)')

    plt.show()
