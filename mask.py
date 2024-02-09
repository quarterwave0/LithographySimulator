import torch

class Mask:

    def __init__(self, geometry: torch.Tensor=None, pixelSize: int=25, device: torch.device=None):

        if type(device) is torch.device:
            self.device = device
        elif torch.cuda.is_available:
            self.device = torch.device('cuda')
            print(f"No device defined for mask! Using {torch.cuda.get_device_name(self.device)}.")
        else:
            self.device = torch.device('cpu')
            print(f"No device defined for mask! Using CPU.")

        if (geometry is None or type(geometry) is not torch.Tensor) or (len(geometry.size()) != 2 or geometry.size()[0] != geometry.size[1]): #First, does it exist? Second, is it the right shape
            print("Mask not defined or invalid. Check that it is a torch tensor and is square. Using demo instead.")
            self.pixelNumber = 64
            self.geometry = torch.zeros((self.pixelNumber, self.pixelNumber), dtype=int, device=self.device)
            self.geometry[9:55, 16:20] = 1
            self.geometry[9:55, 25:29] = 1
            self.geometry[9:55, 34:38] = 1
            self.geometry[9:55, 43:47] = 1
        else:
            self.geometry = geometry.to(dtype=int, device=self.device)
            self.pixelNumber = self.geometry.size()[0]

        self.pixelSize = pixelSize

        self._pixelBound = self.pixelNumber / 2 * self.pixelSize
        self._deltaK = 4 / self.pixelNumber
        self._Kbound = self.pixelNumber / 2 * self._deltaK

    def fraunhofer(self, wavelength: int) -> torch.Tensor: #TODO: add FFT or FT selector
        fraunhoferConstant = (2*1j*torch.pi)/wavelength

        kx = torch.arange(-self._Kbound, self._Kbound, self._deltaK, dtype = torch.float32, device=self.device)
        ky = torch.arange(-self._Kbound, self._Kbound, self._deltaK, dtype = torch.float32, device=self.device)
        KX, KY = torch.meshgrid(kx,ky, indexing='xy')
        k_grid = torch.stack((KX, KY), dim=-1)
        
        xs = torch.arange(-self._pixelBound, self._pixelBound, self.pixelSize, dtype = torch.float32, device=self.device)
        ys = torch.arange(-self._pixelBound, self._pixelBound, self.pixelSize, dtype = torch.float32, device=self.device)
        XS, YS = torch.meshgrid(xs,ys,indexing='xy')
        xy_grid = torch.stack((XS, YS), dim=-1)

        k_grid = k_grid.unsqueeze(2).unsqueeze(2)
        xy_grid = xy_grid.unsqueeze(0).unsqueeze(0)

        solution = torch.zeros((self.pixelNumber, self.pixelNumber), dtype=torch.complex64, device=self.device)
        exponent = torch.sum((k_grid * xy_grid), dim=-1) * fraunhoferConstant

        intermediate = self.geometry * torch.exp(exponent)
        solution = torch.sum(intermediate, dim=(2,3))

        return solution
    
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    demoMask = Mask()
    diffractionPattern = torch.abs((demoMask.fraunhofer(193).cpu())) #In the original matlab, this is a .* (elementwise), but I think they meant abs.

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(demoMask.geometry.cpu())
    ax1.set_title('Mask')
    ax1.set_xlabel('X Position (nm)')
    ax1.set_ylabel('Y Position (nm)')

    ax2.imshow(diffractionPattern)
    ax2.set_title('Diffraction Pattern (Mag)')

    plt.show()
