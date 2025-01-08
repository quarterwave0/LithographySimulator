## About
LithographySimulator is an open-source tool/toy for modeling optical lithography. 

Currently, it simulates partial coherence imaging with the Abbe formulation, Fraunhofer mask diffraction with binary masks, support for annular, quasar, and classical light sources, and arbitrary aberration modeling.

Depending on device support it uses PyTorch for GPU or CPU acceleration and can be reasonably agile when computing the aerial image. The main limiting factor will be VRAM or system RAM, as the current approach is very memory intensive.

![image](https://github.com/user-attachments/assets/bd68ebfc-20ad-4fec-95bf-31748b02c3e5)

## Goals
Right now, although it mostly works, a lot could still be done to improve it. Expect that much of this is incomplete in perpetuity, however, I do look over the code from time to time. Never say never!

- [x] Refactor architecture to be more usable (Objects, perhaps, rather than the current approach with haphazard application of global variables)
- [x] Add support for Zernike polynomial modeling of optical wavefront error for the pupil function. Currently, only defocus is supported
- [ ] Validate the correctness of the lithography model, either by testing against known-correct models or through formally validating the mathematics inside the program.
- [x] Add FFT approximation as appears in [1] alongside the classical solver
- [ ] Add GDSII/OASIS import
- [ ] Add photoresist response modeling, simple or otherwise
- [ ] 2D solver for lithography recipe generation
- [x] Allow for more complicated light sources like quasar or quadrupole

## Acknowledgment and Citations
1. T.-S. Gau et al., “Ultra-fast aerial image simulation algorithm using wavelength scaling and fast Fourier transformation to speed up calculation by more than three orders of magnitude,” JM3 22(2), 023201, SPIE (2023) [doi:10.1117/1.JMM.22.2.023201].

Note: It is very important to note that the prior paper, Gao 2023, provided the starting code for this project. The original MATLAB code is available on request from the corresponding author. I translated the code into Python in a sensible manner and improved performance, but the physics underlying this model is owed in large, but not complete, part to this paper's code.

2. B. J. Lin, Optical Lithography: Here is why, SPIE (2021).
3. X. Wu et al., “Efficient source mask optimization with Zernike polynomial functions for source representation,” Opt. Express, OE 22(4), 3924–3937, Optica Publishing Group (2014) [doi:10.1364/OE.22.003924].
4. N. B. Cobb, “Fast optical and process proximity correction algorithms for integrated circuit manufacturing,” PhD, University of California, Berkeley (1998).
5. M. Guthaus, “mguthaus/DimmiLitho,” (2021).
6. P. Evanschitzky, A. Erdmann, and T. Fuehner, “Extended Abbe approach for fast and accurate lithography imaging simulations,” in 25th European Mask and Lithography Conference, pp. 1–11 (2009) [doi:10.1117/12.835168].
7. E. Hecht, Optics, Pearson Education, Incorporated (2017).
8. C. Mack, Fundamental Principles of Optical Lithography: The Science of Microfabrication, John Wiley & Sons (2008).
