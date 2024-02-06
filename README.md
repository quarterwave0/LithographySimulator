## About
LithographySimulator (full name TBD) is an open source tool/toy for modeling optical lithography. 

In the current state, it simulates partial coherence imaging with the Abbe formulation, Fraunhofer  mask diffraction with binary masks, basic support for annular and classical light sources, and rudimentary pupil defocus support.

It uses PyTorch for either GPU and CPU acceleration, depending on devices support, and can be reasonably agile when computing the Fourier transform. The main limiting factor will be VRAM and/or system RAM, as the current approach is very memory intensive and has yet to be set up for scaling.


## Goals
Right now, although it mostly works, there's still a lot that could be done to improve it. These goals are arranged roughly in order of priority. This is a side project for me, and as such some or all of these may go uncompleted permanently.

- [ ] Refactor architecture to be more usable (Objects, perhaps, rather than the current approach with haphazard application of global variables)
- [ ] Add support for Zernike polynomial modeling of optical wavefront error for the pupil function. Currently, only defocus is supported
- [ ] Validate  correctness of the lithography model, either by testing against known-correct models or through formally validating the mathematics inside the program.
- [ ] Add FFT approximation as appears in [1] alongside the classical solver
- [ ] Fix the memory management, so that it's possible to compute more than a small field on normal systems
- [ ] Add GDSII/OASIS import
- [ ] Add photoresist response modeling, simple or otherwise
- [ ] 2D solver for lithography recipe generation
- [ ] Attach an optimizer to create a simple ILT system
- [ ] Add mask export, perhaps manufacturability checks also
- [ ] Allow for more complicated light sources like quasar or quadrupole
- [ ] Solver for phase shifting masks, likely FDTD or waveguide
- [ ] Add SOCS Hopkins, as the TCC greatly accelerates the simulation process by allowing the variation of a mask with pre-computed system optical characteristics

## Acknowledgment and Citations
1. T.-S. Gau et al., “Ultra-fast aerial image simulation algorithm using wavelength scaling and fast Fourier transformation to speed up calculation by more than three orders of magnitude,” JM3 22(2), 023201, SPIE (2023) [doi:10.1117/1.JMM.22.2.023201].

Note: It is very important to note that the prior paper, Gao 2023, provided the starting code for this project. The original MATLAB code is available on request from the corresponding author. I translated the code into Python in a sensible manner and improved performance, but the physics underlying this model is owed solely,  thus far, to this paper's code. As progress continues, the degree of similarity is likely to change.

2. B. J. Lin, Optical Lithography: Here is why, SPIE (2021).
3. X. Wu et al., “Efficient source mask optimization with Zernike polynomial functions for source representation,” Opt. Express, OE 22(4), 3924–3937, Optica Publishing Group (2014) [doi:10.1364/OE.22.003924].
4. N. B. Cobb, “Fast optical and process proximity correction algorithms for integrated circuit manufacturing,” PhD, University of California, Berkeley (1998).
5. M. Guthaus, “mguthaus/DimmiLitho,” (2021).
6. P. Evanschitzky, A. Erdmann, and T. Fuehner, “Extended Abbe approach for fast and accurate lithography imaging simulations,” in 25th European Mask and Lithography Conference, pp. 1–11 (2009) [doi:10.1117/12.835168].
7. E. Hecht, Optics, Pearson Education, Incorporated (2017).
8. C. Mack, Fundamental Principles of Optical Lithography: The Science of Microfabrication, John Wiley & Sons (2008).
