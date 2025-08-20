# Towards Hybrid Earth System Modeling: A Living Review

This page presents an alphabetical review of emerging hybrid Earth System Models (ESMs), which combine Machine Learning (ML) and physics-based components to simulate the full time evolution of climate variables in response to diverse forcings. These models hold promise for improving long-term projections of Earth's physical climate and biogeochemical cycles. Although new creative approaches continue to emerge, most designs fall into two broad categories:

1. **Hybridizing existing ESMs**  
These approaches build on established ESM codebases, retaining key physics components (e.g., the dynamical core) while replacing or improving parameterizations of hard-to-model processes (e.g., storm formation) with ML. This often involves interfacing Fortran-based codebases with Python-based ML tools. For related technical resources, see [this living review](https://github.com/TRACCS-COMPACT/hybrid_physics_AI_awesome_list/) maintained by Julien Le Sommer and Alexis Barge.

2. **Developing data-driven climate models from scratch**  
Usually referred to as "climate emulators," these efforts write prognostic equations directly in differentiable programming frameworks, incorporating explicit physical laws (e.g., conservation equations) only when needed. This is a longer-term endeavor, involving the progressive development and coupling of the atmosphere–ocean–land–cryosphere components.

If you notice any errors, omissions, or outdated information, please feel free to submit a pull request.

<ins>Author</ins>: Tom Beucler (UNIL); written in the context of [AI4PEX](https://ai4pex.org/) and the [WCRP Lighthouse Activities](https://www.wcrp-climate.org/lha-overview).

## Table of Contents
- [ACE](#ace)
- [CAMulator](#camulator)
- [CBRAIN](#cbrain)  
- [CliMA](#clima)
- [ClimSim](#climsim)
- [Corrective ML](#corrective-ml)
- [DLESyM](#dlesym)
- [Hybrid ARP-GEM](#hybrid-arp-gem)
- [Hybrid Land Surface Modeling](#hybrid-land-surface-modeling)
- [Hybrid SAM](#hybrid-sam)
- [Hybrid SPEEDY](#hybrid-speedy)
- [Hybrid WRF](#hybrid-wrf)  
- [ICON-MLe](#icon-mle)
- [LUCIE](#lucie)
- [MOM6](#mom6)
- [NCAM](#ncam)
- [NeuralGCM](#neuralgcm)
- [Ola](#ola)
- [Samudra](#samudra)

***

## ACE

The [Ai2](https://allenai.org/climate-modeling) Climate Emulator (ACE) emulates [NOAA](https://www.noaa.gov/)'s [FV3GFS atmospheric model](https://journals.ametsoc.org/view/journals/bams/100/7/bams-d-17-0246.1.xml) using spherical Fourier neural operators. ACE operates with six prognostic variables, can be forced through insolation and sea surface skin temperature, diagnoses radiative and energy fluxes at the atmosphere's boundaries, and runs on a single GPU. ACE2 improves upon ACE by enforcing global conservation of dry air mass and humidity, making it a hybrid climate model and improving climate stability and surface pressure representation. ACE2, which can be coupled to a slab ocean, is trained and tested on historical climate reanalysis (1940-2020) and 100 km-resolution [Unified Forecast System (UFS)](https://zenodo.org/records/4460292) simulations forced by historical sea surface temperatures and greenhouse gas concentrations.

### Latest simulations in [Clark, S. K., Watt-Meyer, O., Kwa, A., McGibbon, J., Henn, B., Perkins, W. A., ... & Harris, L. M. (2024). ACE2-SOM: Coupling to a slab ocean and learning the sensitivity of climate to changes in CO2. arXiv:2412.04418](https://arxiv.org/abs/2412.04418).

### See also:
- [Watt-Meyer, O., Henn, B., McGibbon, J., Clark, S. K., Kwa, A., Perkins, W. A., Wu, E., Harris, L., & Bretherton, C. S. (2025). ACE2: Accurately learning subseasonal to decadal atmospheric variability and forced responses. npj Climate and Atmospheric Science, 8.](https://www.nature.com/articles/s41612-025-01090-0)
- [Wu, E., Rebassoo, F., Pappu, P., Proistosescu, C., Nugent, J. M., ... & Bretherton, C. S. (2025). Applying the ACE2 Emulator to SST Green's Functions for the E3SMv3 Global Atmosphere Model. arXiv preprint 2505.08742.](https://arxiv.org/abs/2505.08742)
- [Kent, C., Scaife, A. A., Dunstone, N. J., Smith, D., Hardiman, S. C., Dunstan, T., & Watt-Meyer, O (2025). Skilful global seasonal predictions from a machine learning weather model trained on reanalysis data. arXiv preprint 2503.23953.](https://arxiv.org/abs/2503.23953)
- [Duncan, J. P., Wu, E., Golaz, J. C., Caldwell, P. M., Watt‐Meyer, O., Clark, S. K., ... & Bretherton, C. S. (2024). Application of the AI2 Climate Emulator to E3SMv2's global atmosphere model, with a focus on precipitation fidelity. Journal of Geophysical Research: Machine Learning and Computation, 1(3), e2024JH000136.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024JH000136)
- [Watt-Meyer, O., Dresdner, G., McGibbon, J., Clark, S. K., Henn, B., Duncan, J., ... & Bretherton, C. S. (2023). ACE: A fast, skillful learned global atmospheric model for climate prediction. arXiv preprint 2310.02074.](https://arxiv.org/abs/2310.02074)

***

## CAMulator

CAMulator v1 is a machine-learned emulator of the [Community Atmosphere Model v6 (CAM6)](https://www.cesm.ucar.edu/models/cam) that predicts atmospheric states from sea surface temperatures and solar radiation. It conserves key physical quantities, captures major climate patterns like ENSO and NAO, and runs 350× faster than CAM6—enabling large-scale, physically grounded climate simulations. While it exhibits a cold bias in high-latitude winters outside its training range, CAMulator represents a major advance toward fast, realistic ML-based climate modeling.

### Latest simulations in [Chapman, W. E., Schreck, J. S., Sha, Y., Gagne II, D. J., Kimpara, D., Zanna, L., ... & Berner, J. (2025). CAMulator: Fast Emulation of the Community Atmosphere Model. arXiv preprint 2504.06007.](https://arxiv.org/abs/2504.06007)

### See also:
- [Chapman, W. E., & Berner, J. (2025). Improving climate bias and variability via CNN-based state-dependent model-error corrections. Geophysical Research Letters, 52, e2024GL114106.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024GL114106)
- [Gettelman, A., Gagne, D. J., Chen, C. C., Christensen, M. W., Lebo, Z. J., Morrison, H., & Gantos, G. (2021). Machine learning the warm rain process. Journal of Advances in Modeling Earth Systems, 13(2), e2020MS002268.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002268)
- [Limon, G. C., & Jablonowski, C. (2023). Probing the skill of random forest emulators for physical parameterizations via a hierarchy of simple CAM6 configurations. Journal of Advances in Modeling Earth Systems, 15(6), e2022MS003395.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003395)

***

## CBRAIN

Cloud Brain (CBRAIN) aims to break the convective parameterization deadlock in the [Community Atmosphere Model (CAM)](https://www.cesm.ucar.edu/models/cam) by training neural networks to emulate the total subgrid thermodynamic time tendencies. These tendencies represent the cumulative tendencies of prognostic thermodynamic variables (temperature and specific humidity) due to subgrid-scale processes such as convection, radiation, and turbulence.

- In the aquaplanet ("ocean world") configuration, the [Super-Parameterized Community Atmosphere Model v3 (SPCAM3)](https://journals.ametsoc.org/view/journals/atsc/62/7/jas3453.1.xml?tab_body=fulltext-display) is used. Here, each coarse grid cell contains a two-dimensional convection-permitting model that explicitly resolves convection, providing the target tendencies for the neural networks.

- In the realistic geography configuration, the [Super-Parameterized Community Atmosphere Model v5 (SPCAM5)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014MS000375) is coupled with the [Community Land Model v4 (CLM4)](https://www2.cesm.ucar.edu/models/cesm1.2/clm/CLM4_Tech_Note.pdf). As in the aquaplanet configuration, each coarse grid cell includes a two-dimensional convection-permitting model that explicitly resolves convection, providing the target tendencies for the neural networks. 

### Latest simulations in [Lin, J., Yu, S., Peng, L., Beucler, T., Wong-Toi, E., Hu, Z., ... & Pritchard, M. S. (2024). Sampling Hybrid Climate Simulation at Scale to Reliably Improve Machine Learning Parameterization. arXiv preprint 2309.16177.](https://arxiv.org/abs/2309.16177)

### See also:
- [Behrens, G., Beucler, T., Iglesias-Suarez, F., Yu, S., Gentine, P., Pritchard, M., ... & Eyring, V. (2024). Improving Atmospheric Processes in Earth System Models with Deep Learning Ensembles and Stochastic Parameterizations. arXiv preprint 2402.03079.](https://arxiv.org/abs/2402.03079)
- [Rasp, S., Pritchard, M. S., & Gentine, P. (2018). Deep learning to represent subgrid processes in climate models. Proceedings of the national academy of sciences, 115(39), 9684-9689.](https://www.pnas.org/doi/full/10.1073/pnas.1810286115)  
- [Iglesias‐Suarez, F., Gentine, P., Solino‐Fernandez, B., Beucler, T., Pritchard, M., Runge, J., & Eyring, V. (2024). Causally‐informed deep learning to improve climate models and projections. Journal of Geophysical Research: Atmospheres, 129(4), e2023JD039202.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023JD039202)
- [Ott, J., Pritchard, M., Best, N., Linstead, E., Curcic, M., & Baldi, P. (2020). A fortran‐keras deep learning bridge for scientific computing. Scientific Programming, 2020(1), 8888811.](https://onlinelibrary.wiley.com/doi/full/10.1155/2020/8888811)
- [Beucler, T., Gentine, P., Yuval, J., Gupta, A., Peng, L., Lin, J., ... & Pritchard, M. (2024). Climate-invariant machine learning. Science Advances, 10(6), eadj7250.](https://www.science.org/doi/10.1126/sciadv.adj7250)
- [Mooers, G., Pritchard, M., Beucler, T., Ott, J., Yacalis, G., Baldi, P., & Gentine, P. (2021). Assessing the potential of deep learning for emulating cloud superparameterization in climate models with real‐geography boundary conditions. Journal of Advances in Modeling Earth Systems, 13(5), e2020MS002385.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002385)
- [Gentine, P., Pritchard, M., Rasp, S., Reinaudi, G., & Yacalis, G. (2018). Could machine learning break the convection parameterization deadlock?. Geophysical Research Letters, 45(11), 5742-5751.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018GL078202)

***

## [CliMA](https://clima.caltech.edu/)

The Climate Modeling Alliance (CliMA) is "building a new Earth system model that leverages recent advances in computational and data sciences to learn directly from a wealth of Earth observations from space and the ground." Its atmospheric component adopts a [generalized version of the Eddy-Diffusivity Mass-Flux framework](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2017MS001162) and relies on Bayesian inference for parameter calibration and uncertainty quantification through the ["Calibrate, Emulate, Sample" framework](https://www.sciencedirect.com/science/article/abs/pii/S0021999120304903?via%3Dihub). Its oceanic component is based on the ["Oceananigans" model](https://joss.theoj.org/papers/10.21105/joss.02018), which is designed for the numerical simulation of incompressible, stratified, rotating fluid flows on CPUs and GPUs.

### Latest simulations in [Christopoulos, C., Lopez-Gomez, I., Beucler, T., Cohen, Y., Kawczynski, C., Dunbar, O., & Schneider, T. (2024). Online Learning of Entrainment Closures in a Hybrid Machine Learning Parameterization. Authorea Preprints.](https://essopenarchive.org/doi/full/10.22541/essoar.171804905.55213571/)

### See also:
- [Lopez‐Gomez, I., Christopoulos, C., Langeland Ervik, H. L., Dunbar, O. R., Cohen, Y., & Schneider, T. (2022). Training physics‐based machine‐learning parameterizations with gradient‐free ensemble Kalman methods. Journal of Advances in Modeling Earth Systems, 14(8), e2022MS003105.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022MS003105)
- [Lima, M., Deck, K., Dunbar, O. R., & Schneider, T. (2024). Toward Routing River Water in Land Surface Models with Recurrent Neural Networks. arXiv preprint 2404.14212.](https://arxiv.org/abs/2404.14212)
- [Charbonneau, A., Deck, K., & Schneider, T. (2023). A physics-constrained neural differential equation for data-driven seasonal snowpack forecasting. Artificial Intelligence for the Earth Systems, in review.](https://climate-dynamics.org/publications/2732/)
- [Schneider, T., Lan, S., Stuart, A., & Teixeira, J. (2017). Earth system modeling 2.0: A blueprint for models that learn from observations and targeted high‐resolution simulations. Geophysical Research Letters, 44(24), 12-396.](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2017GL076101)

***

## [ClimSim](https://leap-stc.github.io/ClimSim/README.html)

ClimSim, the first benchmark dataset for hybrid ML-physics climate emulation, includes simulation data from the [Energy Exascale Earth System Model Multi-scale Modeling Framework (E3SM-MMF)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019MS001863). E3SM-MMF embeds GPU-accelerated cloud-resolving models within each grid cell of [E3SM](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018MS001350) and uses [explicit scalar momentum transport](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003206) to ensure the quality of subgrid-scale fluxes. ClimSim provides billions of multivariate input/output vector pairs, capturing the aggregate effect of cloud-resolving models on E3SM's macro-scale state. ClimSim also inspired a [Kaggle competition](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim) and includes an end-to-end workflow for developing hybrid ML-physics simulators. 

### Latest simulations in [Hu, Z., Subramaniam, A., Kuang, Z., Lin, J., Yu, S., Hannah, W. M., ... & Pritchard, M. S. (2024). Stable Machine-Learning Parameterization of Subgrid Processes with Real Geography and Full-physics Emulation. arXiv preprint 2407.00124.](https://arxiv.org/abs/2407.00124)

### See also:
- [Yu, S., Hu, Z., Subramaniam, A., Hannah, W., Peng, L., Lin, J., Bhouri, M. A., Gupta, R., Lütjens, B., Will, J. C., Behrens, G., Busecke, J. J. M., Loose, N., Stern, C. I., Beucler, T., Harrop, B., Heuer, H., Hillman, B. R., Jenney, A., ... Pritchard, M. (2024). ClimSim-Online: A large multi-scale dataset and framework for hybrid ML-physics climate emulation. arXiv preprint 2306.08754](https://arxiv.org/abs/2306.08754)
- [Yu, S., Hannah, W., Peng, L., Lin, J., Bhouri, M. A., Gupta, R., ... & Pritchard, M. (2024). ClimSim: A large multi-scale dataset for hybrid physics-ML climate emulation. Advances in Neural Information Processing Systems, 36.](https://neurips.cc/virtual/2023/poster/73569)

***

## Corrective ML

Building on early efforts to enhance subgrid-scale physics through machine learning with near-global storm-resolving aquaplanet simulations, [AI2](https://allenai.org/climate-modeling) has developed a series of data-driven solutions to improve the (thermo)dynamics of FV3-GFS, the atmospheric component of the [Unified Forecast System (UFS)](https://zenodo.org/records/4460292). The latest efforts focused on learning apparent dynamic tendencies to nudge temperature and humidity toward a reference state derived from a [global storm-resolving GFDL X-SHiELD simulation](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022JD037823), informally called "Corrective ML."

### Latest simulations in [Watt‐Meyer, O., Brenowitz, N. D., Clark, S. K., Henn, B., Kwa, A., McGibbon, J., ... & Bretherton, C. S. (2024). Neural network parameterization of subgrid‐scale physics from a realistic geography global storm‐resolving simulation. Journal of Advances in Modeling Earth Systems, 16(2), e2023MS003668.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003668)

### See also:  
- [Kwa, A., Clark, S. K., Henn, B., Brenowitz, N. D., McGibbon, J., Watt‐Meyer, O., ... & Bretherton, C. S. (2023). Machine‐learned climate model corrections from a global storm‐resolving model: Performance across the annual cycle. Journal of Advances in Modeling Earth Systems, 15(5), e2022MS003400.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003400)
- [Clark, S. K., Brenowitz, N. D., Henn, B., Kwa, A., McGibbon, J., Perkins, W. A., ... & Harris, L. M. (2022). Correcting a 200 km resolution climate model in multiple climates by machine learning from 25 km resolution simulations. Journal of Advances in Modeling Earth Systems, 14(9), e2022MS003219.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003219)
- [Bretherton, C. S., Henn, B., Kwa, A., Brenowitz, N. D., Watt‐Meyer, O., McGibbon, J., ... & Harris, L. (2022). Correcting coarse‐grid weather and climate models by machine learning from global storm‐resolving simulations. Journal of Advances in Modeling Earth Systems, 14(2), e2021MS002794.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002794)
- [Chen, T. C., Penny, S. G., Whitaker, J. S., Frolov, S., Pincus, R., & Tulich, S. (2022). Correcting systematic and state‐dependent errors in the NOAA FV3‐GFS using neural networks. Journal of Advances in Modeling Earth Systems, 14(11), e2022MS003309.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003309)
- [Watt‐Meyer, O., Brenowitz, N. D., Clark, S. K., Henn, B., Kwa, A., McGibbon, J., ... & Bretherton, C. S. (2021). Correcting weather and climate models by machine learning nudged historical simulations. Geophysical Research Letters, 48(15), e2021GL092555.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL092555)
- [Sanford, C., Kwa, A., Watt‐Meyer, O., Clark, S. K., Brenowitz, N., McGibbon, J., & Bretherton, C. (2023). Improving the reliability of ML‐corrected climate models with novelty detection. Journal of Advances in Modeling Earth Systems, 15(11), e2023MS003809.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003809)
- [Brenowitz, N. D., Henn, B., McGibbon, J., Clark, S. K., Kwa, A., Perkins, W. A., ... & Bretherton, C. S. (2020). Machine learning climate model dynamics: Offline versus online performance. NeurIPS 2020 CCAI workshop.](https://arxiv.org/abs/2011.03081)
- [Brenowitz, N. D., & Bretherton, C. S. (2019). Spatially extended tests of a neural network parametrization trained by coarse‐graining. Journal of Advances in Modeling Earth Systems, 11(8), 2728-2744.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019MS001711)
- [Brenowitz, N. D., & Bretherton, C. S. (2018). Prognostic validation of a neural network unified physics parameterization. Geophysical Research Letters, 45(12), 6289-6298.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018GL078510)

***

## DLESyM

The Deep Learning Earth SYstem Model (DLESyM) couples a [deep learning weather prediction model](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS004021) with a deep learning prognostic sea surface temperature (SST) model. The resulting model is parsimonious, using an order of magnitude fewer prognostic variables per grid point (9 for the atmosphere and only SST for the ocean), partly due to its use of the [Hierarchical Equal Area isoLatitude Pixelization (HEALPix)](https://arxiv.org/abs/astro-ph/0409513) spatial grid.

### Latest simulations in [Cresswell-Clay, N., Liu, B., Durran, D., Liu, A., Espinosa, Z. I., Moreno, R., & Karlbauer, M. (2024). A deep learning earth system model for stable and efficient simulation of the current climate. arXiv preprint 2409.16247.](https://arxiv.org/abs/2409.16247)

***

## Hybrid ARP-GEM

Hybrid ARP-GEM1 combines the dynamical core of the new global atmospheric model [ARP-GEM1](https://arxiv.org/abs/2409.19083) (Global, Efficient, and Multiscale version of [ARPEGE]( https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002075) version 1) with neural network-based parameterizations. It employs the Python interface of the Message Passing Interface-based “field-exchange” method [OASIS3]( https://gitlab.com/cerfacs/oasis3-mct/-/tree/OASIS3-MCT_5.0), enabling neural network integration on heterogenous High-Performance Computing (HPC) architectures. Initial prototypes emulate deep learning parameterization, and Hybrid ARP-GEM1's modular design enables the coupling of diverse data-driven parameterizations in the near term.

### Latest simulations in [Balogh, B., Saint-Martin, D., & Geoffroy, O. (2024). Online test of a neural network deep convection parameterization in ARP-GEM1. arXiv preprint 2410.21920](https://arxiv.org/abs/2410.21920)

***

## Hybrid Land Surface Modeling

Attempts to augment global land surface models with ML components that learn hard-to-parameterize processes are emerging, from [data-driven fire schemes coupled to carbon fluxes and vegetation](https://gmd.copernicus.org/articles/18/4103/2025/gmd-18-4103-2025.html) to [emulators of surface fluxes (e.g., evaporation)](https://www.nature.com/articles/s41467-022-29543-7). ML models are trained on observations and high-resolution products, often with physical constraints to preserve the host model's water/energy/carbon budgets.

### See:
- [Liu, Y., Huang, H., Wang, S. C., Zhang, T., Xu, D., & Chen, Y. (2025). ELM2. 1-XGBfire1. 0: improving wildfire prediction by integrating a machine learning fire model in a land surface model. Geoscientific Model Development, 18(13), 4103-4117.](https://gmd.copernicus.org/articles/18/4103/2025/gmd-18-4103-2025.html)
- [Son, R., Stacke, T., Gayler, V., Nabel, J. E., Schnur, R., Alonso, L., ... & Carvalhais, N. (2024). Integration of a Deep‐Learning‐Based Fire Model Into a Global Land Surface Model. Journal of Advances in Modeling Earth Systems, 16(1), e2023MS003710.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003710)
- [Koppa, A., Rains, D., Hulsman, P., Poyatos, R., & Miralles, D. G. (2022). A deep learning-based hybrid model of global terrestrial evaporation. Nature communications, 13(1), 1912.](https://www.nature.com/articles/s41467-022-29543-7)

***

## Hybrid SAM

Using the [hypohydrostatic configuration](https://journals.ametsoc.org/view/journals/atsc/73/2/jas-d-15-0049.1.xml) of the [System for Atmospheric Modeling (SAM)](https://you.stonybrook.edu/somas/sam/), quasi-global aquaplanet simulations can represent convection and large-scale circulation simultaneously at horizontal resolutions as coarse as 12 km. This provides an ideal testbed for machine learning parameterization approaches, such as leveraging [non-local information across grid columns to model subgrid momentum fluxes](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS002984), employing [reduced-precision computations](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL091363), and making parameterizations [scale-aware](https://www.nature.com/articles/s41467-020-17142-3). Numerical stability is ensured through a per-process flux prediction framework and a short integration timestep, which allow the use of tailored SAM prognostic equations and precipitation diagnostics. 

### Latest simulations in [Yuval, J., & O’Gorman, P. A. (2023). Neural‐network parameterization of subgrid momentum transport in the atmosphere. Journal of Advances in Modeling Earth Systems, 15(4), e2023MS003606.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003606)

### See also:
- [Yuval, J., O'Gorman, P. A., & Hill, C. N. (2021). Use of neural networks for stable, accurate and physically consistent parameterization of subgrid atmospheric processes with good performance at reduced precision. Geophysical Research Letters, 48(6), e2020GL091363.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL091363)
- [Wang, P., Yuval, J., & O’Gorman, P. A. (2022). Non‐local parameterization of atmospheric subgrid processes with neural networks. Journal of Advances in Modeling Earth Systems, 14(10), e2022MS002984.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS002984)
- [Yuval, J., & O’Gorman, P. A. (2020). Stable machine-learning parameterization of subgrid processes for climate modeling at a range of resolutions. Nature communications, 11(1), 3295.](https://www.nature.com/articles/s41467-020-17142-3)

***

## Hybrid SPEEDY

Hybrid SPEEDY builds on the combined hybrid-parallel prediction [(CHyPP)](https://pubs.aip.org/aip/cha/article/30/5/053111/1030728/Combining-machine-learning-with-knowledge-based) framework, coupling reservoir computing with the Simplified Parameterization, primitive‐Equation Dynamics [(SPEEDY)](https://link.springer.com/article/10.1007/s00382-002-0268-2) atmospheric model. Beyond the standard SPEEDY atmospheric state (3.75°×3.75°, 8 levels), it prognoses 6-h accumulated precipitation, sea-surface temperature (SST), and 0-300 m upper-ocean heat content via a [hybridized recurrent neural network](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002712). Hybrid SPPEDY captures variability from intraseasonal (MJO, convectively coupled waves) to interannual (ENSO) scales. 

### Latest simulations in [Patel, D., Arcomano, T., Hunt, B., Szunyogh, I., & Ott, E. (2025). Prediction beyond the medium range with an atmosphere-ocean model that combines physics-based modeling and machine learning. Journal of Advances in Modeling Earth Systems, 17, e2024MS004480.](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2024MS004480)

### See also:
- [Arcomano, T., Szunyogh, I., Wikner, A., Hunt, B. R., & Ott, E. (2023). A hybrid atmospheric model incorporating machine learning can capture dynamical processes not captured by its physics‐based component. Geophysical Research Letters, 50(8), e2022GL102649.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022GL102649)
- [Arcomano, T., Szunyogh, I., Wikner, A., Pathak, J., Hunt, B. R., & Ott, E. (2022). A hybrid approach to atmospheric modeling that combines machine learning with a physics-based numerical model. Journal of Advances in Modeling Earth Systems, 14, e2021MS002712.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002712)
- [Wikner, A., Pathak, J., Hunt, B., Girvan, M., Arcomano, T., Szunyogh, I., ... & Ott, E. (2020). Combining machine learning with knowledge-based modeling for scalable forecasting and subgrid-scale closure of large, complex, spatiotemporal systems. Chaos: An Interdisciplinary Journal of Nonlinear Science, 30(5).](https://pubs.aip.org/aip/cha/article/30/5/053111/1030728/Combining-machine-learning-with-knowledge-based)

***

## Hybrid WRF

Hybrid versions of the [Weather Research & Forecasting Model (WRF)](https://www.mmm.ucar.edu/models/wrf) preserve WRF's dynamical core while replacing uncertain WRF parameterizations with ML surrogates. This includes the emulation of radiation, convection, cloud microphysics, and select chemical formation processes.

### See:
#### Radiation
- [Zhong, X., Ma, Z., Yao, Y., Xu, L., Wu, Y., & Wang, Z. (2023). WRF–ML v1. 0: a bridge between WRF v4. 3 and machine learning parameterizations and its application to atmospheric radiative transfer. Geoscientific Model Development, 16(1), 199-209.](https://gmd.copernicus.org/articles/16/199/2023/)
- [Mu, B., Chen, L., Yuan, S., & Qin, B. (2023). A radiative transfer deep learning model coupled into WRF with a generic fortran torch adaptor. Frontiers in Earth Science, 11, 1149566.](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2023.1149566/full)
- [Ko, J. S., Kim, S. H., Jo, J., Jang, S., Song, H. J., & Lim, K. S. S. Neural Network Emulator for the Bulk-Type Cloud Microphysics Scheme: Application of the Melting Processes. Available at SSRN 4857342.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4857342)

#### Convection
- [Zhong, X., Yu, X., & Li, H. (2024). Machine learning parameterization of the multi-scale Kain–Fritsch (MSKF) convection scheme and stable simulation coupled in the Weather Research and Forecasting (WRF) model using WRF–ML v1. 0. Geoscientific Model Development, 17(9), 3667-3685.](https://gmd.copernicus.org/articles/17/3667/2024/)

#### Cloud microphysics
- [Georgakaki, P., & Nenes, A. (2024). RaFSIP: Parameterizing ice multiplication in models using a machine learning approach. Journal of Advances in Modeling Earth Systems, 16(6), e2023MS003923.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS003923)
- [Takeishi, A., & Wang, C. (2024). Parameterizing Raindrop Formation Using Machine Learning. Monthly Weather Review, 152(3), 649-665.](https://journals.ametsoc.org/view/journals/mwre/152/3/MWR-D-22-0175.1.xml)

#### Chemistry

- [Sharma, H., Shrivastava, M., & Singh, B. (2023). Physics informed deep neural network embedded in a chemical transport model for the Amazon rainforest. npj Climate and Atmospheric Science, 6(1), 28.](https://www.nature.com/articles/s41612-023-00353-y)

***

## ICON-MLe

The Machine Learning–enhanced (MLe) ICOsahedral Non-hydrostatic (ICON) climate model builds on ICON-A at approximately 80 km resolution (R2B5), [primarily using the ECHAM physics package](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2017MS001242). Suboptimal parameterizations, such as those for cloud cover and convection, are progressively replaced with improved data-driven versions trained on high-fidelity model outputs (e.g., [DYAMOND](https://www.esiwace.eu/the-project/past-phases/dyamond-initiative), [ClimSim](#climsim)) and observations. This results in a flexible, hybrid AI-climate version of ICON.

### Latest simulations in [Grundner, A., Beucler, T., Savre, J., Lauer, A., Schlund, M., & Eyring, V. (2025). Reduced Cloud Cover Errors in a Hybrid AI-Climate Model Through Equation Discovery And Automatic Tuning. arXiv preprint 2505.04358.](https://arxiv.org/abs/2505.04358)

### See also:
- [Heuer, H., Schwabe, M., Gentine, P., Giorgetta, M. A., & Eyring, V. (2024). Interpretable multiscale machine learning-based parameterizations of convection for ICON. Journal of Advances in Modeling Earth Systems, 16, e2024MS004398.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024MS004398)
- [Grundner, A., Beucler, T., Gentine, P., & Eyring, V. (2024). Data‐driven equation discovery of a cloud cover parameterization. Journal of Advances in Modeling Earth Systems, 16(3), e2023MS003763.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS003763)
- [Grundner, A., Beucler, T., Gentine, P., Iglesias‐Suarez, F., Giorgetta, M. A., & Eyring, V. (2022). Deep learning based cloud cover parameterization for ICON. Journal of Advances in Modeling Earth Systems, 14(12), e2021MS002959.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002959)

***

## LUCIE

The Lightweight Uncoupled ClImate Emulator [(LUCIE)](https://arxiv.org/abs/2405.16297) is a fully data-driven, Spherical Fourier Neural Operator (SFNO)-based emulator of the [ERA5 meteorological reanalysis](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.3803). By prognosing only 5 single-level variables, LUCIE allows large ensembles via weight perturbation, which approximate reproduce the main global climate's modes of variability. 

### Latest simulations in [Guan, H., Arcomano, T., Chattopadhyay, A., & Maulik, R. (2024). LUCIE: A Lightweight Uncoupled ClImate Emulator with long-term stability and physical consistency for O (1000)-member ensembles. arXiv preprint 2405.16297.](https://arxiv.org/abs/2405.16297)


***

## MOM6

The Modular Ocean Model version 6 (MOM6) is a major testbed for hybrid ocean modeling, where ML augments or replaces oceanic parameterizations to correct systematic biases. This includes [bias-correcting sea ice concentration](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL106776), parameterizing [mesoscale eddy fluxes](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS004104) and [vertical mixing](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS003890), all of which improve the emergent statistics of the coupled ML-dynamical ocean simulation.

### Latest simulations in [Gregory, W., Bushuk, M., Zhang, Y. F., Adcroft, A., Zanna, L., McHugh, C., & Jia, L. (2025). Advancing global sea ice prediction capabilities using a fully-coupled climate model with integrated machine learning. arXiv preprint:2505.18328.](https://arxiv.org/abs/2505.18328)

### See also:
- [Perezhogin, P., Adcroft, A., & Zanna, L. (2025). Generalizable neural-network parameterization of mesoscale eddies in idealized and global ocean models. arXiv preprint:2505.08900.](https://arxiv.org/abs/2505.08900)
- [Gregory, W., Bushuk, M., Zhang, Y., Adcroft, A., & Zanna, L. (2024). Machine learning for online sea ice bias correction within global ice‐ocean simulations. Geophysical Research Letters, 51(3), e2023GL106776.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL106776)
- [Zhang, C., Perezhogin, P., Adcroft, A., & Zanna, L. (2025). Addressing out‐of‐sample issues in multi‐layer convolutional neural‐network parameterization of mesoscale eddies applied near coastlines. Journal of Advances in Modeling Earth Systems, 17(5), e2024MS004819.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004819)
- [Perezhogin, P., Zhang, C., Adcroft, A., Fernandez‐Granda, C., & Zanna, L. (2024). A stable implementation of a data‐driven scale‐aware mesoscale parameterization. Journal of Advances in Modeling Earth Systems, 16(10), e2023MS004104.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS004104)
- [Zhang, C., Perezhogin, P., Gultekin, C., Adcroft, A., Fernandez‐Granda, C., & Zanna, L. (2023). Implementation and evaluation of a machine learned mesoscale eddy parameterization into a numerical ocean circulation model. Journal of Advances in Modeling Earth Systems, 15(10), e2023MS003697.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS003697)  
- [Sane, A., Reichl, B. G., Adcroft, A., & Zanna, L. (2023). Parameterizing vertical mixing coefficients in the ocean surface boundary layer using neural networks. Journal of Advances in Modeling Earth Systems, 15(10), e2023MS003890.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS003890)
- [Partee, S., Ellis, M., Rigazzi, A., Shao, A. E., Bachman, S., Marques, G., & Robbins, B. (2022). Using machine learning at scale in numerical simulations with SmartSim: An application to ocean climate modeling. Journal of Computational Science, 62, 101707.](https://www.sciencedirect.com/science/article/pii/S1877750322001065#b35)

***

## NCAM

The Neural [Community Atmosphere Model](https://www.cesm.ucar.edu/models/cam) (NCAM) uses a real geography setup and combines residual and convolutional strategies, with memory (two time steps are sufficient). It replaces only the moist physics components of CAM, including deep and shallow convection and latent heating from microphysics, but does not replace radiation. 

### Latest simulations in [Han, Y., Zhang, G. J., & Wang, Y. (2023). An ensemble of neural networks for moist physics processes, its generalizability and stable integration. Journal of Advances in Modeling Earth Systems, 15(10), e2022MS003508.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003508)

### See also:   
- [Wang, X., Han, Y., Xue, W., Yang, G., & Zhang, G. J. (2022). Stable climate simulations using a realistic general circulation model with neural network parameterizations for atmospheric moist physics and radiation processes. Geoscientific Model Development, 15(9), 3923-3940.](https://gmd.copernicus.org/articles/15/3923/2022/)
- [Han, Y., Zhang, G. J., Huang, X., & Wang, Y. (2020). A moist physics parameterization based on deep learning. Journal of Advances in Modeling Earth Systems, 12(9), e2020MS002076.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020MS002076) 

***

## [NeuralGCM](https://neuralgcm.readthedocs.io/en/latest/)

The Neural General Circulation Model (NeuralGCM) is based on a differentiable pseudo-spectral [dynamical core implemented in JAX](https://github.com/neuralgcm/dinosaur). Processes not represented by the core are learned in an end-to-end manner using a single-column parameterization that optimizes [medium-range weather forecasting](https://github.com/google-research/weatherbench2). NeuralGCM enables stable, multi-decadal simulations of climate variability under prescribed sea surface temperatures and is being updated to accurately simulate [observed global precipitation fields](https://arxiv.org/abs/2412.11973). 

### Latest simulations in [Kochkov, D., Yuval, J., Langmore, I., Norgaard, P., Smith, J., Mooers, G., ... & Hoyer, S. (2024). Neural general circulation models for weather and climate. Nature, 1-7.](https://www.nature.com/articles/s41586-024-07744-y)

### See also:  
- [Yuval, J., Langmore, I., Kochkov, D., & Hoyer, S. (2024). Neural general circulation models optimized to predict satellite-based precipitation observations. arXiv preprint 2412.11973](https://arxiv.org/abs/2412.11973).

***

## Ola

The Ocean-linked-atmosphere (Ola) model is a 0.25°-resolution model coupling two autoregressive spherical Fourier neural operators for the atmosphere and the ocean.

### Latest simulations in [Wang, C., Pritchard, M. S., Brenowitz, N., Cohen, Y., Bonev, B., Kurth, T., ... & Pathak, J. (2024). Coupled Ocean-Atmosphere Dynamics in a Machine Learning Earth System Model. arXiv preprint 2406.08632.](https://arxiv.org/abs/2406.08632)

***

## Samudra

Trained on coupled simulations using the [OM4 model](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019MS001726), Samudra is a fully data-driven, global 3D ocean emulator that autoregressively predicts sea-surface height, horizontal velocity, potential temperature, and salinity. Samudra's architecture is based on the [ConvNeXt U-Net architecture](https://arxiv.org/abs/2405.18585) and reproduces interannual variability signatures such as ENSO while accelerating the parent model by a factor ~150.

### Latest simulations in [Dheeshjith, S., Subel, A., Adcroft, A., Busecke, J., Fernandez‐Granda, C., Gupta, S., & Zanna, L. (2025). Samudra: An AI global ocean emulator for climate. Geophysical Research Letters, 52(10), e2024GL114318.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024GL114318?af=R)