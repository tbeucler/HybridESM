# Towards Hybrid Earth System Modeling

This page reviews and organizes emerging hybrid Earth System Models (ESMs), which combine Machine Learning (ML) and physics-based components, alphabetically. Hybrid ESMs retain essential components for physical consistency (e.g., the dynamical core) while using ML to enhance parameterizations for small-scale processes (e.g., clouds). These models hold promise for improving long-term projections of Earth's physical climate and biogeochemical cycles.

If you notice any errors, omissions, or outdated information, please feel free to submit a pull request.

<ins>Author</ins>: Tom Beucler (UNIL); written in the context of [AI4PEX](https://ai4pex.org/).

## Table of Contents
- [CBRAIN](#cbrain)  
- [CliMA](#clima)
- [ClimSim](#climsim)
- [ICON-ML](#icon-ml)
- [Hybrid FV3-GFS](#hybrid-fv3-gfs)
- [Hybrid SAM](#hybrid-sam)
- [NCAM](#ncam)
- [NeuralGCM](#neuralgcm)

***

## CBRAIN

Cloud Brain (CBRAIN) aims to break the convective parameterization deadlock by training neural networks to emulate the total subgrid thermodynamics time-tendencies, i.e., the cumulative tendency of prognostic thermodynamic variables (temperature and specific humidity) produced by subgrid-scale convection, radiation, and turbulence. 

### Latest coupled simulations in [Lin, J., Yu, S., Peng, L., Beucler, T., Wong-Toi, E., Hu, Z., ... & Pritchard, M. S. (2024). Sampling Hybrid Climate Simulation at Scale to Reliably Improve Machine Learning Parameterization. arXiv preprint 2309.16177.](https://arxiv.org/abs/2309.16177)

### See also:
- [Rasp, S., Pritchard, M. S., & Gentine, P. (2018). Deep learning to represent subgrid processes in climate models. Proceedings of the national academy of sciences, 115(39), 9684-9689.](https://www.pnas.org/doi/full/10.1073/pnas.1810286115)  
- [Iglesias‐Suarez, F., Gentine, P., Solino‐Fernandez, B., Beucler, T., Pritchard, M., Runge, J., & Eyring, V. (2024). Causally‐informed deep learning to improve climate models and projections. Journal of Geophysical Research: Atmospheres, 129(4), e2023JD039202.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023JD039202)
- [Behrens, G., Beucler, T., Iglesias-Suarez, F., Yu, S., Gentine, P., Pritchard, M., ... & Eyring, V. (2024). Improving Atmospheric Processes in Earth System Models with Deep Learning Ensembles and Stochastic Parameterizations. arXiv preprint 2402.03079.](https://arxiv.org/abs/2402.03079)
- [Ott, J., Pritchard, M., Best, N., Linstead, E., Curcic, M., & Baldi, P. (2020). A fortran‐keras deep learning bridge for scientific computing. Scientific Programming, 2020(1), 8888811.](https://onlinelibrary.wiley.com/doi/full/10.1155/2020/8888811)
- [Beucler, T., Gentine, P., Yuval, J., Gupta, A., Peng, L., Lin, J., ... & Pritchard, M. (2024). Climate-invariant machine learning. Science Advances, 10(6), eadj7250.](https://www.science.org/doi/10.1126/sciadv.adj7250)
- [Mooers, G., Pritchard, M., Beucler, T., Ott, J., Yacalis, G., Baldi, P., & Gentine, P. (2021). Assessing the potential of deep learning for emulating cloud superparameterization in climate models with real‐geography boundary conditions. Journal of Advances in Modeling Earth Systems, 13(5), e2020MS002385.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002385)
- [Gentine, P., Pritchard, M., Rasp, S., Reinaudi, G., & Yacalis, G. (2018). Could machine learning break the convection parameterization deadlock?. Geophysical Research Letters, 45(11), 5742-5751.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018GL078202)

***

## [CliMA](https://clima.caltech.edu/)

### Latest coupled simulations in [Christopoulos, C., Lopez-Gomez, I., Beucler, T., Cohen, Y., Kawczynski, C., Dunbar, O., & Schneider, T. (2024). Online Learning of Entrainment Closures in a Hybrid Machine Learning Parameterization. Authorea Preprints.](https://essopenarchive.org/doi/full/10.22541/essoar.171804905.55213571/)

### See also:
- [Lopez‐Gomez, I., Christopoulos, C., Langeland Ervik, H. L., Dunbar, O. R., Cohen, Y., & Schneider, T. (2022). Training physics‐based machine‐learning parameterizations with gradient‐free ensemble Kalman methods. Journal of Advances in Modeling Earth Systems, 14(8), e2022MS003105.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022MS003105)
- [Lima, M., Deck, K., Dunbar, O. R., & Schneider, T. (2024). Toward Routing River Water in Land Surface Models with Recurrent Neural Networks. arXiv preprint 2404.14212.](https://arxiv.org/abs/2404.14212)
- [Charbonneau, A., Deck, K., & Schneider, T. (2023). A physics-constrained neural differential equation for data-driven seasonal snowpack forecasting. Artificial Intelligence for the Earth Systems, in review.](https://climate-dynamics.org/publications/2732/)
- [Schneider, T., Lan, S., Stuart, A., & Teixeira, J. (2017). Earth system modeling 2.0: A blueprint for models that learn from observations and targeted high‐resolution simulations. Geophysical Research Letters, 44(24), 12-396.](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2017GL076101)

***

## [ClimSim](https://leap-stc.github.io/ClimSim/README.html)

### Latest coupled simulations in [Hu, Z., Subramaniam, A., Kuang, Z., Lin, J., Yu, S., Hannah, W. M., ... & Pritchard, M. S. (2024). Stable Machine-Learning Parameterization of Subgrid Processes with Real Geography and Full-physics Emulation. arXiv preprint 2407.00124.](https://arxiv.org/abs/2407.00124)

### See also:
- [Yu, S., Hu, Z., Subramaniam, A., Hannah, W., Peng, L., Lin, J., Bhouri, M. A., Gupta, R., Lütjens, B., Will, J. C., Behrens, G., Busecke, J. J. M., Loose, N., Stern, C. I., Beucler, T., Harrop, B., Heuer, H., Hillman, B. R., Jenney, A., ... Pritchard, M. (2024). ClimSim-Online: A large multi-scale dataset and framework for hybrid ML-physics climate emulation. arXiv preprint 2306.08754](https://arxiv.org/abs/2306.08754)
- [Yu, S., Hannah, W., Peng, L., Lin, J., Bhouri, M. A., Gupta, R., ... & Pritchard, M. (2024). ClimSim: A large multi-scale dataset for hybrid physics-ML climate emulation. Advances in Neural Information Processing Systems, 36.](https://neurips.cc/virtual/2023/poster/73569)

***

## ICON-ML

### Latest coupled simulations in [Heuer, H., Schwabe, M., Gentine, P., Giorgetta, M. A., & Eyring, V. (2023). Interpretable multiscale Machine Learning-Based Parameterizations of Convection for ICON. arXiv preprint 2311.03251.](https://arxiv.org/abs/2311.03251)

### See also:
- [Grundner, A., Beucler, T., Gentine, P., & Eyring, V. (2024). Data‐driven equation discovery of a cloud cover parameterization. Journal of Advances in Modeling Earth Systems, 16(3), e2023MS003763.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS003763)
- [Grundner, A., Beucler, T., Gentine, P., Iglesias‐Suarez, F., Giorgetta, M. A., & Eyring, V. (2022). Deep learning based cloud cover parameterization for ICON. Journal of Advances in Modeling Earth Systems, 14(12), e2021MS002959.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002959)

***

## Hybrid FV3-GFS

FV3-GFS is the atmospheric component of the [Unified Forecast System (UFS)](https://zenodo.org/records/4460292). 

### Latest coupled simulations in [Watt‐Meyer, O., Brenowitz, N. D., Clark, S. K., Henn, B., Kwa, A., McGibbon, J., ... & Bretherton, C. S. (2024). Neural network parameterization of subgrid‐scale physics from a realistic geography global storm‐resolving simulation. Journal of Advances in Modeling Earth Systems, 16(2), e2023MS003668.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003668)

### See also:  
- [Kwa, A., Clark, S. K., Henn, B., Brenowitz, N. D., McGibbon, J., Watt‐Meyer, O., ... & Bretherton, C. S. (2023). Machine‐learned climate model corrections from a global storm‐resolving model: Performance across the annual cycle. Journal of Advances in Modeling Earth Systems, 15(5), e2022MS003400.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003400)
- [Clark, S. K., Brenowitz, N. D., Henn, B., Kwa, A., McGibbon, J., Perkins, W. A., ... & Harris, L. M. (2022). Correcting a 200 km resolution climate model in multiple climates by machine learning from 25 km resolution simulations. Journal of Advances in Modeling Earth Systems, 14(9), e2022MS003219.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003219)
- [Bretherton, C. S., Henn, B., Kwa, A., Brenowitz, N. D., Watt‐Meyer, O., McGibbon, J., ... & Harris, L. (2022). Correcting coarse‐grid weather and climate models by machine learning from global storm‐resolving simulations. Journal of Advances in Modeling Earth Systems, 14(2), e2021MS002794.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002794)
- [Watt‐Meyer, O., Brenowitz, N. D., Clark, S. K., Henn, B., Kwa, A., McGibbon, J., ... & Bretherton, C. S. (2021). Correcting weather and climate models by machine learning nudged historical simulations. Geophysical Research Letters, 48(15), e2021GL092555.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL092555)
- [Brenowitz, N. D., Henn, B., McGibbon, J., Clark, S. K., Kwa, A., Perkins, W. A., ... & Bretherton, C. S. (2020). Machine learning climate model dynamics: Offline versus online performance. NeurIPS 2020 CCAI workshop.](https://arxiv.org/abs/2011.03081)
- [Brenowitz, N. D., & Bretherton, C. S. (2019). Spatially extended tests of a neural network parametrization trained by coarse‐graining. Journal of Advances in Modeling Earth Systems, 11(8), 2728-2744.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019MS001711)
- [Brenowitz, N. D., & Bretherton, C. S. (2018). Prognostic validation of a neural network unified physics parameterization. Geophysical Research Letters, 45(12), 6289-6298.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018GL078510)

***

## Hybrid SAM

### Latest coupled simulations in [Yuval, J., O'Gorman, P. A., & Hill, C. N. (2021). Use of neural networks for stable, accurate and physically consistent parameterization of subgrid atmospheric processes with good performance at reduced precision. Geophysical Research Letters, 48(6), e2020GL091363.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL091363)

### See also:
- [Yuval, J., & O’Gorman, P. A. (2023). Neural‐network parameterization of subgrid momentum transport in the atmosphere. Journal of Advances in Modeling Earth Systems, 15(4), e2023MS003606.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003606)
- [Wang, P., Yuval, J., & O’Gorman, P. A. (2022). Non‐local parameterization of atmospheric subgrid processes with neural networks. Journal of Advances in Modeling Earth Systems, 14(10), e2022MS002984.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS002984)
- [Yuval, J., & O’Gorman, P. A. (2020). Stable machine-learning parameterization of subgrid processes for climate modeling at a range of resolutions. Nature communications, 11(1), 3295.](https://www.nature.com/articles/s41467-020-17142-3)

***

## NCAM

### Latest coupled simulations in [Han, Y., Zhang, G. J., & Wang, Y. (2023). An ensemble of neural networks for moist physics processes, its generalizability and stable integration. Journal of Advances in Modeling Earth Systems, 15(10), e2022MS003508.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003508)

### See also:   
- [Wang, X., Han, Y., Xue, W., Yang, G., & Zhang, G. J. (2022). Stable climate simulations using a realistic general circulation model with neural network parameterizations for atmospheric moist physics and radiation processes. Geoscientific Model Development, 15(9), 3923-3940.](https://gmd.copernicus.org/articles/15/3923/2022/)
- [Han, Y., Zhang, G. J., Huang, X., & Wang, Y. (2020). A moist physics parameterization based on deep learning. Journal of Advances in Modeling Earth Systems, 12(9), e2020MS002076.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020MS002076) 

***

## [NeuralGCM](https://neuralgcm.readthedocs.io/en/latest/)

### Latest coupled simulations in [Kochkov, D., Yuval, J., Langmore, I., Norgaard, P., Smith, J., Mooers, G., ... & Hoyer, S. (2024). Neural general circulation models for weather and climate. Nature, 1-7.](https://www.nature.com/articles/s41586-024-07744-y)

