# Soft Contrastive Learning for Visual Localization
This repository contains the code for our NeurIPS 2020 publication [*Soft Contrastive Learning for Visual Localization*](https://www.dropbox.com/s/0pi6kdzpyj6152n/localization_Neurips20_preprint.pdf?dl=0).

The corresponding models, training/testing image lists and a movie with visual results can be downloaded [here](https://www.dropbox.com/sh/kvwqajyl49ax290/AABuDQ7NETKw7OF37jsrJXVxa?dl=0).

This code was tested using TensorFlow 1.10.0 and Python 3.5.6.

It uses the following git repositories as dependencies:

- [netvlad_tf_open](https://github.com/uzh-rpg/netvlad_tf_open)
- [pointnetvlad](https://github.com/mikacuy/pointnetvlad)
- [robotcar-dataset-sdk](https://github.com/ori-mrg/robotcar-dataset-sdk)

The training data can be downloaded using: 

- [RobotCarDataset-Scraper](https://github.com/mttgdd/RobotCarDataset-Scraper)


#### [Models](https://www.dropbox.com/sh/kvwqajyl49ax290/AABuDQ7NETKw7OF37jsrJXVxa?dl=0) used in the paper
| Name | Model |
|-|-|
| Off-the-shelf | offtheshelf |
| Triplet trained on Pittsburgh | pittsnetvlad |
| Triplet | triplet_xy_000 |
| Quadruplet | quadruplet_xy_000 |
| Lazy triplet | ha0_lolazy_triplet_muTrue_renone_vl64_pca_eccv_002 |
| Lazy quadruplet | ha0_lolazy_quadruplet_muTrue_renone_vl64_pca_eccv_002 |
| Trip.~+ Huber dist.  | huber_distance_triplet_xy_000 |
| Log-ratio | ha0_lologratio_ma15_mi15_muTrue_renone_tu1_vl64_pca_eccv_002 |
| Multi-similarity | ha0_loms_loss_msTrue_muTrue_renone_tu1_vl64_pca_eccv_001 |
| Ours | al0.8_be15_ha0_lowms_ma15_mi15_msTrue_muTrue_renone_tu1_vl64_pca_eccv_000 |
