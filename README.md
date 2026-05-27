# Geometry-based Discovery of Calcium Battery Cathodes Accelerated by Foundational Machine-Learned Models

This repository contains the data and workflow of screening pipeline that identifies promising Ca-ion battery cathode materials. The workflow combines structural data from the Materials Project (MP) with machine-learning interatomic potentials (MACE-MP-0, Orb-v3) and a transfer-learning (TL) migration-barrier model.

## Repository Structure
#### NEB_MACE_Orb-v3/
Nudged elastic band (NEB) calculation results for the 221 candidate frameworks, computed using the MACE and Orb-v3 machine-learned interatomic potentials. These results characterize the Ca-ion migration barriers used in the mobility analysis.

#### dft_neb_data/
DFT-NEB calculation results for the selected final candidate frameworks. These first-principles calculations validate the migration barriers obtained from the machine-learned models.

#### mp_queried_compounds/
The MP material_id of every structure queried from the MP in this project, providing a complete record of the initial search space.

#### scripts/
A Jupyter notebook implementing the entire screening pipeline for Ca-cathode discovery.
