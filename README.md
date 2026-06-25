# Thesis
### Data Augmentation Techniques for 3D generative models for post-radiotherapy MRI prediction. 
This work explores the extent to which data augmentation techniques, and more specifically domain randomization, can enhance the robustness of conditional 3D generative AI models for longitudinal post-treatment morphological prediction to variations in image quality. To evaluate our models, we measure performance using image quality, segmentation, and deformation metrics, comparing the baseline model with the domain randomization model across multiple levels of transformation in the pre-treatment image for three imaging modalities: T1, T1C, and T2-FLAIR.
### Methodology Overview
Two models are investigated: a baseline model and a domain randomization model. Each architecture consists of a diffusion model integrated with a ControlNet. While the same diffusion model was utilized for both, the experiments differed in the training of the ControlNet. 
At first, all modalities in the training set were compressed into latent embeddings separately using pre-trained MAISI autoencoder. The produced embeddings are concatenated along the channel dimension at the session level. During training of the diffusion model, noise is added to the concatenated embeddings at a randomly sampled timestep via the noise scheduler. The diffusion model is trained using an L1 loss function and the Adam optimizer, with the model learning to predict the denoised latent representation at each timestep. During ControlNet training, the follow-up images embeddings across all modalities for the same patient and session are concatenated along the channel dimension, as are the baseline image embeddings.Additionally, the raw baseline images across all modalities for the same patient and session are concatenated along the channel dimension. Noise is added to the concatenated follow-up embeddings at a randomly sampled timestep via the noise scheduler. The resulting noisy latents, along with the dose map, follow-up times, and concatenated baseline images, are passed as inputs to the ControlNet. The ControlNet processes this conditioning information and produces residual feature maps, which are subsequently passed to the pre-trained diffusion U-Net to steer its predictions toward the denoised follow-up latent representation. The model is trained using an L1 loss function and the Adam optimizer. The key distinction between the baseline and domain randomization ControlNet training lies in the training dataset. In the domain randomization model, the dataset is augmented with synthetically generated samples. For each patient in the dataset, an additional augmented sample is created in which all information is preserved except for the baseline images, which are subjected to a set of random transformations. Each transformation is applied independently with a probability of 0.5. The augmented sample is treated as an independent patient.

# **Workflow**
0. Set up environment
1. Prepare data and configs
2. Test autoencoder and create embeddings
3. Train unconditional diffusion model
4. Train Baseline Model ControlNet for conditional generation
5. Train Domain Randomisation Model ControlNet for conditional generation
6. Test Models
7. Result Analysis

## 0. Set up environment
### 0.1 Python Virtual Environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 0.2 WanDB
Create a wandb account, get an API key from https://wandb.ai/authorize and add it to your environment variables (if you don't want to add it every time you start a new terminal, add it to your .bashrc or .zshrc file):
export WANDB_API_KEY=<your_api_key>

### 0.3 Container Environment
Model evaluation requires a containerized environment bundling neuroimaging tools including FreeSurfer, FSL, and ANTs. The container is provided as a .sif file and executed via Apptainer. We also provide TumorSynth as a separate container.
Setup
Download TumorSynth from its official repository and follow the installation instructions provided there.
Install dependencies for both the neuroimaging container and the TumorSynth container as specified in their respective installation guides.
Build the containers using the provided build script, which automatically generates both .sif files.
Finally, set the environment variables:
#### Paths to container files
```export NEURO_CONTAINER_PATH=<neuro_suite_sif>
export TUMORSYNTH_CONTAINER_PATH=<tumorsynth_sif>
```
#### Bind paths: map host directories to container paths
```
export APPTAINER_BINDPATH="\
<model_restore_script>:/opt/miniconda/lib/python3.XX/site-packages/nnunet/training/model_restore.py,\
<data_dir>,\
<results_dir>,\
<output_dir>,\
<nnUNet_trained_models>:/opt/nnUNet/nnUNet_v1.7/nnUNet_trained_models,\
<nnUNet_trained_models>:/opt/nnUNet/results,\
<nnUNet_raw_data>:/opt/nnUNet/nnUNet_v1.7/nnUNet_raw_data_base,\
<nnUNet_preprocessed>:/opt/nnUNet/nnUNet_v1.7/nnUNet_preprocessed,\
<nnUNet_raw_data2>:/opt/nnUNet/raw"
```
#### Paths to bind into the container (required if data is on network filesystem)
`export APPTAINER_BINDPATH=/path/to/data,/path/to/output`

## 1. Prepare data and configs
### Dataset format and cofigs
Information about subjects, dataset metadata, and image quality are stored in two Excel files: source_excel (subject and metadata information) and quality_excel (image quality information).
Dataset should have the following structure:
```
project <--- data_base_dir
└── subject
└── session
└── anat
└── Modality Name
```
To set up the config files, run the following command (also see example usage in the module source code):

```
python -m data.configure_paths \
    --data_base_dir <data_base_dir> \
    --output_dir <output_dir> \
    --img_filename <img_filename> \
    --brain_mask_filename <brain_mask_filename> \
    --roi_mask_filename <roi_mask_filename> \
    --dose_filename <dose_filename>
```
Where `data_base_dir` is the path to the dataset (as shown above in the dataset structure), `output_dir` is the path to the output directory (for training and inference runs), `img_filename` is the name of the image file, `brain_mask_filename` is the name of the brain mask file, `roi_mask_filename` is the name of the region-of-interest (tumor mask) file, and `dose_filename` is the name of the dose file.
### Create Unconditional Model Datalists and Configure Image Dimensions/Stats
```
python -m data.parse_excel --data_base_dir<data_base_dir> --source_excel <soyrce_excel.xlsx> --quality_excel <quality_excel.csv>
python -m data.create_datalist --mode unconditional --output_filename <baseline_datalist_name.json>
python -m data.configure_dims_and_stats --method baseline
'''

## 2. Test autoencoder and create embeddings
To test the autoencoder on a number of samples run:
```
python -m data.create_embeddings --num_samples n --run_name <baseline_embedings_name>  --method baseline
```
To create all the latent embeddings that the diffusion model will be trained on, run:
```
python -m data.create_embeddings --run_name <baseline_embedings_name>  --method baseline
```
## 3. Train unconditional diffusion model
Before training, ensure all parameters in the corresponding configuration script (`model_config_diff_model_train.json`) are set correctly. Then run:
'''
python -m train.diff_model_train --run_name <name>
```
## 4. Train Baseline Model ControlNet for conditional generation
Before training, ensure all parameters in the corresponding configuration script (`env_config_controlnet_train.json`) are set correctly. 
Set `trained_diff_model_path` to the checkpoint of the desired diffusion model epoch. 
Then run:
```
python -m train.controlnet_train --mode concat --method baseline --run_name <baseline_controlNet_name>
```
## 5. Train Domain Randomisation Model ControlNet for conditional generation
Create datalist and embeddings for domain randomisation model. 
```
python -m data.create_datalist --mode domainRand --output_filename <domainRand_datalist_name.json>

python -m data.create_embeddings --run_name <domainRand_embedings_name>  --method domainRand
```
Before training, ensure all parameters in the corresponding configuration script (`env_config_controlnet_train.json`) are set correctly. 
Set `trained_diff_model_path` to the checkpoint of the desired diffusion model epoch.
Then run:
```
python -m train.controlnet_train --mode concat --method domainRand --run_name <domain_randomisation_controlNet_name>
```
## 6. Test Models
Ensure `trained_diff_model_path` and `trained_controlnet_path` in `env_config_controlnet_infer.json` point to the correct trained diffusion model and ControlNet checkpoints, respectively. Both the baseline and domain randomisation models share the same `trained_diff_model_path`, but each requires a different `trained_controlnet_path` corresponding to their respective trained ControlNet model.
Then run:
```
python -m infer.controlnet_infer --mode concat --max_follow_ups n --num_subjects m --run_name <inference_name>
```
where n is the maximum number of follow-ups per subject and m is the maximum number of subjects in the test set used for inference.

## 7. Result Analysis
Produced results can be further analysed using the Results_Analysis Jupyter Notebook. The metrics.csv files generated per level during inference of each model are used for this analysis. Additionally, the produced Jacobian maps in .nii.gz format are used for plotting.

# Files
**configs:** Includes configuration files for the models. Here, users can set model parameters, data source paths, output directories, etc.\
**data:** Scripts for data handling, including reading data information, dataset splitting (train, validation, test), data loading, image statistics computation, embedding creation, and image transformations.\
**eval:** Scripts for evaluation, metric definitions, and the ControlNet evaluation pipeline.\
**infer:** Scripts for inference across all trained models, specifically  the diffusion model and ControlNet models (baseline model and domain randomisation model)\
**model:** Scripts for model definitions, including the autoencoder, diffusion model, and ControlNet.\
**plot:** Scripts for generating plots and figures.\
**train:** Scripts for training the diffusion model and ControlNet models (baseline model and domain randomisation model).\
**utils:** Helper functions.\
**Results_Analysis.ipynb:** A Jupyter notebook that processes results and generates figures and plots for the thesis document.\
**requirements:** Python packages required for installation.
