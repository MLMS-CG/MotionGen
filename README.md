# Shape-conditioned human motion generation
A shape conditioned human motion generation model which is able to generate 3d motion with diffusion model.

# Walk motion generation conditioned with 3 different body shapes
<video src="https://github.com/KebingXUE/MotionGen/assets/47482603/b8d96176-42db-4832-bdff-46a547737f1b" width="20"></video>

# The folders
This project contains a complete set of codes associated to our SMD (Shape-conditioned Motion Diffusion) model for human motion generation.
It contains:
- **dataExploration**: Contains tools to explore a public human motion dataset, Babel. You can visualize the distribution of shape identities and the labeled actions in the form of a matrix-like image.
- **preProcessing**: Contains codes to import dataset files and preprocess them: from the original dataset containing SMPL parameters to mesh, and then to spectral coefficients, and, optionally, the skeleton (preProcessing_beta.py).
- **data_loaders**: Contains codes for loading data files (~kebing/Work/MotionGen/MotionGen/data/datasets/[experiment_name]/*.bin and *.npy) for training. *.pt files are not actually used. 
- **diffusion**: Contains the diffusion process, the training, inference, etc. The 'training_losses function' at gaussian_diffusion.py takes the input ('x_input'), the time step 't', and the 'model'. After generating the denoised version of the data ('model_output'), it compares it with the given data and computes the error.
- **model**: Contains our SMD (baseline.py), shape encoder (encodec.py: computes T-posed mesh from an arbitrarily posed mesh), and the action classifier (classifier.py).
- **targetMeshes**: contains target meshes we used for the video demo of EG2025 paper.
- **train**: Contains the training process either for the baseline or the classifier.
- **utils**: Contains model configurations (**parser_util.py**).

# The files
- **beta_all_\* files** and **beta_\* files**: precomputed distribution (mean and variance) of the training dataset, which are used for the normalization later.
- **beta_all_\* files** and **beta_\* files**
- **beta_physique_test.py**, and **physique_test.py** contains codes to generate motions with conditions.
- **shape_consistency.py**, **shapeConsis_colormap.py** contain codes to measure the shape consistency and to visualize the consistency error.
- **test.py** contains codes to post-process the generated results in a format that is compatible with some evaluation code written by others. The evaluation code can be found in the 'MDM (Motion Diffusion Model) project', under 'eval/eval_humanml.py'.

# Create dataset
Call 'python createDataset.py' to generate dataset for training 

# training
Run the script 'run.sh', after checking hyperparameters inside. 

# Test (generation)
Run 'python beta_physique_test.py' for skeleton based generation. The parameters or conditioning signals are hard-coded:
- Conditioning body shape is 'targetBeta' (beta_physique_test.py) or 'target' (physique_test.py).
- Conditioning action label is with 'exps'.
- Conditioning text prompt is in '??'.  <== Kebing will complete it, after modifying the code.
  
Run 'python physique_test.py' for mesh based generation

# Test (evaluation)
Using code in [MDM](https://github.com/GuyTevet/motion-diffusion-model) to evaluate 





