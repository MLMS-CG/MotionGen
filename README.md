# Shape-conditioned human motion generation
A shape conditioned human motion generation model which is able to generate 3d motion with diffusion model.

# Walk motion generation conditioned with 3 different body shapes
<video src="https://github.com/KebingXUE/MotionGen/assets/47482603/b8d96176-42db-4832-bdff-46a547737f1b" width="20"></video>

# The folders
This project contains a complete set of codes associated to our SMD (Shape-conditioned Motion Diffusion) model for human motion generation.
It contains:
- '''dataExploration''': Contains tools to explore a public human motion dataset, Babel. You can visualize the distribution of shape identities and the labeled actions in the form of a matrix-like image.
- preProcessing: Contains codes to import dataset files and preprocess them: from the original dataset containing SMPL parameters to mesh, and then to spectral coefficients, and, optionally, the skeleton (preProcessing_beta.py).
- data_loaders: Contains codes for loading data files (~kebing/Work/MotionGen/MotionGen/data/datasets/[experiment_name]/*.bin and *.npy) for training. *.pt files are not actually used. 
- diffusion: Contains the diffusion process, the training, inference, etc. The 'training_losses function' at gaussian_diffusion.py takes the input ('x_input'), the time step 't', and the 'model'. After generating the denoised version of the data ('model_output'), it compares it with the given data and computes the error.
- 
# Create dataset
Run 'python createDataset.py' to generate dataset for training 

# training
Using the script 'run.sh'

# Test
Run 'python beta_physique_test.py' for skeleton based generation
Run 'python physique_test.py' for mesh based generation
Using code in [MDM](https://github.com/GuyTevet/motion-diffusion-model) to evaluate 





