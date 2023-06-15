- What did you do?
- Where did you download the code (including links)? Did you change anything and (if so) what?
- Did you have problems running the code? If so, how did you fix them?
- Which tests did you perform and what are the results? Discuss the results and include images and figures.

# Overview

Paper link: https://arxiv.org/abs/2302.05543

# Installation & Setup
The code for ControlNet was downloaded from the official python implementation (https://github.com/lllyasviel/ControlNet) and placed into src/ControlNet.

The pretrained models were downloaded from https://huggingface.co/lllyasviel/ControlNet and put into the src/ControlNet/models folder. As the pretrained models are quite large, we did not include them in our submission.

A python virtual environment can be created using the requirements.txt file we added, with updated pytorch versions compared to the original implementation. For faster inference, and even lower memory consumption we also installed the xformers library (refer to https://github.com/lllyasviel/ControlNet/issues/3).

## Changes
To allow inference with graphics card having 8GB of RAM, we use the low memory mode, setting the save_memory variable under config.py to True.


To get the code running with the newer package versions, the import from the rank_zero_only from the pytorch_lightning module, had to be changed from 
```
    from pytorch_lightning.utilities.distributed import rank_zero_only
```
to
```
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
```
in two seperate files:
- src/ControlNet/cldm/logger.py
- src/ControlNet/ldm/models/diffusion/ddpm.py


# Tests
## Synthetic data generation
