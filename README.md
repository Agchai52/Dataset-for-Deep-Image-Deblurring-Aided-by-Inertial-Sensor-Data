# Dataset for Deep Image Deblurring Aided by Inertial Sensor Data
The inertial sensor aided scheme is one of the most well known branches of traditional image deblurring algorithms. Compared to other image deblurring schemes, however, this scheme is less successful in the deep learning field. It's partly because of lacking a proper dataset. To address this blank, in this paper, a synthetic training and testing dataset for inertial sensors aided deep image deblurring neural network is proposed. This dataset includes groundtruth and blurry image pairs as well as inertial sensor data recorded during the exposure time of each blurry image. To simulate the real situations, the proposed dataset also considers error effects, including misalignment, rotation center shift, rolling shutter effect as well as inertial sensor data noise and image noise. 

The proposed dataset is called **DeblurIMUDataset** and is available online. [here](https://drive.google.com/file/d/18_PcNpadgxPOSaSpsUcFiTHpxNDmMtO3/view?usp=sharing)
## Required Libraries
- Python==2.7
- matplotlib==2.2.2
- numpy==1.14.3
- opencv-python==3.4.0.12
 
 Or just run: `pip install -r requirements.txt`

## How to Generate Single Set of Data: 
1. Put an image in the folder "/InputImages".
2. In "main.py", set `phase = single` (default).
3. If you want to plot sensor data, just set `isPlot = True` in "main.py", otherwise `isPlot = False`.
4. Run: `python main.py`.
5. A single set of synthetic data will be saved in the folder "/Output".
6. The output includes a set of reference/original blurry/error blurry frames, original/error inertial sensor data and 
all parameters.

## How to Generate Train/Test Dataset: 
1. Put "generateDatasetIMU.py" and "SynIMU2Blurry.py" into "GOPRO_Large_all/". "GOPRO_Large_all" can be downloaded from 
[this link](https://github.com/SeungjunNah/DeepDeblur_release) 
2. Generate train datatset and save them in "Dataset/train", run:
    ```buildoutcfg
    python generateDatasetIMU.py --phase train
    ```
3. Generate test datatset and save them in "Dataset/test", run:
    ```buildoutcfg
    python generateDatasetIMU.py --phase test
    ```
 ## Output Dataset
 In the phase folder, like "Dataset/train/"
 - "ImageName_ref.png": the groundtruth sharp image
 - "ImageName_blur_ori.png": the blurry image **without** error effects
 - "ImageName_blur_err.png": the blurry image **with error** effects
 - "ImageName_IMU_ori.txt": the gyro and acc data **without error** effects
 - "ImageName_IMU_err.txt": the gyro and acc data **with error** effects
 - "ImageName_param.txt": the parameters used in original data and error data