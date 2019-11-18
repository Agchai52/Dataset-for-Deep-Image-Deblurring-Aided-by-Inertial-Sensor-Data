# Dataset for Deep Image Deblurring Aided by Inertial Sensor Data

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
6. The output include a set of reference/original blurry/error blurry frames, original/error inertial sensor data and 
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