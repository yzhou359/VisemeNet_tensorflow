# VisemeNet Code Readme

## Environment

+ Python 3.5 
+ Tensorflow 1.1.0 
+ Cudnn 5.0

## Python Package

+ numpy
+ scipy
+ python_speech_features
+ matplotlib

## At test time:

1). **Create and install required envs and packages**
  * conda create -n visnet python=3.5
  * pip install --ignore-installed --upgrade https://download.tensorflow.google.cn/linux/gpu/tensorflow_gpu-1.1.0-cp35-cp35m-linux_x86_64.whl

2). **Download this repository to your local machine:**  
   * git clone https://github.com/yzhou359/VisemeNet_tensorflow.git  
   * cd VisemeNet_tensorflow 

3). **Prepare data and model:**  
   * convert your test audio files into WAV format, put it to the directory data/test_audio/   
   * download the latest model from [Here](https://www.dropbox.com/sh/7nbqgwv0zz8pbk9/AAAghy76GVYDLqPKdANcyDuba?dl=0), put all 4 files to data/ckpt/pretrain_biwi/  

4). **Forward inference:**  
   * put your test audio file name in file 'main_test.py', line 7. 
   * Then run command line

   > python main_test.py
   
   &nbsp; &nbsp; The result locates at:  
   &nbsp; &nbsp; &nbsp; &nbsp; data/output_viseme/[your_audio_file_name]/mayaparam_viseme.txt

5). **JALI animation in Maya:**
   * put your test audio file name in file 'maya_animation.py', line 4.
   * Then run 'maya_animation.py' in Maya with JALI environment to create talking face animation automatically.

