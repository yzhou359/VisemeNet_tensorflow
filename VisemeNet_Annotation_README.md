# VisemeNet Annotation README

## Dataset

BIWI dataset, 14 speakers (8 female, 6 male)

## Structure

There are 14 folders representing for 14 different speaker in BIWI dataset, such as 'F1', 'F2', ...

In each folder, 

 - file_dir.csv : an info file containing the frame range information in the annotation file for each video clip. 

```
# FORMAT 
video_name start_frame_index frame_lenght
```

 - maya_param_public_model.csv : the viseme annotation file for face rig http://www.dgp.toronto.edu/%7Eelf/jali.html]

 	+ Each line represents the viseme parameter values in each frame.
 	+ The viseme parameters are in this order
```
'JALI.translateX', 'JALI.translateY', 'AAA', 'Eh', 'AHH', 'OHH', 'UUU', 'IEE', 'RRR', 'WWW', 'SSS', 'FFF', 'TTH', 'MBP', 'SSH', 'Schwa', 'GK', 'LNTD', 'COARTIC.LNTD', 'COARTIC.GK', 'COARTIC.MMM', 'COARTIC.FFF', 'COARTIC.WA_PEDAL', 'COARTIC.YA_PEDAL'
```

 - maya_param.csv : the viseme annotation file for the face rig (not public available now) used in paper.

