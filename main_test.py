import numpy as np
from src.train_visemenet import test
from src.create_dataset_csv import create_dataset_csv
from src.utl.load_param import *
from src.eval_viseme import eval_viseme

test_audio_name = 'visemenet_intro.wav'


# convert audio wav to network input format
create_dataset_csv(csv_dir, test_audio_name=test_audio_name)

# feedforward testing
test(model_name='pretrain_biwi', test_audio_name=test_audio_name[:-4])
print('Finish forward testing.')

# output viseme parameter
eval_viseme(test_audio_name[:-4])
print('Done.')