import maya.cmds as cmds

# change your test audio file name here
test_audio_name = 'visemenet_intro'

# change to absolute root directory if necessary
f_pred = '/Users/yangzhou/Documents/git/yzhou359/VisemeNet_tensorflow/data/output_viseme/' + test_audio_name + '/mayaparam_viseme.txt'

params = ['Jaw', 'Lip', 'Ah', 'Aa', 'Eh', 'Ee', 'Ih', 'Oh', 'Uh', 'U', 'Eu', 'Schwa', 'R', 'S', 'Sh Ch Zh', 'Th', 'JY', 'LNTD', 'GK', 'MBP', 'FV', 'WA_PEDAL']

param_public_model = ['translateX', 'translateY', 'AAA', 'Eh', 'AHH', 'OHH', 'UUU', 'IEE', 'RRR', 'WWW', 'SSS', 'FFF', 'TTH', 'MBP', 'SSH', 'Schwa', 'GK', 'LNTD', 'COARTIC.LNTD', 'COARTIC.GK', 'COARTIC.MMM', 'COARTIC.FFF', 'COARTIC.WA_PEDAL', 'COARTIC.YA_PEDAL']

map_i = p_map = [0, 1, 3, 4, 2, [7, 8], 9, [5, 6], 12, 9, 13, 20, 15, 19, 14, 11, 18, 17, 17, 18, 19, 20, 21]


pred = open(f_pred, 'r')
y_pred = []
for line in pred:
    y_pred.append([float(f) for f in line.split()])
pred.close()

print(len(y_pred))

for sample_idx in range(0, len(y_pred), 4):
    print(sample_idx)
    cmds.currentTime(sample_idx/4)
    #cmds.currentTime(sample_idx)
    sample = y_pred[sample_idx]
    #print sample
    for i in range(len(sample)):
        if sample[i] < 0:
            sample[i] = 0
        elif sample[i] > 1:
            sample[i] = 1

    for i in range(len(map_i)):
        if i in [0, 1]:
            cmds.setAttr("CNT_JaLi."+param_public_model[i], sample[map_i[i]]*10)
            cmds.setKeyframe("CNT_JaLi."+param_public_model[i])

        elif i in range(2, 18):
            if(type(map_i[i]) == list):
                tmp = max(sample[map_i[i][0]], sample[map_i[i][1]])
            else:
                tmp = sample[map_i[i]]
            cmds.setAttr("CNT_PHONEMES."+param_public_model[i], tmp*10)
            cmds.setKeyframe("CNT_PHONEMES."+param_public_model[i])
        else:
            cmds.setAttr("CNT_"+param_public_model[i], sample[map_i[i]]*10)
            cmds.setKeyframe("CNT_"+param_public_model[i])
