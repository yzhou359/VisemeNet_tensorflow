import maya.cmds as cmds

# change your test audio file name here
test_audio_name = 'visemenet_intro'

# change to absolute root directory if necessary
f_pred = 'data/output_viseme/' + test_audio_name + '/mayaparam_viseme.txt'

params = ['Jaw', 'Lip', 'Ah', 'Aa', 'Eh', 'Ee', 'Ih', 'Oh', 'Uh', 'U', 'Eu', 'Schwa', 'R', 'S', 'Sh Ch Zh', 'Th', 'JY', 'LNTD', 'GK', 'MBP', 'FV', 'WA_PEDAL']
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
        if i in [0,1]:
            cmds.setAttr("JaLi."+params[i], sample[i]*12)
            cmds.setKeyframe("JaLi."+params[i])
        elif i in range(2,19):
            cmds.setAttr("CNT_PHONEMES."+params[i], sample[i]*10)
            cmds.setKeyframe("CNT_PHONEMES."+params[i])
        elif i in range(19,22):
            cmds.setAttr("CNT_NOJAW."+params[i], sample[i]*10)
            cmds.setKeyframe("CNT_NOJAW."+params[i])