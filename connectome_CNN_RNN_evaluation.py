# Connectome-based CNN-RNN neurIPS
# 2021.03.16    Angel Canelo

###### import ######################
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import scipy.stats as measures
from pymatreader import read_mat
####################################
### Functions and Initializations ##
tf.config.experimental.list_physical_devices('GPU')
def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx
####################################
###### Dataset preparation #########
# Load DAVIS test data
test_set = ['rollerblade', 'scooter-black','scooter-gray', 'soapbox', 'soccerball',
            'stroller', 'surf', 'swing', 'tennis', 'train']
data = read_mat('.\\data\\DAVIS_CNNRNN_data.mat')
print(data.keys())
pos_x = np.array([]); pos_y = np.array([]); pos_z = np.array([])
delta_x = np.array([]); delta_y = np.array([]); delta_z = np.array([]); fr_timed = []
check = 0
for i in range(len(data['training_data'])):
    if any(ele in data['training_data'][i]['label'] for ele in test_set)==True:
        if check==0:
            input_frames = data['training_data'][i]['images']
            check += 1
        else:
            input_frames = np.concatenate((input_frames,data['training_data'][i]['images']), axis=0)
        for j in range(data['training_data'][i]['images'].shape[0]-10):
            fr_timed.append(data['training_data'][i]['images'][j:j+10,:,:])
        pos_x = np.append(pos_x,[data['training_data'][i]['x'][0:-1-9]])
        pos_y = np.append(pos_y,[data['training_data'][i]['y'][0:-1-9]])
        pos_z = np.append(pos_z, [data['training_data'][i]['z'][0:-1-9]])
        delta_x = np.append(delta_x, [data['training_data'][i]['delta_x'][0:-1-9]])
        delta_y = np.append(delta_y, [data['training_data'][i]['delta_y'][0:-1-9]])
        delta_z = np.append(delta_z, [data['training_data'][i]['delta_z'][0:-1-9]])
timed_fr = np.array(fr_timed)
print('Frames with time dimension', timed_fr.shape)
print('size of frames', input_frames.shape,
      'size of x', pos_x.shape, 'size of y', pos_y.shape, 'size of delta_x',
      delta_x.shape, 'size of delta_y', delta_y.shape)
y_true = np.stack((pos_x, pos_y, pos_z), axis=1); print('Array of true outputs', y_true.shape)
####################################
###### Load model ##################
connectome_cnn = load_model('connectome_model_CNNRNN_v3')
print(connectome_cnn.summary())
#results = connectome_cnn.evaluate(input_frames, y_true, batch_size=10); print(results[0])
####################################
###### Predict DAVIS test data #####
pred_davis = connectome_cnn.predict(timed_fr); print('Shape of prediction', pred_davis.shape)
# PERFORMANCE RMSE and Pearson's r
RMSE_x = np.sqrt(np.mean((pos_x[300:-1] - pred_davis[300:-1,0])**2)); print('RMSE_x', RMSE_x)
RMSE_y = np.sqrt(np.mean((pos_y[300:-1] - pred_davis[300:-1,1])**2)); print('RMSE_y', RMSE_y)
RMSE_z = np.sqrt(np.mean((pos_z[300:-1] - pred_davis[300:-1,2])**2)); print('RMSE_z', RMSE_z)
x_pearson_corr = measures.pearsonr(pred_davis[300:-1,0], pos_x[300:-1])[0]; print('x_Pearson', x_pearson_corr)
y_pearson_corr = measures.pearsonr(pred_davis[300:-1,1], pos_y[300:-1])[0]; print('y_Pearson', y_pearson_corr)
z_pearson_corr = measures.pearsonr(pred_davis[300:-1,2], pos_z[300:-1])[0]; print('z_Pearson', z_pearson_corr)
# Plot prediction against ground truth
fig2, ax2 = plt.subplots(3); idd_l = np.array([0,0,0,1,1,1]); idd_r = np.array([0,1,2,0,1,2])
labels = ['x','y','z','delta_x','delta_y','delta_z']
labels2 = ['x','y','z']
fig2.suptitle('Test data prediction')
for i in range(pred_davis.shape[1]):
    ax2[i].plot(np.arange(0,293), y_true[300:-1,i], linewidth=1, color='black', alpha=0.7)
    ax2[i].plot(np.arange(0,293), pred_davis[300:-1,i], linewidth=1, color='blue')
    ax2[i].legend(['ground truth', 'prediction'], loc='upper right', frameon=False)
    ax2[i].set_title('{bar_dir}'.format(bar_dir=labels[i]))
    ax2[i].set_ylabel('Distance (a.u.)'); ax2[i].set_xlabel('Time (a.u.)')
    ax2[i].set_xlim(0, 300+1); ax2[i].set_xticks(np.arange(0,300+1,150))
ax2[0].set_ylim(0, 30+1); ax2[0].set_yticks(np.arange(0,30+1,10))
ax2[1].set_ylim(0, 15+1); ax2[1].set_yticks(np.arange(0,15+1,5))
ax2[2].set_ylim(0, 15+1); ax2[2].set_yticks(np.arange(0,15+1,5))
fig_b, ax_b = plt.subplots(); x_l = np.arange(len(labels2)); vec = [RMSE_x,RMSE_y,RMSE_z]
bb = ax_b.bar(x_l, vec); colo = ['y', 'g', 'b']
ax_b.set_xticks(x_l); ax_b.set_xticklabels(labels2); ax_b.set_ylabel('RMSE'); ax_b.set_ylim(0,5)
for index, value in enumerate(vec):
    ax_b.text(x=index, y=value, s = str("{:.2f}".format(value))); bb[index].set_color(colo[index])
fig_c, ax_c = plt.subplots(); vec2 = [x_pearson_corr,y_pearson_corr,z_pearson_corr]
bb2 = ax_c.bar(x_l, vec2)
ax_c.set_xticks(x_l); ax_c.set_xticklabels(labels2); ax_c.set_ylabel('Pearson r'); ax_c.set_ylim(0,1)
for index, value in enumerate(vec2):
    ax_c.text(x=index, y=value, s = str("{:.2f}".format(value))); bb2[index].set_color(colo[index])
plt.show()
####################################
###### Plot learned filters ########
show_filt = []; show_special = []; all_filters = []
layer_names = ['L1R', 'L2R', 'L3R', 'L5L1', 'L5L2',
               'Mi1L1', 'Mi1L5', 'Tm3L1', 'Tm3L5', 'Mi9L3', 'Mi4L5', 'C3L1',
               'Tm1L2', 'Tm2L2', 'Tm4L2', 'Tm9L3', 'Tm9Mi4',
               'T4aMi1', 'T4aTm3', 'T4aMi9', 'T4aMi4', 'T4aC3',
               'T4bMi1', 'T4bTm3', 'T4bMi9', 'T4bMi4', 'T4bC3',
               'T4cMi1', 'T4cTm3', 'T4cMi9', 'T4cMi4', 'T4cC3',
               'T4dMi1', 'T4dTm3', 'T4dMi9', 'T4dMi4', 'T4dC3',
               'T5aTm1', 'T5aTm2', 'T5aTm4', 'T5aTm9',
               'T5bTm1', 'T5bTm2', 'T5bTm4', 'T5bTm9',
               'T5cTm1', 'T5cTm2', 'T5cTm4', 'T5cTm9',
               'T5dTm1', 'T5dTm2', 'T5dTm4', 'T5dTm9',
               'LPLC2T4a', 'LPLC2T4b', 'LPLC2T4c', 'LPLC2T4d',
               'LPLC2T5a', 'LPLC2T5b', 'LPLC2T5c', 'LPLC2T5d']
for ele in layer_names:
    ind_layer = getLayerIndexByName(connectome_cnn, ele)
    filters = connectome_cnn.layers[ind_layer].get_weights()[0]
    all_filters.append(np.squeeze(filters, axis=(2,3)))
    if ele == 'T4aMi9' or ele == 'T4bMi9' or ele == 'T5aTm4' or ele == 'T5bTm4':
        show_special.append(filters)
        print(ele, filters.shape)
    else:
        show_filt.append(filters)
        print(ele, filters.shape)
show_kernel = np.array(show_filt); show_kernel = np.squeeze(show_kernel, axis=(3,4))
show_kernel_sp = np.array(show_special); show_kernel_sp = np.squeeze(show_kernel_sp, axis=(3,4))
print('All filters', len(all_filters))
print('3x3 filters', show_kernel.shape); print('5x5 filters', show_kernel_sp.shape)
#### Lamina
fig_lam, ax_lam = plt.subplots(2,3)
im_lam = []
im_lam.append(ax_lam[0,0].imshow(all_filters[0], cmap='RdYlBu', vmin=-1, vmax=2)); ax_lam[0,0].set_title('L1')
im_lam.append(ax_lam[0,1].imshow(all_filters[1], cmap='RdYlBu', vmin=-1, vmax=2)); ax_lam[0,1].set_title('L2')
im_lam.append(ax_lam[0,2].imshow(all_filters[2], cmap='RdYlBu', vmin=-1, vmax=2)); ax_lam[0,2].set_title('L3')
im_lam.append(ax_lam[1,0].imshow(all_filters[3], cmap='RdYlBu', vmin=-1, vmax=2)); ax_lam[1,0].set_title('L5L1')
im_lam.append(ax_lam[1,1].imshow(all_filters[4], cmap='RdYlBu', vmin=-1, vmax=2)); ax_lam[1,1].set_title('L5L2')
fig_lam.suptitle('LAMINA trained'); fig_lam.colorbar(im_lam[0], ax=ax_lam, label='a.u.')
fig_lam.delaxes(ax = ax_lam[1,2])
for i in range(ax_lam.shape[0]):
    for j in range(ax_lam.shape[1]):
        ax_lam[i,j].set_axis_off()
#### Outer medulla
fig_med, ax_med = plt.subplots(4,4)
im_med = []
# ON PATHWAY
im_med.append(ax_med[0,0].imshow(all_filters[5], cmap='YlGnBu', vmin=-1, vmax=2)); ax_med[0,0].set_title('Mi1L1')
im_med.append(ax_med[0,1].imshow(all_filters[6], cmap='YlGnBu', vmin=-1, vmax=2)); ax_med[0,1].set_title('Mi1L5')
im_med.append(ax_med[0,2].imshow(all_filters[7], cmap='YlGnBu', vmin=-1, vmax=2)); ax_med[0,2].set_title('Tm3L1')
im_med.append(ax_med[0,3].imshow(all_filters[8], cmap='YlGnBu', vmin=-1, vmax=2)); ax_med[0,3].set_title('Tm3L5')
im_med.append(ax_med[1,0].imshow(all_filters[9], cmap='YlGnBu', vmin=-1, vmax=2)); ax_med[1,0].set_title('Mi9L3')
im_med.append(ax_med[1,1].imshow(all_filters[10], cmap='YlGnBu', vmin=-1, vmax=2)); ax_med[1,1].set_title('Mi4L5')
im_med.append(ax_med[1,2].imshow(all_filters[11], cmap='YlGnBu', vmin=-1, vmax=2)); ax_med[1,2].set_title('C3L1')
# OFF PATHWAY
im_med.append(ax_med[2,0].imshow(all_filters[12], cmap='YlGnBu', vmin=-1, vmax=2)); ax_med[2,0].set_title('Tm1L2')
im_med.append(ax_med[2,1].imshow(all_filters[13], cmap='YlGnBu', vmin=-1, vmax=2)); ax_med[2,1].set_title('Tm2L2')
im_med.append(ax_med[2,2].imshow(all_filters[14], cmap='YlGnBu', vmin=-1, vmax=2)); ax_med[2,2].set_title('Tm4L2')
im_med.append(ax_med[3,0].imshow(all_filters[15], cmap='YlGnBu', vmin=-1, vmax=2)); ax_med[3,0].set_title('Tm9L3')
im_med.append(ax_med[3,1].imshow(all_filters[16], cmap='YlGnBu', vmin=-1, vmax=2)); ax_med[3,1].set_title('Tm9Mi4')
###
fig_med.suptitle('Outer MEDULLA trained'); fig_med.colorbar(im_med[0], ax=ax_med, label='a.u.')
#fig_med.delaxes(ax = ax_med[0,2])
for i in range(ax_med.shape[0]):
    for j in range(ax_med.shape[1]):
        ax_med[i,j].set_axis_off()
### Inner medulla (T4)
fig_lp, ax_lp = plt.subplots(4,5)
im_lp = []
im_lp.append(ax_lp[0,0].imshow(all_filters[17], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[0,0].set_title('T4aMi1')
im_lp.append(ax_lp[0,1].imshow(all_filters[18], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[0,1].set_title('T4aTm3')
im_lp.append(ax_lp[0,2].imshow(all_filters[19], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[0,2].set_title('T4aMi9 (5x5)')
im_lp.append(ax_lp[0,3].imshow(all_filters[20], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[0,3].set_title('T4aMi4')
im_lp.append(ax_lp[0,4].imshow(all_filters[21], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[0,4].set_title('T4aC3')
###
im_lp.append(ax_lp[1,0].imshow(all_filters[22], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[1,0].set_title('T4bMi1')
im_lp.append(ax_lp[1,1].imshow(all_filters[23], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[1,1].set_title('T4bTm3')
im_lp.append(ax_lp[1,2].imshow(all_filters[24], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[1,2].set_title('T4bMi9 (5x5)')
im_lp.append(ax_lp[1,3].imshow(all_filters[25], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[1,3].set_title('T4bMi4')
im_lp.append(ax_lp[1,4].imshow(all_filters[26], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[1,4].set_title('T4bC3')
###
im_lp.append(ax_lp[2,0].imshow(all_filters[27], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[2,0].set_title('T4cMi1')
im_lp.append(ax_lp[2,1].imshow(all_filters[28], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[2,1].set_title('T4cTm3')
im_lp.append(ax_lp[2,2].imshow(all_filters[29], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[2,2].set_title('T4cMi9')
im_lp.append(ax_lp[2,3].imshow(all_filters[30], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[2,3].set_title('T4cMi4')
im_lp.append(ax_lp[2,4].imshow(all_filters[31], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[2,4].set_title('T4cC3')
###
im_lp.append(ax_lp[3,0].imshow(all_filters[32], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[3,0].set_title('T4dMi1')
im_lp.append(ax_lp[3,1].imshow(all_filters[33], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[3,1].set_title('T4dTm3')
im_lp.append(ax_lp[3,2].imshow(all_filters[34], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[3,2].set_title('T4dMi9')
im_lp.append(ax_lp[3,3].imshow(all_filters[35], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[3,3].set_title('T4dMi4')
im_lp.append(ax_lp[3,4].imshow(all_filters[36], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp[3,4].set_title('T4dC3')
###
for i in range(ax_lp.shape[0]):
    for j in range(ax_lp.shape[1]):
        ax_lp[i,j].set_axis_off()
fig_lp.suptitle('Inner MEDULLA trained'); fig_lp.colorbar(im_lp[1], ax=ax_lp, label='a.u.')
### LOBULA
fig_lo, ax_lo = plt.subplots(4,4)
im_lo = []
im_lo.append(ax_lo[0,0].imshow(all_filters[37], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo[0,0].set_title('T5aTm1')
im_lo.append(ax_lo[0,1].imshow(all_filters[38], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo[0,1].set_title('T5aTm2')
im_lo.append(ax_lo[0,2].imshow(all_filters[39], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo[0,2].set_title('T5aTm4 (5x5)')
im_lo.append(ax_lo[0,3].imshow(all_filters[40], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo[0,3].set_title('T5aTm9')
###
im_lo.append(ax_lo[1,0].imshow(all_filters[41], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo[1,0].set_title('T5bTm1')
im_lo.append(ax_lo[1,1].imshow(all_filters[42], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo[1,1].set_title('T5bTm2')
im_lo.append(ax_lo[1,2].imshow(all_filters[43], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo[1,2].set_title('T5bTm4 (5x5)')
im_lo.append(ax_lo[1,3].imshow(all_filters[44], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo[1,3].set_title('T5bTm9')
###
im_lo.append(ax_lo[2,0].imshow(all_filters[45], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo[2,0].set_title('T5cTm1')
im_lo.append(ax_lo[2,1].imshow(all_filters[46], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo[2,1].set_title('T5cTm2')
im_lo.append(ax_lo[2,2].imshow(all_filters[47], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo[2,2].set_title('T5cTm4')
im_lo.append(ax_lo[2,3].imshow(all_filters[48], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo[2,3].set_title('T5cTm9')
###
im_lo.append(ax_lo[3,0].imshow(all_filters[49], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo[3,0].set_title('T5dTm1')
im_lo.append(ax_lo[3,1].imshow(all_filters[50], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo[3,1].set_title('T5dTm2')
im_lo.append(ax_lo[3,2].imshow(all_filters[51], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo[3,2].set_title('T5dTm4')
im_lo.append(ax_lo[3,3].imshow(all_filters[52], cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo[3,3].set_title('T5dTm9')
###
for i in range(ax_lo.shape[0]):
    for j in range(ax_lo.shape[1]):
        ax_lo[i,j].set_axis_off()
fig_lo.suptitle('LOBULA trained'); fig_lo.colorbar(im_lo[1], ax=ax_lo, label='a.u.')
### OPTIC GLOMERULI
fig_op, ax_op = plt.subplots(2,4)
im_op = []
im_op.append(ax_op[0,0].imshow(all_filters[53], cmap='RdYlBu', vmin=-0.5, vmax=0.5)); ax_op[0,0].set_title('LPLC2T4a')
im_op.append(ax_op[0,1].imshow(all_filters[54], cmap='RdYlBu', vmin=-0.5, vmax=0.5)); ax_op[0,1].set_title('LPLC2T4b')
im_op.append(ax_op[0,2].imshow(all_filters[55], cmap='RdYlBu', vmin=-0.5, vmax=0.5)); ax_op[0,2].set_title('LPLC2T4c')
im_op.append(ax_op[0,3].imshow(all_filters[56], cmap='RdYlBu', vmin=-0.5, vmax=0.5)); ax_op[0,3].set_title('LPLC2T4d')
###
im_op.append(ax_op[1,0].imshow(all_filters[57], cmap='RdYlBu', vmin=-0.5, vmax=0.5)); ax_op[1,0].set_title('LPLC2T5a')
im_op.append(ax_op[1,1].imshow(all_filters[58], cmap='RdYlBu', vmin=-0.5, vmax=0.5)); ax_op[1,1].set_title('LPLC2T5b')
im_op.append(ax_op[1,2].imshow(all_filters[59], cmap='RdYlBu', vmin=-0.5, vmax=0.5)); ax_op[1,2].set_title('LPLC2T5c')
im_op.append(ax_op[1,3].imshow(all_filters[60], cmap='RdYlBu', vmin=-0.5, vmax=0.5)); ax_op[1,3].set_title('LPLC2T5d')
###
for i in range(ax_op.shape[0]):
    for j in range(ax_op.shape[1]):
        ax_op[i,j].set_axis_off()
fig_op.suptitle('OPTIC GLOMERULI trained'); fig_op.colorbar(im_op[1], ax=ax_op, label='a.u.')
####################################
########INITIAL WEIGHTS#############
scale = 1/75
# LAMINA #
L1R = scale*np.array([[0,0, 0],[0, -35, 0],[0, 0, 0]])
L2R = scale*np.array([[0,0, 0],[0, -45, 0],[0, 0, 0]])
L3R = scale*np.array([[0,0, 0],[0, -10, 0],[0, 0, 0]])
L5L1 = scale*np.array([[0,0, 0],[0, 120, 0],[0, 0, 0]])
L5L2 = scale*np.array([[0,0, 0],[0, 60, 0],[0, 0, 0]])
# Outer MEDULLA #
Mi1L1 = scale*np.array([[0,0, 0],[0, 140, 0],[0, 0, 0]])            # excit
Mi1L5 = scale*np.array([[0,0, 0],[0, 50, 0],[0, 0, 0]])
Tm1L2 = scale*np.array([[0,0, 0],[0, 180, 0],[0, 0, 0]])
Tm2L2 = scale*np.array([[0,0, 0],[0, 160, 0],[0, 0, 0]])
Tm3L1 = scale*np.array([[50,50, 50],[50, 110, 50],[50, 50, 50]])    # excit
Tm3L5 = scale*np.array([[0,0, 0],[0, 35, 0],[0, 0, 0]])
Tm4L2 = scale*np.array([[0,0, 0],[0, 70, 0],[0, 0, 0]])
Tm9L3 = scale*np.array([[0,0, 0],[0, 26, 0],[0, 0, 0]])
Tm9Mi4 = scale*np.array([[0,0, 0],[0, -12, 0],[0, 0, 0]])
Mi9L3 = scale*np.array([[0,0, 0],[0, 60, 0],[0, 0, 0]])             # inhib
Mi4L5 = scale*np.array([[5,5, 5],[5, 20, 5],[5, 5, 5]])
C3L1 = scale*np.array([[0,0, 0],[0, 80, 0],[0, 0, 0]])
# Inner MEDULLA #
# T4b
T4bMi1 = scale*np.array([[0,8, 8],[0, 32, 8],[0, 8, 8]])      #1e-6 is microSiemens
T4bTm3 = scale*np.array([[0,0, 0],[0, 10, 0],[8, 0, 8]])
T4bMi9 = scale*np.array([[0,0,0,0,0],[0,-16,0,0,0],[-8,-16,0,0,0], [0,-16,0,0,0], [0,0,0,0,0]])
T4bMi4 = scale*np.array([[0,0, -8],[0, 0, -8],[0, 0, -8]])
T4bC3 = scale*np.array([[0,0, -6],[0, 0, -6],[0, 0, -6]])
# T4a
T4aMi1 = scale*np.array([[8,8, 0],[8, 32, 0],[8, 24, 0]])
T4aTm3 = scale*np.array([[8,0, 8],[0, 10, 0],[0, 0, 0]])
T4aMi9 = scale*np.array([[0,0,0,0,0],[0,0,0,-8,0],[0,0,0,-8,-4], [0,0,0,-6,0], [0,0,0,0,0]])
T4aMi4 = scale*np.array([[-4,0, 0],[-6, 0, 0],[-8, 0, 0]])
T4aC3 = scale*np.array([[-6,0, 0],[-6, 0, 0],[-6, 0, 0]])
# T4c
T4cMi1 = scale*np.array([[10,8, 16],[8, 32, 0],[6, 0, 0]])
T4cTm3 = scale*np.array([[0,8, 0],[0, 10, 0],[0, 8, 0]])
T4cMi9 = scale*np.array([[0,0, 0],[0, -6, 0],[-8, -6, 0]])
T4cMi4 = scale*np.array([[0,-6, 0],[0, 0, 0],[0, 0, 0]])
T4cC3 = scale*np.array([[0,-6, 0],[0, 0, 0],[0, 0, 0]])
# T4d
T4dMi1 = scale*np.array([[8,0, 0],[8, 32, 0],[8, 8, 10]])
T4dTm3 = scale*np.array([[0,8, 0],[0, 10, 0],[0, 8, 0]])
T4dMi9 = scale*np.array([[-8,-16, -8],[0, -6, 0],[0, 0, 0]])
T4dMi4 = scale*np.array([[0,0, 0],[0, 0, 0],[0, -8, 0]])
T4dC3 = scale*np.array([[0,0, 0],[0, 0, 0],[0, -6, 0]])
# LOBULA
T5aTm1 = scale*np.array([[8,8, 0],[8, 32, 0],[8, 24, 0]])
T5aTm2 = scale*np.array([[-4,0, 0],[-6, 0, 0],[-8, 0, 0]])
T5aTm4 = scale*np.array([[0,0,0,0,0],[0,0,0,-8,0],[0,0,0,-8,-4], [0,0,0,-6,0], [0,0,0,0,0]])
T5aTm9 = scale*np.array([[8,0, 8],[0, 10, 0],[0, 0, 0]])
T5bTm1 = scale*np.array([[0,8, 8],[0, 32, 8],[0, 8, 8]])
T5bTm2 = scale*np.array([[0,0, -8],[0, 0, -8],[0, 0, -8]])
T5bTm4 = scale*np.array([[0,0,0,0,0],[0,-16,0,0,0],[-8,-16,0,0,0], [0,-16,0,0,0], [0,0,0,0,0]])
T5bTm9 = scale*np.array([[0,0, 0],[0, 0, 0],[0, 0, 8]])
T5cTm1 = scale*np.array([[10,8, 16],[8, 32, 0],[6, 0, 0]])
T5cTm2 = scale*np.array([[0,-6, 0],[0, 0, 0],[0, 0, 0]])
T5cTm4 = scale*np.array([[0,0, 0],[0, -6, 0],[-8, -6, 0]])
T5cTm9 = scale*np.array([[0,8, 0],[0, 10, 0],[0, 8, 0]])
T5dTm1 = scale*np.array([[8,0, 0],[8, 32, 0],[8, 8, 10]])
T5dTm2 = scale*np.array([[0,0, 0],[0, 0, 0],[0, -8, 0]])
T5dTm4 = scale*np.array([[-8,-16, -8],[0, -6, 0],[0, 0, 0]])
T5dTm9 = scale*np.array([[0,8, 0],[0, 10, 0],[0, 8, 0]])
# OPTIC GLOMERULI
LPLC2T4a = scale*np.array([[0,0, 0],[0, 27, 0],[0, 0, 0]])
LPLC2T4b = scale*np.array([[0,0, 0],[0, 27, 0],[0, 0, 0]])
LPLC2T4c = scale*np.array([[0,0, 0],[0, 27, 0],[0, 0, 0]])
LPLC2T4d = scale*np.array([[0,0, 0],[0, 27, 0],[0, 0, 0]])
LPLC2T5a = scale*np.array([[0,0, 0],[0, 27, 0],[0, 0, 0]])
LPLC2T5b = scale*np.array([[0,0, 0],[0, 27, 0],[0, 0, 0]])
LPLC2T5c = scale*np.array([[0,0, 0],[0, 27, 0],[0, 0, 0]])
LPLC2T5d = scale*np.array([[0,0, 0],[0, 27, 0],[0, 0, 0]])
####################################
######PLOTTING INITIAL WEIGHTS######
# LAMINA
fig_lam_ini, ax_lam_ini = plt.subplots(2,3)
im_lam_ini = []
im_lam_ini.append(ax_lam_ini[0,0].imshow(L1R, cmap='RdYlBu', vmin=-1, vmax=2)); ax_lam_ini[0,0].set_title('L1')
im_lam_ini.append(ax_lam_ini[0,1].imshow(L2R, cmap='RdYlBu', vmin=-1, vmax=2)); ax_lam_ini[0,1].set_title('L2')
im_lam_ini.append(ax_lam_ini[0,2].imshow(L3R, cmap='RdYlBu', vmin=-1, vmax=2)); ax_lam_ini[0,2].set_title('L3')
im_lam_ini.append(ax_lam_ini[1,0].imshow(L5L1, cmap='RdYlBu', vmin=-1, vmax=2)); ax_lam_ini[1,0].set_title('L5L1')
im_lam_ini.append(ax_lam_ini[1,1].imshow(L5L2, cmap='RdYlBu', vmin=-1, vmax=2)); ax_lam_ini[1,1].set_title('L5L2')
fig_lam_ini.suptitle('LAMINA initial'); fig_lam_ini.colorbar(im_lam_ini[0], ax=ax_lam_ini, label='a.u.')
fig_lam_ini.delaxes(ax = ax_lam_ini[1,2])
for i in range(ax_lam_ini.shape[0]):
    for j in range(ax_lam_ini.shape[1]):
        ax_lam_ini[i,j].set_axis_off()
# Outer MEDULLA
fig_med_ini, ax_med_ini = plt.subplots(4,4)
im_med_ini = []
# ON PATHWAY
im_med_ini.append(ax_med_ini[0,0].imshow(Mi1L1, cmap='YlGnBu', vmin=-1, vmax=2)); ax_med_ini[0,0].set_title('Mi1L1')
im_med_ini.append(ax_med_ini[0,1].imshow(Mi1L5, cmap='YlGnBu', vmin=-1, vmax=2)); ax_med_ini[0,1].set_title('Mi1L5')
im_med_ini.append(ax_med_ini[0,2].imshow(Tm3L1, cmap='YlGnBu', vmin=-1, vmax=2)); ax_med_ini[0,2].set_title('Tm3L1')
im_med_ini.append(ax_med_ini[0,3].imshow(Tm3L5, cmap='YlGnBu', vmin=-1, vmax=2)); ax_med_ini[0,3].set_title('Tm3L5')
im_med_ini.append(ax_med_ini[1,0].imshow(Mi9L3, cmap='YlGnBu', vmin=-1, vmax=2)); ax_med_ini[1,0].set_title('Mi9L3')
im_med_ini.append(ax_med_ini[1,1].imshow(Mi4L5, cmap='YlGnBu', vmin=-1, vmax=2)); ax_med_ini[1,1].set_title('Mi4L5')
im_med_ini.append(ax_med_ini[1,2].imshow(C3L1, cmap='YlGnBu', vmin=-1, vmax=2)); ax_med_ini[1,2].set_title('C3L1')
# OFF PATHWAY
im_med_ini.append(ax_med_ini[2,0].imshow(Tm1L2, cmap='YlGnBu', vmin=-1, vmax=2)); ax_med_ini[2,0].set_title('Tm1L2')
im_med_ini.append(ax_med_ini[2,1].imshow(Tm2L2, cmap='YlGnBu', vmin=-1, vmax=2)); ax_med_ini[2,1].set_title('Tm2L2')
im_med_ini.append(ax_med_ini[2,2].imshow(Tm4L2, cmap='YlGnBu', vmin=-1, vmax=2)); ax_med_ini[2,2].set_title('Tm4L2')
im_med_ini.append(ax_med_ini[3,0].imshow(Tm9L3, cmap='YlGnBu', vmin=-1, vmax=2)); ax_med_ini[3,0].set_title('Tm9L3')
im_med_ini.append(ax_med_ini[3,1].imshow(Tm9Mi4, cmap='YlGnBu', vmin=-1, vmax=2)); ax_med_ini[3,1].set_title('Tm9Mi4')
fig_med_ini.suptitle('Outer MEDULLA initial'); fig_med_ini.colorbar(im_med_ini[0], ax=ax_med_ini, label='a.u.')
for i in range(ax_med_ini.shape[0]):
    for j in range(ax_med_ini.shape[1]):
        ax_med_ini[i,j].set_axis_off()
# Inner MEDULLA
fig_lp_ini, ax_lp_ini = plt.subplots(4,5)
im_lp_ini = []
im_lp_ini.append(ax_lp_ini[0,0].imshow(T4aMi1, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[0,0].set_title('T4aMi1')
im_lp_ini.append(ax_lp_ini[0,1].imshow(T4aTm3, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[0,1].set_title('T4aTm3')
im_lp_ini.append(ax_lp_ini[0,2].imshow(T4aMi9, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[0,2].set_title('T4aMi9 (5x5)')
im_lp_ini.append(ax_lp_ini[0,3].imshow(T4aMi4, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[0,3].set_title('T4aMi4')
im_lp_ini.append(ax_lp_ini[0,4].imshow(T4aC3, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[0,4].set_title('T4aC3')
###
im_lp_ini.append(ax_lp_ini[1,0].imshow(T4bMi1, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[1,0].set_title('T4bMi1')
im_lp_ini.append(ax_lp_ini[1,1].imshow(T4bTm3, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[1,1].set_title('T4bTm3')
im_lp_ini.append(ax_lp_ini[1,2].imshow(T4bMi9, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[1,2].set_title('T4bMi9 (5x5)')
im_lp_ini.append(ax_lp_ini[1,3].imshow(T4bMi4, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[1,3].set_title('T4bMi4')
im_lp_ini.append(ax_lp_ini[1,4].imshow(T4bC3, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[1,4].set_title('T4bC3')
###
im_lp_ini.append(ax_lp_ini[2,0].imshow(T4cMi1, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[2,0].set_title('T4cMi1')
im_lp_ini.append(ax_lp_ini[2,1].imshow(T4cTm3, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[2,1].set_title('T4cTm3')
im_lp_ini.append(ax_lp_ini[2,2].imshow(T4cMi9, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[2,2].set_title('T4cMi9')
im_lp_ini.append(ax_lp_ini[2,3].imshow(T4cMi4, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[2,3].set_title('T4cMi4')
im_lp_ini.append(ax_lp_ini[2,4].imshow(T4cC3, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[2,4].set_title('T4cC3')
###
im_lp_ini.append(ax_lp_ini[3,0].imshow(T4dMi1, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[3,0].set_title('T4dMi1')
im_lp_ini.append(ax_lp_ini[3,1].imshow(T4dTm3, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[3,1].set_title('T4dTm3')
im_lp_ini.append(ax_lp_ini[3,2].imshow(T4dMi9, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[3,2].set_title('T4dMi9')
im_lp_ini.append(ax_lp_ini[3,3].imshow(T4dMi4, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[3,3].set_title('T4dMi4')
im_lp_ini.append(ax_lp_ini[3,4].imshow(T4dC3, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lp_ini[3,4].set_title('T4dC3')
###
for i in range(ax_lp_ini.shape[0]):
    for j in range(ax_lp_ini.shape[1]):
        ax_lp_ini[i,j].set_axis_off()
fig_lp_ini.suptitle('Inner MEDULLA initial'); fig_lp_ini.colorbar(im_lp_ini[1], ax=ax_lp_ini, label='a.u.')
# LOBULA
fig_lo_ini, ax_lo_ini = plt.subplots(4,4)
im_lo_ini = []
im_lo_ini.append(ax_lo_ini[0,0].imshow(T5aTm1, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo_ini[0,0].set_title('T5aTm1')
im_lo_ini.append(ax_lo_ini[0,1].imshow(T5aTm2, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo_ini[0,1].set_title('T5aTm2')
im_lo_ini.append(ax_lo_ini[0,2].imshow(T5aTm4, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo_ini[0,2].set_title('T5aTm4 (5x5)')
im_lo_ini.append(ax_lo_ini[0,3].imshow(T5aTm9, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo_ini[0,3].set_title('T5aTm9')
###
im_lo_ini.append(ax_lo_ini[1,0].imshow(T5bTm1, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo_ini[1,0].set_title('T5bTm1')
im_lo_ini.append(ax_lo_ini[1,1].imshow(T5bTm2, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo_ini[1,1].set_title('T5bTm2')
im_lo_ini.append(ax_lo_ini[1,2].imshow(T5bTm4, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo_ini[1,2].set_title('T5bTm4 (5x5)')
im_lo_ini.append(ax_lo_ini[1,3].imshow(T5bTm9, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo_ini[1,3].set_title('T5bTm9')
###
im_lo_ini.append(ax_lo_ini[2,0].imshow(T5cTm1, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo_ini[2,0].set_title('T5cTm1')
im_lo_ini.append(ax_lo_ini[2,1].imshow(T5cTm2, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo_ini[2,1].set_title('T5cTm2')
im_lo_ini.append(ax_lo_ini[2,2].imshow(T5cTm4, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo_ini[2,2].set_title('T5cTm4')
im_lo_ini.append(ax_lo_ini[2,3].imshow(T5cTm9, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo_ini[2,3].set_title('T5cTm9')
###
im_lo_ini.append(ax_lo_ini[3,0].imshow(T5dTm1, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo_ini[3,0].set_title('T5dTm1')
im_lo_ini.append(ax_lo_ini[3,1].imshow(T5dTm2, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo_ini[3,1].set_title('T5dTm2')
im_lo_ini.append(ax_lo_ini[3,2].imshow(T5dTm4, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo_ini[3,2].set_title('T5dTm4')
im_lo_ini.append(ax_lo_ini[3,3].imshow(T5dTm9, cmap='jet', vmin=-0.5, vmax=0.5)); ax_lo_ini[3,3].set_title('T5dTm9')
###
for i in range(ax_lo_ini.shape[0]):
    for j in range(ax_lo_ini.shape[1]):
        ax_lo_ini[i,j].set_axis_off()
fig_lo_ini.suptitle('LOBULA initial'); fig_lo_ini.colorbar(im_lo_ini[1], ax=ax_lo_ini, label='a.u.')
# OPTIC GLOMERULI
fig_op_ini, ax_op_ini = plt.subplots(2,4)
im_op_ini = []
im_op_ini.append(ax_op_ini[0,0].imshow(LPLC2T4a, cmap='RdYlBu', vmin=-0.5, vmax=0.5)); ax_op_ini[0,0].set_title('LPLC2T4a')
im_op_ini.append(ax_op_ini[0,1].imshow(LPLC2T4b, cmap='RdYlBu', vmin=-0.5, vmax=0.5)); ax_op_ini[0,1].set_title('LPLC2T4b')
im_op_ini.append(ax_op_ini[0,2].imshow(LPLC2T4c, cmap='RdYlBu', vmin=-0.5, vmax=0.5)); ax_op_ini[0,2].set_title('LPLC2T4c')
im_op_ini.append(ax_op_ini[0,3].imshow(LPLC2T4d, cmap='RdYlBu', vmin=-0.5, vmax=0.5)); ax_op_ini[0,3].set_title('LPLC2T4d')
###
im_op_ini.append(ax_op_ini[1,0].imshow(LPLC2T5a, cmap='RdYlBu', vmin=-0.5, vmax=0.5)); ax_op_ini[1,0].set_title('LPLC2T5a')
im_op_ini.append(ax_op_ini[1,1].imshow(LPLC2T5b, cmap='RdYlBu', vmin=-0.5, vmax=0.5)); ax_op_ini[1,1].set_title('LPLC2T5b')
im_op_ini.append(ax_op_ini[1,2].imshow(LPLC2T5c, cmap='RdYlBu', vmin=-0.5, vmax=0.5)); ax_op_ini[1,2].set_title('LPLC2T5c')
im_op_ini.append(ax_op_ini[1,3].imshow(LPLC2T5d, cmap='RdYlBu', vmin=-0.5, vmax=0.5)); ax_op_ini[1,3].set_title('LPLC2T5d')
###
for i in range(ax_op_ini.shape[0]):
    for j in range(ax_op_ini.shape[1]):
        ax_op_ini[i,j].set_axis_off()
fig_op_ini.suptitle('OPTIC GLOMERULI initial'); fig_op_ini.colorbar(im_op_ini[1], ax=ax_op_ini, label='a.u.')
plt.show()
####################################