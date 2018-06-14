# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 09:40:32 2018

@author: Juan Sebastián Gómez
"""

import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.colorbar import colorbar
eps = np.finfo(np.float32).eps

def concatenate_files(path_to_files, list_inst):
    full_audio = list()
    for num, inst in enumerate(list_inst):
        for root, dirs, files in os.walk(path_to_files):
            for file in files:
                if file.startswith(inst):
                    full_file = os.path.join(path_to_files, file)
                    data, sr = librosa.load(full_file, sr=22050, mono=True)
                    data /= np.max(np.abs(data))
                    print(file)
                    full_audio.extend(data.tolist())
    return np.array(full_audio)

def create_spectrogram(audio):
    # normalize data
    audio /= np.max(np.abs(audio))
    audio = np.squeeze(audio)
    # short time fourier transform
    D = np.abs(librosa.stft(audio, win_length=1024, hop_length=512, center=True))
    # mel frequency representation
    S = librosa.feature.melspectrogram(S=D, sr=22050, n_mels=128)
    # natural logarithm
    ln_S = np.log(S + eps)
    # create tensor
    seg_dur = 43 # segment duration eq to 1 second
    spec_list = list()
    for idx in range(0, ln_S.shape[1] - seg_dur + 1, int(seg_dur * 0.5)):
        spec_list.append(ln_S[:, idx:(idx+seg_dur)])
    # print('Number of spectrograms:', len(spec_list))
    X = np.expand_dims(np.array(spec_list), axis=1)
    return ln_S, X

def plot_everything(list_inst, full_audio, ln_S, org_pred_tl, agg_pred_tl, org_pred_sc, agg_pred_sc):
    fontsize = 10
    # fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5,4), sharex=True)
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(10,7), sharex=False)
    ax[0].imshow(ln_S, origin='lower', aspect='auto', extent=[0,60,0,128])#, interpolation='nearest')
    ax[0].set_title('Melspectrogram', fontsize=fontsize+1)
    ax[0].set_ylabel('Mel-bands', fontsize=fontsize)
    im = ax[1].imshow(org_pred_tl, aspect='auto', extent=[0,60,6,0])
    # plot ground truth
    color = '#ffffff'
    center = 0
    for j in range(0, 6):
        ax[1].add_patch(patches.Rectangle(((j*10)-center, j-center), 10, 1, fill=False, edgecolor=color, linewidth=2))
    ax[1].set_title('Segment Predictions with Transfer Learning', fontsize=fontsize+1)
    ax[1].set_ylabel('Labels' , fontsize=fontsize)
    ax[1].set_yticks(np.arange(len(list_inst)))
    ax[1].set_yticklabels(list_inst, fontsize=fontsize)
    im = ax[2].imshow(agg_pred_tl, aspect='auto', extent=[0,60,6,0])
    # plot ground truth
    for j in range(0, 6):
        ax[2].add_patch(patches.Rectangle(((j*10)-center, j-center), 10, 1, fill=False, edgecolor=color, linewidth=2))
    ax[2].set_title('Aggregated Predictions with Transfer Learning', fontsize=fontsize+1)
    ax[2].set_ylabel('Labels', fontsize=fontsize)
    ax[2].set_yticks(np.arange(len(list_inst)))
    ax[2].set_yticklabels(list_inst, fontsize=fontsize)
    
    im = ax[3].imshow(org_pred_sc, aspect='auto', extent=[0,60,6,0])
    # plot ground truth
    color = '#ffffff'
    for j in range(0, 6):
        ax[3].add_patch(patches.Rectangle(((j*10)-center, j-center), 10, 1, fill=False, edgecolor=color, linewidth=2))
    ax[3].set_title('Segment Predictions without Transfer Learning', fontsize=fontsize+1)
    ax[3].set_ylabel('Labels' , fontsize=fontsize)
    ax[3].set_yticks(np.arange(len(list_inst)))
    ax[3].set_yticklabels(list_inst, fontsize=fontsize)
    im = ax[4].imshow(agg_pred_sc, aspect='auto', extent=[0,60,6,0])
    # plot ground truth
    for j in range(0, 6):
        ax[4].add_patch(patches.Rectangle(((j*10)-center, j-center), 10, 1, fill=False, edgecolor=color, linewidth=2))
    ax[4].set_title('Aggregated Predictions without Transfer Learning', fontsize=fontsize+1)
    ax[4].set_ylabel('Labels', fontsize=fontsize)
    ax[4].set_yticks(np.arange(len(list_inst)))
    ax[4].set_yticklabels(list_inst, fontsize=fontsize)
    ax[4].set_xlabel('Seconds', fontsize=fontsize)

    plt.tight_layout()
    fig.savefig('jazz_show_case.png', bbox_inches='tight')
    plt.show()

def load_model(filename, path_to_model, trans_learn):
    if trans_learn == True and filename.find('solo') > 0:
            json_filename = os.path.join(path_to_model, 'model_transfer_solo.json')
            weights_filename = os.path.join(path_to_model, 'model_transfer_solo.hdf5')
    elif trans_learn == False and filename.find('solo') > 0:
            json_filename = os.path.join(path_to_model, 'model_scratch_solo.json')
            weights_filename = os.path.join(path_to_model, 'model_scratch_solo.hdf5')

    with open(json_filename, 'r') as json_file:
        loaded_model = json_file.read()
    model = model_from_json(loaded_model)
    model.load_weights(weights_filename)
    print('Model', json_filename, ' loaded!')
    return model

def organize_predictions(list_inst, labels_inst, pred):
    org_pred = np.zeros(pred.shape)
    for num, inst in enumerate(list_inst):
        for key in labels_inst.keys():
            if inst == labels_inst[key]:
                org_pred[num, :] = pred[key, :]
    return org_pred

def aggregate_predictions(pred):
    agg_pred = np.zeros(pred.shape)
    chunk = 20
    for i in range(pred.shape[0]):
        agg_pred[:, i*chunk : (i+1)*chunk] = np.tile(np.sum(pred[:, i*chunk : (i+1)*chunk], axis=1) / chunk, (chunk, 1)).T
        agg_pred[:, i*chunk : (i+1)*chunk] /= np.max(agg_pred[:, i*chunk : (i+1)*chunk], axis=0)
    agg_pred = agg_pred[:, :120]
    return agg_pred
    

if __name__ == '__main__':
    path_to_audio = './audio/'
    path_to_models = './models/'
    mode = 1

    list_inst = ['as','ts','ss','tb','tp','cl']
    labels_inst = {0: 'as', 1: 'cl', 2: 'ss', 3: 'tb', 4: 'tp', 5: 'ts'}

    if mode == 0:
        # concatenate mix files
        full_audio = concatenate_files(os.path.join(path_to_audio, 'original/'), list_inst)
        librosa.output.write_wav(os.path.join(path_to_audio, 'all_audio.wav'), full_audio, sr=22050)
        # concatenate solo files
        full_audio_solo = concatenate_files(os.path.join(path_to_audio, 'solo/'), list_inst)
        librosa.output.write_wav(os.path.join(path_to_audio, 'all_audio_solo.wav'), full_audio_solo, sr=22050)

    else:
        # predict files
        from keras.models import model_from_json
        filename = 'all_audio_solo.wav'
        
        # load full_audio
        full_audio, sr = librosa.load(os.path.join(path_to_audio, filename))
        #extract spectrograms
        ln_S, X = create_spectrogram(full_audio)
        # load prediction model with TL
        model_transf = load_model(filename, path_to_models, True)
        # load prediction model without TL
        model_scratch = load_model(filename, path_to_models, False)
        # TRANSFER LEARNING!
        # make predictions
        pred_tl = model_transf.predict(X)
        # organize predictions
        org_pred_tl = organize_predictions(list_inst, labels_inst, pred_tl.T)
        # aggregate
        agg_pred_tl = aggregate_predictions(org_pred_tl)
        
        # SCRATCH LEARNING
        # make predictions
        pred_sc = model_scratch.predict(X)
        # organize predictions
        org_pred_sc = organize_predictions(list_inst, labels_inst, pred_sc.T)
        # aggregate
        agg_pred_sc = aggregate_predictions(org_pred_sc)
        
        # plot!
        plot_everything(list_inst, full_audio, ln_S, org_pred_tl, agg_pred_tl, org_pred_sc, agg_pred_sc)
