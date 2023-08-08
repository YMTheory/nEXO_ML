# This script is used to extract features from the raw waveforms
# A bessel filtering is firstly applied.
# Calculate fht, charge, max-min_Amp...
# Author: miaoyu@ihep.ac.cn

import numpy as np
import uproot as up
import h5py as h5
import argparse
import matplotlib.pyplot as plt
import re
import os

from scipy import signal

import plotConfig


def load_waveforms(rootfile, start, stop):

    infile = up.open(rootfile)
    intree = infile["Event/Elec/ElecEvent"]
    if stop == -1:
        stop = intree.num_entries

    #tileIds   = intree["ElecEvent/fElecChannels/fElecChannels.fTileId"].array(entry_start=start, entry_stop=stop)
    #xtile        = intree["ElecEvent/fElecChannels/fElecChannels.fxTile"].array(entry_start=start, entry_stop=stop)
    #ytile        = intree["ElecEvent/fElecChannels/fElecChannels.fyTile"].array(entry_start=start, entry_stop=stop)
    xpositions   = intree["ElecEvent/fElecChannels/fElecChannels.fXPosition"].array(entry_start=start, entry_stop=stop)
    ypositions   = intree["ElecEvent/fElecChannels/fElecChannels.fYPosition"].array(entry_start=start, entry_stop=stop)
    waveforms    = intree["ElecEvent/fElecChannels/fElecChannels.fWFAndNoise"].array(entry_start=start, entry_stop=stop)
    charges      = intree["ElecEvent/fElecChannels/fElecChannels.fChannelCharge"].array(entry_start=start, entry_stop=stop)
    localStripIDs= intree["ElecEvent/fElecChannels/fElecChannels.fChannelLocalId"].array(entry_start=start, entry_stop=stop)
    noiseTags    = intree["ElecEvent/fElecChannels/fElecChannels.fChannelNoiseTag"].array(entry_start=start, entry_stop=stop)

    return xpositions, ypositions, waveforms, charges, localStripIDs, noiseTags


def waveform_fft(waveform):
    fft_waveform = fft(waveform)
    amplitude_spectrum = np.abs(fft_waveform)
    phase_spectrum = np.angle(fft_waveform)


def waveform_filtering(waveform, N=4, Wn=0.3):
    # sampling rate is 2 MHz
    # bessel filter: cut-off freq is 300 kHz
    b, a = signal.bessel(N, Wn, btype='lowpass', analog=False, output='ba')
    filterd_waveform = signal.lfilter(b, a, waveform)
    return filterd_waveform 


def draw_waveforms(waveform, filterd_waveform):
    fig, ax = plt.subplots(figsize=(6, 4))
    t = np.arange(0, 0.5*len(waveform), 0.5)
    ax.plot(t, waveform,         "-", label='unfiltered')
    ax.plot(t, filterd_waveform, '-', label='filtered')
    plotConfig.setaxis(ax, xlabel='time [us]', ylabel='amplitude', lg=True, title='Bessel filter')
    plt.tight_layout()
    plt.show()


################################################################################################
### Features selection:

def abnormal_waveform(waveform, noiseTag, charge, threshold=400):
    # a simple condition:
    if -np.min(waveform) > threshold and np.max(waveform) < threshold:
        my_charge = simple_charge_integration(waveform, noiseTag, charge)
        if noiseTag:
            print(f'Get an abnormal waveform, which is a noise channel, the simTruth charge is {charge} and integral charge = {my_charge},')
        else:
            print(f'Get an abnormal waveform, which is not a noise channel, the simTruth charge is {charge} and integral charge = {my_charge},')

        return waveform, True, charge, my_charge
    return 0, False, 0, 0


### 1. integral charge
def simple_charge_integration(waveform, noiseTag, chargeSim):
    # simple charge intergration
    # for noise channel, read charge from simulation truth (from understanding of WN simulation)
    if noiseTag:
        return chargeSim
    charge = np.sum(waveform)  * 0.5 # as the original sampling rate is 2 MHz (0.5 us per point)
    return charge

### 2. max - min amplitude
def amplitude_max2min(waveform, threshold=800):
    time = np.arange(0, 0.5*len(waveform), 0.5)    # unit: us
    
    # negative amplitude
    index_neg = np.where(waveform < 0)
    time_neg = time[index_neg]
    waveform_neg = waveform[index_neg]
    waveform_neg_abs = np.abs(waveform_neg)
    peaks_neg, _ = signal.find_peaks(waveform_neg_abs, height=threshold)

    # positive amplitude
    index_pos = np.where(waveform >= 0)
    time_pos = time[index_pos]
    waveform_pos = waveform[index_pos]
    peaks_pos, _ = signal.find_peaks(waveform_pos, height=threshold)

    ## Largest max - min:
    amp_peaks_neg = 0
    if len(peaks_neg) > 0:
        amp_peaks_neg = -1000
        for i in peaks_neg:
            if waveform_neg_abs[i] > amp_peaks_neg:
                amp_peaks_neg = waveform_neg_abs[i]
        amp_peaks_neg = -amp_peaks_neg

    amp_peaks_pos = 0
    if len(peaks_pos) > 0:
        amp_peaks_pos = -1000
        for i in peaks_pos:
            if waveform_pos[i] > amp_peaks_pos:
                amp_peaks_pos = waveform_pos[i]


    return (amp_peaks_pos - amp_peaks_neg)

### 3. first pass-threshold time:
def first_passThr_time(waveform, threshold=800):
    # nominal threshold is set as 4*sigma of the white noise
    # simple passing threshold time, linear interpolation
    waveform = np.array(waveform)
    time = np.arange(0, 0.5*len(waveform), 0.5)    # unit: us
    for i in range(len(waveform)-1):
        if waveform[i] <= threshold <= waveform[i+1]:
            fht = time[i] + 0.5 / (waveform[i+1] - waveform[i]) * (threshold - waveform[i])
            return fht
    return -100

### 4. waveform width
def waveform_width(waveform, threshold=800):
    abs_waveform = np.abs(waveform) 
    uppass, downpass, widths = [], [], []
    time = np.arange(0, 0.5*len(waveform), 0.5)
    for i in range(len(waveform)-1):
        if waveform[i] <= threshold <= waveform[i+1]:
            val = time[i] + 0.5 / (waveform[i+1] - waveform[i]) * (threshold - waveform[i])
            uppass.append(val)
        if waveform[i] >= threshold >= waveform[i+1]:
            val = time[i] + (0.5 - 0.5 / (waveform[i+1] - waveform[i]) * (threshold - waveform[i+1]))
            downpass.append(val)

    uppass      = np.array(uppass)
    downpass    = np.array(downpass)

    if len(uppass) != len(downpass):
        print("Points do not match...")
        print(uppass)
        print(downpass)
        return [0]
    else:
        width = downpass - uppass
        return width


################################################################################################



def run():

    parser = argparse.ArgumentParser(description='Arguments of data manipulation.')
    parser.add_argument("--start",      type=int,       default=0,              help="Start event number.")
    parser.add_argument("--stop",       type=int,       default=10,             help="End event number.")
    parser.add_argument('--rootfile',   type=str,       default='test.root',    help='Input root file.')
    #parser.add_argument('--h5file',     type=str,       default='test.h5',      help='Input h5 file.')
    args = parser.parse_args()

    particle = 'bb0n'
    if "gamma" in args.rootfile:
        particle = "gamma"

    match = re.search(r"seed(\d+)", args.rootfile)
    seed = match.group(1)

    h5file = f'/junofs/users/miaoyu/0nbb/nEXO_simulation/scripts/h5files/nexo_{particle}_seed{seed}_v1_start{args.start}stop{args.stop}.h5'
    if os.path.exists(h5file):
        fh5out = h5.File(h5file, 'a')
    else:
        fh5out = h5.File(h5file, 'w')

    groupname = f'nexo_data_seed{seed}'
    if groupname not in fh5out.keys():
        group = fh5out.create_group(groupname)
    else:
        group = fh5out[groupname]

    NCHANNEL        = 3
    MAX_NTILE       = 14
    NSTRIP_PERTILE  = 16
    TILE_WIDTH      = 96 ## mm
    STRIP_WIDTH     = 6 ## mm

    AMP_THRESHOLD   = 800  # unit: e-, 4 * sigma of white noise

    AMP_SCALE       = 9  # from nEXO offline

    XOFFSET         = 7 * TILE_WIDTH
    YOFFSET         = 7 * TILE_WIDTH

    xpositions, ypositions, waveforms, charges, localStripIDs, noiseTags = load_waveforms(args.rootfile, args.start, args.stop)    

    for ievt in range(len(waveforms)):

        images_xstrip = np.ones((NCHANNEL, MAX_NTILE, MAX_NTILE*NSTRIP_PERTILE), dtype=np.float16) * (-100)
        images_ystrip = np.ones((NCHANNEL, MAX_NTILE*NSTRIP_PERTILE, MAX_NTILE), dtype=np.float16) * (-100)

        for iwf in range(len(waveforms[ievt])):
            cur_wf = np.array(waveforms[ievt][iwf]) * AMP_SCALE
            filterd_waveform = waveform_filtering(cur_wf)

            #draw_waveforms(waveforms[ievt][0], filterd_waveform)

            charge  = simple_charge_integration(filterd_waveform, noiseTags[ievt][iwf], charges[ievt][iwf])
            damp    = amplitude_max2min(filterd_waveform)
            fht     = first_passThr_time(filterd_waveform, threshold=AMP_THRESHOLD)
            #widths  = waveform_width(filterd_waveform)

            if localStripIDs[ievt][iwf] < 16:   # x-strip
                #print("X_strip:   ", end=" ")
                x0, y0 = xpositions[ievt][iwf], ypositions[ievt][iwf]
                x, y = x0 + XOFFSET, y0 + YOFFSET
                images_xstrip[0, int(y/TILE_WIDTH), int(x/STRIP_WIDTH)] = charge
                images_xstrip[1, int(y/TILE_WIDTH), int(x/STRIP_WIDTH)] = damp
                images_xstrip[2, int(y/TILE_WIDTH), int(x/STRIP_WIDTH)] = fht

                #print(f"{x0}->{x}, {y0}->{y}, ({int(y/TILE_WIDTH)}, {int(x/STRIP_WIDTH)})")

                #if fht > 0:
                #    print(2, int(y/TILE_WIDTH), int(x/STRIP_WIDTH), fht )
                #    print(images_xstrip[2, int(y/TILE_WIDTH), int(x/STRIP_WIDTH)])

            else: # y-strip
                #print("Y_strip:  ", end=" ")
                x0, y0 = xpositions[ievt][iwf], ypositions[ievt][iwf]
                x, y = x0 + XOFFSET, y0 + YOFFSET
                #print(f"{x0}->{x}, {y0}->{y}, ({int(y/STRIP_WIDTH)}, {int(x/TILE_WIDTH)})")

                images_ystrip[0, int(y/STRIP_WIDTH), int(x/TILE_WIDTH)] = charge
                images_ystrip[1, int(y/STRIP_WIDTH), int(x/TILE_WIDTH)] = damp
                images_ystrip[2, int(y/STRIP_WIDTH), int(x/TILE_WIDTH)] = fht

                #if fht > 0:
                #    print(2, int(y/STRIP_WIDTH), int(x/TILE_WIDTH), fht )
                #    print(images_ystrip[2, int(y/STRIP_WIDTH), int(x/TILE_WIDTH)])

        images = np.concatenate((images_xstrip, np.transpose(images_ystrip, (0, 2, 1))), axis=0)

        datasetname = f'{particle}_seed{seed}_event{ievt}'
        group.create_dataset(f'{particle}_seed{seed}_event{ievt}', data=images)

    fh5out.close()

        #np.save(f"{subdir}/{particle}_seed{seed}_event{ievt+args.start}_image2d_xfeatures", images_xstrip)
        #np.save(f"{subdir}/{particle}_seed{seed}_event{ievt+args.start}_image2d_yfeatures", images_ystrip)



#main()

