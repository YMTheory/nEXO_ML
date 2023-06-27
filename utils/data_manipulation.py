import numpy as np
import uproot as up
import h5py as h5
import argparse
import matplotlib.pyplot as plt
import re
import os

import plotConfig

# In this script, we build images for DNN basically based on description from zepeng li (Simulation of charge readout with segmented tiles in nEXO, Journal of Instrumentation 14.09 (2019): P09020.)
# The image size is (250, 250, 3), 1st for position idnex, 2nd for wavelength time (after down-sampling), 3 for if x or y.
# Center of the imgae is selecting by iterating all recorded channels, and the one which contains most channels in the image is selected.
# Waveform is down-sampled as 250, following the method mentioned in the reference.

def threeTypesChannelTags(charges, noiseTag):
    collTag, indTag = np.zeros(len(charges)), np.zeros(len(charges))
    for i in range(len(charges)):
        if not noiseTag[i]:
            if charges[i] == 0:
                indTag[i] = 1
            if charges[i] > 0:
                collTag[i] = 1
    return collTag, indTag


def channel_threshold_compareTruth(waveforms, noiseTag, collTag, indTag, integral_cut=450, induction_cut=800):
    # input: all recorded waveforms from elecsim
    # output: indices for channels which pass the selection thresholds:
    ## 1. integral charge > integral_cut;
    ## 2. if 1 is not satisfied, maxWF-minWF > induction_cut -> induction channels.

    NrightColl, NrightInd, NrightNoise = 0, 0, 0
    NwrongTyped_noiseCha = 0
    useful_channelID, useful_induced_channelID = [], []
    scale = 9   # Note the waveform amplitude is divided a factor of 9 from the true current (from nEXO offline doc)
    for i, wf in enumerate(waveforms):
        charge = 0.5 * np.sum(wf) * scale  # 0.5 for 2MHz sampling rate
        if charge > integral_cut:
            useful_channelID.append(i)
        else:
            if scale * (np.max(wf) - np.min(wf)) > induction_cut:
                useful_channelID.append(i)
                useful_induced_channelID.append(i)
    for i in useful_channelID:
        if noiseTag[i]:
            NwrongTyped_noiseCha += 1
        if collTag[i]:
            NrightColl += 1
        if indTag[i]:
            NrightInd += 1
    NrightNoise = np.sum(noiseTag) - NwrongTyped_noiseCha

    useful_channelID = np.array(useful_channelID)
    NwrongColl = int(sum(collTag) - NrightColl)
    NwrongInd = int(sum(indTag) - NrightInd)
    NwrongNoise = int(NwrongTyped_noiseCha)
    return useful_channelID, useful_induced_channelID, NwrongColl, NwrongInd, NwrongNoise


def separateXYchannels(useful_channelID, channelID, xposition, yposition):
    xchannels_id, ychannels_id = [], []
    for i in useful_channelID:
        if channelID[i] < 16:   # this is a x-strip
            xchannels_id.append(i)
        else: # this is a y-strip
            ychannels_id.append(i)

    return xchannels_id, ychannels_id



def calcImageHeight(xcenter, ycenter, channels, xposition, yposition, flag, height=250):
    heights = []
    channelid = []
    for i in channels:
        x, y = xposition[i], yposition[i]
        if flag == "x":
            H = int((x - xcenter)/6) + 16*int((y - ycenter)/16.) + int(height/2)
            if H < 0 or H >= height:
                continue
            else:
                channelid.append(i)
                heights.append(H)
        
        if flag == "y":
            H = int((y - ycenter)/6) + 16*int((x - xcenter)/16.) + int(height/2)
            if H < 0 or H >= height:
                continue
            else:
                channelid.append(i)
                heights.append(H)

    heights = np.array(heights)
    return channelid, heights


def selectCenter(channels, xposition, yposition, flag, height=250):
    Nheight = 0
    xcenter, ycenter = 0, 0
    chaid, heights = [], []
    for i in channels:
        xcenter0, ycenter0 = xposition[i], yposition[i]
        chaid0, heights0 = calcImageHeight(xcenter0, ycenter0, channels, xposition, yposition, flag=flag, height=height)
        if len(heights0) > Nheight:
            Nheight = len(heights0)
            xcenter = xcenter0
            ycenter = ycenter0
            chaid = chaid0
            heights = heights0

    return xcenter, ycenter, chaid, heights



def downSampling(waveform):
    time_interval   = 0.5 ## us
    rate0           = 2e3
    rates           = [1e3, 500, 333, 250]   # unit: kHz
    points          = [0, int(40/time_interval), int(75/time_interval), int(100/time_interval)]

    # settle time interval for different down-sampling rate
    wflen           = len(waveform)
    for i in range(len(points)-1):
        if points[i] < wflen <= points[i+1]:
            points_used = points[0:i+1]
            rates_used = rates[0:i+1]
            break
    if wflen > points[i+1]:
        points_used = points
        rates_used  = rates

    points_used.append(-1)

    # do down-sampling
    waveform_rev = waveform[::-1]
    waveform_downsampling = []
    for i in range(len(rates_used)) :
        merge_Npoints = int(rate0/rates_used[i])
        tmp_waveform0 = waveform_rev[points_used[i]:points_used[i+1]]
        tmp_waveform = []
        n_cluster = int(len(tmp_waveform0)/merge_Npoints)
        remainder = len(tmp_waveform0) % merge_Npoints
        for j in range(n_cluster):
            tmp_waveform.append( np.sum(tmp_waveform0[j:j+merge_Npoints])/merge_Npoints )
        if remainder != 0:
            tmp_waveform.append( np.sum(tmp_waveform0[int(n_cluster*merge_Npoints):-1])/remainder )

        waveform_downsampling += tmp_waveform

    waveform_downsampling = waveform_downsampling + [0 for k in range(250-len(waveform_downsampling))]
    waveform_downsampling = np.array(waveform_downsampling)

    return waveform_downsampling[::-1]


def buildImage(waveform, channels, heights):
    image2d = np.zeros((250, 250))
    for i, c in enumerate(channels):
        waveform_downsampled = downSampling(waveform[c])
        H = heights[i]
        image2d[H, :] += waveform_downsampled

    return image2d



def data_manipulation():

    start       = 0
    stop        = 100
    rootfile    = "test.root"
    h5file      = "test.h5"
    particle    = "bb0n"

    parser = argparse.ArgumentParser(description='Arguments of data manipulation.')
    parser.add_argument("--start",      type=int,       default=0,              help="Start event number.")
    parser.add_argument("--stop",       type=int,       default=10,             help="End event number.")
    parser.add_argument('--rootfile',   type=str,       default='test.root',    help='Input root file.')
    parser.add_argument('--h5file',     type=str,       default='test.h5',      help='Input h5 file.')
    args = parser.parse_args()

    start   = args.start
    stop    = args.stop
    rootfile= args.rootfile
    h5file  = args.h5file
    if "gamma" in rootfile:
        particle = "gamma"

    match = re.search(r"seed(\d+)", rootfile)
    seed = match.group(1)

    infile = up.open(rootfile)
    intree = infile["Event/Elec/ElecEvent"]
    if stop == -1:
        stop = intree.num_entries

    #tileIds   = intree["ElecEvent/fElecChannels/fElecChannels.fTileId"].array(entry_start=start, entry_stop=stop)
    xtile     = intree["ElecEvent/fElecChannels/fElecChannels.fxTile"].array(entry_start=start, entry_stop=stop)
    ytile     = intree["ElecEvent/fElecChannels/fElecChannels.fyTile"].array(entry_start=start, entry_stop=stop)
    xposition = intree["ElecEvent/fElecChannels/fElecChannels.fXPosition"].array(entry_start=start, entry_stop=stop)
    yposition = intree["ElecEvent/fElecChannels/fElecChannels.fYPosition"].array(entry_start=start, entry_stop=stop)
    waveforms = intree["ElecEvent/fElecChannels/fElecChannels.fWFAndNoise"].array(entry_start=start, entry_stop=stop)
    charge    = intree["ElecEvent/fElecChannels/fElecChannels.fChannelCharge"].array(entry_start=start, entry_stop=stop)
    channelID = intree["ElecEvent/fElecChannels/fElecChannels.fChannelLocalId"].array(entry_start=start, entry_stop=stop)
    noiseTag  = intree["ElecEvent/fElecChannels/fElecChannels.fChannelNoiseTag"].array(entry_start=start, entry_stop=stop)


    for i in range(stop-start):
        if i%100 == 0:
            print(f"Finished {i} event.")
        collTag, indTag = threeTypesChannelTags(charge[i], noiseTag[i]) 
        useful_channelID, useful_induced_channelID, NwrongColl, NwrongInd, NwrongNoise = channel_threshold_compareTruth(waveforms[i], noiseTag[i], collTag, indTag, integral_cut=450, induction_cut=800)
        xchannels_id, ychannels_id = separateXYchannels(useful_channelID, channelID[i], xposition[i], yposition[i])
        xchannels_xcenter, xchannels_ycenter, chaid_x, xchannels_heights = selectCenter(xchannels_id, xposition[i], yposition[i], "x")
        ychannels_xcenter, ychannels_ycenter, chaid_y, ychannels_heights = selectCenter(ychannels_id, xposition[i], yposition[i], "y")
        #print(f"Charge center for x-channels: [{xchannels_xcenter}, {xchannels_ycenter}] with included strips {len(xchannels_heights)}.")
        #print(f"Charge center for y-channels: [{ychannels_xcenter}, {ychannels_ycenter}] with included strips {len(ychannels_heights)}.")
        image2d_x = buildImage(waveforms[i], chaid_x, xchannels_heights)
        image2d_y = buildImage(waveforms[i], chaid_y, ychannels_heights)

        image2d = np.zeros((250, 250, 3))
        image2d[:, :, 0] = image2d_x
        image2d[:, :, 1] = image2d_y
    
        subdir = f"/junofs/users/miaoyu/0nbb/nEXO_simulation/scripts/channelq_npy/{particle}_seed{seed}_start{start}stop{stop}"
        if not os.path.isdir(subdir):
            os.system(f'mkdir {subdir}')
        np.save(f"{subdir}/{particle}_seed{seed}_event{i+start}_image2d", image2d)

if __name__ == '__main__' :
    data_manipulation()








