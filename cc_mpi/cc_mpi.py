#!/usr/bin/env python
import sys
import numpy as np
import math
import os
import glob
import collections
import subprocess
import utm
import obspy
import glob
import time
from obspy.clients.fdsn import Client
from obspy.clients.iris import Client as tt
from obspy import UTCDateTime
from obspy import read
from numba import jit, cuda
from mpi4py import MPI
from obspy.signal.cross_correlation import correlate
from scipy.signal import detrend, resample
from utils import *

# Launch MPI
time_start = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
status = MPI.Status()
nthreads = comm.Get_size()

###### EDIT THIS #######
nchunk = 10  # number of data packets/chunks
########################
tags = enum('READY', 'DONE', 'EXIT', 'START')

# Count files needed
########## EDIT THIS #######################
files = glob.glob('waveforms/*.MSEED')  # Location of files changeable
############################################
nfile = len(files)

# Start Client
# Pull station inventory
# Start Client

###### EDIT THIS #################################################################
client = Client('NCEDC')
tt_iris = tt()
inventory = client.get_stations(network='NC')

# Info from Event.dat file
# FOR LARGE DATASETS THIS WILL NEED TO CHANGE TO READ IN EVENT.DAT INFORMATION
eventnames = ["38542", "484120", "30107759", "30065107", "30058032", "402094", "30034705",
              "238298", "86036", "52942", "48565", "45165", "44289", "38520"]
###################################################################################

evpairs = []
for event1 in range(0, len(eventnames) - 1):
    for event2 in range(len(eventnames) - 1, event1, -1):
        evpairs.append((eventnames[event1], eventnames[event2]))

# Pull station inventory
stalst = np.empty(len(inventory[0]), dtype='object')
for stindx, stations in enumerate(inventory[0]):
    stat = stations.code
    stalst[stindx] = stat

nev = len(eventnames)
nsta = len(stalst)

if rank == 0:  # Master branch
    # Read in information
    waveforms = np.empty((nfile), dtype='int')
    ev_index = np.empty((nfile), dtype='int')
    sta_index = np.empty((nfile), dtype='int')
    ncc = math.ceil(nfile * (nfile - 1) / 2)
    step = math.ceil(ncc / nchunk)

    wf_count = 0
    for ind, file in enumerate(files):
        waveforms[wf_count] = ind  # Point to file - file read in worker
        # File index in glob file list
        strs = (file.split('/'))[1]
        event = (strs.split('.'))[0]
        station = (strs.split('.'))[1]

        try:
            eind = eventnames.index(event)
            sind = np.where(stalst == station)[0][0]
        except:
            continue

        ev_index[wf_count] = eind  # EQ event index in list
        sta_index[wf_count] = sind  # Stat index in list
        wf_count = wf_count + 1

    ev_index = ev_index[0:wf_count - 1]  # Cut to size
    sta_index = sta_index[0:wf_count - 1]  # Cut to size
    waveforms = waveforms[0:wf_count - 1]  # Cut to size
    wf_count = wf_count - 1
else:
    wf_count = 0

# Broadcast indexes to all workers
wf_count = comm.bcast(wf_count, root=0)
if rank > 0:
    ev_index = np.empty((wf_count), dtype='int')
    sta_index = np.empty((wf_count), dtype='int')
    waveforms = np.empty((wf_count), dtype='int')
comm.Bcast(ev_index, root=0)
comm.Bcast(sta_index, root=0)
comm.Bcast(waveforms, root=0)

# Now to cross correlations
if rank == 0:  # Master
    num_workers = nthreads - 1
    closed_workers = 0
    chunk = 0

    # Make sent/receive data
    indexes = np.zeros((math.ceil(nfile * (nfile - 1) / 2), 2),
                       dtype='int')  # This holds the sendable index data for reference to broadcasted arrays
    Nccs = 0
    for i in range(0, wf_count):
        for j in range(i + 1, wf_count):
            if sta_index[i] == sta_index[j]:
                indexes[Nccs, 0] = i  # First waveform in xcorr ---- what index in arrays ev_index,sta_index,waveforms
                indexes[Nccs, 1] = j  # Second waveform in xcorr ---- what index in arrays ev_index,sta_index,waveforms
                Nccs += 1

    indexes = indexes[0:Nccs, :]  # Trim to size

    # Data chunks for number of processes to run
    len_chunk = int(np.ceil(Nccs / nchunk))
    len_lastchunk = (Nccs - len_chunk * (nchunk - 1))

    ccs_final = np.zeros((Nccs, 7), dtype='double')  # This holds the returned xcorr data
    # CC data array
    ####################
    # [:,0] = ev1 index
    # [:,1] = ev2 index
    # [:,2] = st index
    # [:,3] = P cc dt time
    # [:,4] = P weight
    # [:,5] = S cc dt time
    # [:,6] = S weight
    ####################
    # -999 is nan

    while closed_workers < num_workers:  # Workers close when idle --- loop runs until all workers are closed
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        worker = status.Get_source()
        worker_tag = status.Get_tag()
        if worker_tag == tags.READY:  # If worker signals it is ready
            print('Worker ', worker, ' ready.')
            if chunk < nchunk:  # If there is still data to be sent
                if chunk == nchunk - 1:  # If last chunk --- different size
                    print('Sending task #', chunk, ' to worker ', worker)
                    index_send = np.zeros(2 * (len_lastchunk) + 2,
                                          dtype=int)  # Must reformat indexes to 1D array to be sent to worker
                    index_send[0] = chunk  # First index is data chunk #
                    index_send[1] = len_lastchunk  # Second index is the length of the data chunk
                    # Then send indexes
                    index_send[2:(len_lastchunk + 2)] = indexes[chunk * len_chunk:, 0]
                    index_send[(len_lastchunk + 2):] = indexes[chunk * len_chunk:, 1]
                    comm.send(index_send, dest=worker, tag=tags.START)  # Send to worker
                else:
                    print('Sending task #', chunk, ' to worker ', worker)
                    index_send = np.zeros(2 * len_chunk + 2,
                                          dtype=int)  # Must reformat indexes to 1D array to be sent to worker
                    index_send[0] = chunk  # First index is data chunk #
                    index_send[1] = len_chunk  # Second index is the length of the data chunk
                    # Then send indexes
                    index_send[2:(len_chunk + 2)] = indexes[chunk * len_chunk:chunk * len_chunk + len_chunk, 0]
                    index_send[(len_chunk + 2):(len_chunk + 2) + len_chunk] = indexes[chunk * len_chunk:chunk * len_chunk + len_chunk, 1]
                    comm.send(index_send, dest=worker, tag=tags.START)  # Send to worker
            else:  # If there's no more work --- signal to close the worker
                print('Closing worker #', worker)
                comm.send(None, dest=worker, tag=tags.EXIT)
            chunk += 1
        elif worker_tag == tags.DONE:  # If x-corr data is received from the worker --- i.e. process DONE
            data_chunk = int(data[0])  # Which data chunk received
            if data_chunk == nchunk - 1:  # If last chunk
                data = data[1:]
                data = data.reshape((len_lastchunk, 7), order='F')  # Reshape to NDArray
                ccs_final[data_chunk * len_chunk:, :] = data  # Save to data array
                print('Saving data process #', data_chunk, ' from worker ', worker)
            else:
                data = data[1:]
                data = data.reshape((len_chunk, 7), order='F')  # Reshape to ND array
                ccs_final[data_chunk * len_chunk:data_chunk * len_chunk + len_chunk, :] = data  # Save to data array
                print('Saving data process #', data_chunk, ' from worker ', worker)
        elif worker_tag == tags.EXIT:  # If worker has been closed
            closed_workers += 1

    # Print to CC file ---- for hypoDD format only

    # Replace indexes with event ids and station names
    ccs_final = ccs_final.astype(object)
    ccs_final[:, 1:3] = ccs_final[:, 1:3].astype(int)
    for i in range(0, len(ccs_final[:, 0])):
        ccs_final[i, 0] = int(eventnames[ev_index[int(ccs_final[i, 0])]])
        ccs_final[i, 1] = int(eventnames[ev_index[int(ccs_final[i, 1])]])
        ccs_final[i, 2] = str(stalst[int(ccs_final[i, 2])])
    np.save('cc_arr.npy', ccs_final)  # Save data array
    ccs_final = ccs_final[ccs_final[:, 2].argsort()]  # Sort by station name (gives sorted xcorr file)

    # Sort by evpair order
    a = np.empty(ccs_final.shape, dtype='object')
    evpairs = np.asarray(evpairs, dtype='int')
    count = 0
    for ev_pair in evpairs:
        ev1 = ev_pair[0]
        ev2 = ev_pair[1]
        for ind in range(0, len(ccs_final)):
            if ccs_final[ind, 0] == ev1 and ccs_final[ind, 1] == ev2:
                a[count, :] = ccs_final[ind, :]
                count += 1
            elif ccs_final[ind, 1] == ev1 and ccs_final[ind, 0] == ev2:
                a[count, :] = ccs_final[ind, :]
                a[count, 0] = ev1
                a[count, 1] = ev2
                a[count, 3] = -1. * ccs_final[ind, 3]
                a[count, 5] = -1. * ccs_final[ind, 5]
                count += 1
    np.save('cc_arr_sorted.npy', a)  # Save sorted data array

    # Now write to file
    ccfiles = open('MPI_CCS.cc', 'w')
    oldev1 = 0
    oldev2 = 0
    for line in a:
        ev1 = line[0]
        ev2 = line[1]
        if ev1 == oldev1 and ev2 == oldev2:
            if line[4] > 0.5:
                ccfiles.write('NC%s %2.6f %2.6f P \n' % (line[2], line[3], line[4]))
            if line[6] > 0.5:
                ccfiles.write('NC%s %2.6f %2.6f S \n' % (line[2], line[5], line[6]))
        else:
            ccfiles.write('# %i %i 0.0 \n' % (ev1, ev2))
            if line[4] > 0.5:
                ccfiles.write('NC%s %2.6f %2.6f P \n' % (line[2], line[3], line[4]))
            if line[6] > 0.5:
                ccfiles.write('NC%s %2.6f %2.6f S \n' % (line[2], line[5], line[6]))
            oldev1 = ev1
            oldev2 = ev2
    ccfiles.close()

    # Timed process ends here
    print('Time to run: ', time.time() - time_start)

elif rank > 0:  # IF WORKER
    while True:
        comm.send(None, dest=0, tag=tags.READY)  # If idle send Ready
        cc_indexes = comm.recv(source=0, status=status)  # Receive indexes for xcorr or exit tag
        worker_tag = status.Get_tag()

        if worker_tag == tags.START:  # If process to xcorr
            chunk_num = cc_indexes[0]  # index 0 is the chunk #
            chunk_len = cc_indexes[1]  # index 1 is the length of the chunk
            cc_indexes = cc_indexes[2:].reshape((chunk_len, 2), order='F')  # Reshape to fit data
            print('Worker #', rank, ' starting cc process #', chunk_num)
            cc_worker = np.zeros((chunk_len, 7), dtype='object')  # Data array to return

            ####################
            ####################
            # CC CODE STARTS HERE
            ####################

            for ipair, ind in enumerate(cc_indexes[:, 0]):  # Loop over all indexes to x-correlate
                # Pull events and stations
                ev1 = eventnames[ev_index[ind]]
                ev2 = eventnames[ev_index[cc_indexes[ipair, 1]]]
                cc_worker[ipair, 0] = ind
                cc_worker[ipair, 1] = cc_indexes[ipair, 1]
                sta1 = sta_index[ind]
                sta2 = sta_index[cc_indexes[ipair, 1]]
                if sta1 != sta2:
                    for i in range(0, 7):
                        cc_worker[ind, i] = -999
                    continue
                cc_worker[ipair, 2] = sta_index[ind]
                station = stalst[sta1]
                # Read in waves --- into obspy
                waveform1 = waveforms[ind]
                waveform2 = waveforms[cc_indexes[ipair, 1]]
                stream1 = read(files[waveform1])
                stream2 = read(files[waveform2])
                # Process streams
                stream1.detrend('constant')
                stream1.detrend('linear')
                stream1.taper(0.1)
                stream2.detrend('constant')
                stream2.detrend('linear')
                stream2.taper(0.1)
                stream1.filter(type='bandpass', freqmin=1.5, freqmax=15)
                stream2.filter(type='bandpass', freqmin=1.5, freqmax=15)

                # Pull arrival information for event1
                cat = client.get_events(eventid=int(ev1), includearrivals=True)
                picks = cat[0].picks
                time1 = cat[0].origins[0].time
                parr1 = 0.0
                sarr1 = 0.0
                p1 = 0.
                s1 = 0.
                for pick in picks:  # If the pick has been saved to catalog
                    if pick.waveform_id.station_code == station:
                        picktime = pick.time
                        picktime = picktime  # - UTCDateTime(time)
                        if pick.waveform_id.channel_code[-1] == 'Z':
                            parr1 = picktime
                        elif pick.waveform_id.channel_code[-1] == 'N':
                            sarr1 = picktime
                    if parr1 != 0.:
                        p1 = parr1 - UTCDateTime(time1)
                    if sarr1 != 0.:
                        s1 = sarr1 - UTCDateTime(time1)
                if parr1 == 0.0:  # Otherwise calculate the P and S arrival
                    lat = cat[0].origins[0].latitude
                    lon = cat[0].origins[0].longitude
                    dep = cat[0].origins[0].depth / 1000
                    inventory = client.get_stations(network='NC', sta=station)
                    slat = inventory[0][0].latitude
                    slon = inventory[0][0].longitude
                    result = tt_iris.traveltime(model='iasp91', phases=['p'], evdepth=dep, evloc=(lat, lon),
                                                staloc=(slat, slon))
                    result = list(filter(None, str(result).split(' ')))
                    parr1 = time1 + float(result[26])
                    p1 = float(result[26])
                if sarr1 == 0.0:
                    lat = cat[0].origins[0].latitude
                    lon = cat[0].origins[0].longitude
                    dep = cat[0].origins[0].depth / 1000
                    inventory = client.get_stations(network='NC', sta=station)
                    slat = inventory[0][0].latitude
                    slon = inventory[0][0].longitude
                    result = tt_iris.traveltime(model='iasp91', phases=['s'], evdepth=dep, evloc=(lat, lon),
                                                staloc=(slat, slon))
                    result = list(filter(None, str(result).split(' ')))
                    sarr1 = time1 + float(result[26])
                    s1 = float(result[26])
                # Trim streams to 3 length centered on arrivals
                str1_p = stream1.copy()
                str1_s = stream1.copy()
                str1_p.trim(starttime=parr1 - 1.5, endtime=parr1 + 1.5)
                str1_s.trim(starttime=sarr1 - 1.5, endtime=sarr1 + 1.5)

                # Repeat for event 2
                cat = client.get_events(eventid=int(ev2), includearrivals=True)
                picks = cat[0].picks
                time2 = cat[0].origins[0].time
                parr2 = 0.0
                sarr2 = 0.0
                p2 = 0.
                s2 = 0.
                for pick in picks:
                    if pick.waveform_id.station_code == station:
                        picktime = pick.time
                        picktime = picktime  # - UTCDateTime(time)
                        if pick.waveform_id.channel_code[-1] == 'Z':
                            parr2 = picktime
                        elif pick.waveform_id.channel_code[-1] == 'N':
                            sarr2 = picktime
                    if parr2 != 0.:
                        p2 = parr2 - UTCDateTime(time2)
                    if sarr2 != 0.:
                        s2 = sarr2 - UTCDateTime(time2)
                if parr2 == 0.0:
                    lat = cat[0].origins[0].latitude
                    lon = cat[0].origins[0].longitude
                    dep = cat[0].origins[0].depth / 1000
                    inventory = client.get_stations(network='NC', sta=station)
                    slat = inventory[0][0].latitude
                    slon = inventory[0][0].longitude
                    result = tt_iris.traveltime(model='iasp91', phases=['p'], evdepth=dep, evloc=(lat, lon),
                                                staloc=(slat, slon))
                    result = list(filter(None, str(result).split(' ')))
                    parr2 = time2 + float(result[26])
                    p2 = float(result[26])
                if sarr2 == 0.0:
                    lat = cat[0].origins[0].latitude
                    lon = cat[0].origins[0].longitude
                    dep = cat[0].origins[0].depth / 1000
                    inventory = client.get_stations(network='NC', sta=station)
                    slat = inventory[0][0].latitude
                    slon = inventory[0][0].longitude
                    result = tt_iris.traveltime(model='iasp91', phases=['s'], evdepth=dep, evloc=(lat, lon),
                                                staloc=(slat, slon))
                    result = list(filter(None, str(result).split(' ')))
                    sarr2 = time2 + float(result[26])
                    s2 = float(result[26])
                str2_p = stream2.copy()
                str2_s = stream2.copy()
                str2_p.trim(starttime=parr2 - 1.5, endtime=parr2 + 1.5)
                str2_s.trim(starttime=sarr2 - 1.5, endtime=sarr2 + 1.5)

                # Calculate arrival dts -- cc dt added to this
                pdt = p1 - p2
                sdt = s1 - s2

                # For s streams 0 pad the data --- necessary for some small P-S separation events
                str1_s_0pad = str1_s.copy()
                str2_s_0pad = str2_s.copy()
                str1_s_0pad[0].data[0:100] = 0.
                str2_s_0pad[0].data[0:100] = 0.
                # Apply a one-sided hanning window to smooth transition from 0padding to signal
                hanning = np.hanning(20)
                hwind = np.ones((301))
                hwind[0:100] = 0.
                hwind[100:110] = hanning[0:10]
                str1_s_0pad[0].data = str1_s_0pad[0].data * hwind
                str2_s_0pad[0].data = str2_s_0pad[0].data * hwind

                # Cross-correlations for both P and S including upsampling
                [ccp, timingp, max_idxp, dtp] = cross_correlate(str1_p[0].data, str2_p[0].data,
                                                                str1_p[0].stats.sampling_rate)
                [cc_upp, timing_upp, max_idx_upp, dt_upp] = upsample(ccp, 100, stream1[0].stats.sampling_rate)
                [ccs_0pad, timings_0pad, max_idxs_0pad, dts_0pad] = cross_correlate(str1_s_0pad[0].data,
                                                                                    str2_s_0pad[0].data,
                                                                                    str1_s[0].stats.sampling_rate)
                [cc_ups_0pad, timing_ups_0pad, max_idx_ups_0pad, dt_ups_0pad] = upsample(ccs_0pad, 100,
                                                                                         stream1[0].stats.sampling_rate)
                # Save to data array
                cc_worker[ipair, 3] = pdt + dt_upp
                cc_worker[ipair, 4] = cc_upp[max_idx_upp]
                cc_worker[ipair, 5] = sdt + dt_ups_0pad
                cc_worker[ipair, 6] = cc_ups_0pad[max_idx_ups_0pad]

            # Once all cross-correlations are calculated reshape data array
            print('Sending process #', chunk_num, ' back to root from worker #', rank)
            send_back = np.zeros((chunk_len * 7 + 1))
            send_back[0] = chunk_num  # Index 0 is chunk #
            send_back[1:] = cc_worker.reshape((chunk_len * 7), order='F')  # Reshape to 1D
            comm.send(send_back, dest=0, tag=tags.DONE)  # Return to master

        elif worker_tag == tags.EXIT:
            break  # Break out of while loop to exit the process, exit tag sent from master
    comm.send(None, dest=0, tag=tags.EXIT)  # Exited worker

# comm.barrier()
sys.stdout.flush()  # Flush system
# comm.Finalize()
