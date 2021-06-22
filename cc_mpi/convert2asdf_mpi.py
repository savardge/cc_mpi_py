import pyasdf
import glob
import os
import time
import sys
import numpy as np
from mpi4py import MPI
import obspy
# os.system('export HDF5_USE_FILE=FALSE')
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def enum(*sequential, **named):
    """
    Way to fake an enumerated type in Python
    Pulled from:  http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


# USER INPUT
nchunk = 11
qmdir = "/home/genevieve.savard/cami/quakemigrate/outputs/dets_fromscamp"
filelist = glob.glob(os.path.join(qmdir, "quakeml", "2021*.xml"))
output_dir = "/home/genevieve.savard/cami/quakemigrate/outputs/dets_fromscamp_h5"
overwrite = True
staxmllist = glob.glob("/Volumes/SanDisk_2TB/cami/stationxml/station_cami_all_hawk250sps.xml")
# --------------------------------------------------------------------

nfile = len(filelist)
# Launch MPI
tags = enum('READY', 'DONE', 'EXIT', 'START')
time_start = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nthreads = comm.Get_size()
status = MPI.Status()

if rank == 0:  # Master
    num_workers = nthreads - 1
    closed_workers = 0
    chunk = 0

    # Make sent/receive data
    indexes = np.arange(nfile)  # This holds the sendable index data for reference to broadcasted arrays
    splits = len(indexes)

    # Data chunks for number of processes to run
    len_chunk = int(np.ceil(splits / nchunk))
    len_lastchunk = (splits - len_chunk * (nchunk - 1))

    results = np.zeros(splits, dtype=int)

    print("Now distributing tasks to workers")
    while closed_workers < num_workers:  # Workers close when idle --- loop runs until all workers are closed
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        worker = status.Get_source()
        worker_tag = status.Get_tag()
        if worker_tag == tags.READY:  # If worker signals it is ready
            print('Worker ', worker, ' ready.')
            if chunk < nchunk:  # If there is still data to be sent
                if chunk == nchunk - 1:  # If last chunk --- different size
                    print('Sending task #', chunk, ' to worker ', worker)
                    index_send = np.zeros(len_lastchunk + 2, dtype=int)  # Must reformat indexes to 1D array to be sent to worker
                    index_send[0] = chunk  # First index is data chunk #
                    index_send[1] = len_lastchunk  # Second index is the length of the data chunk
                    # Then send indexes
                    index_send[2:] = indexes[chunk * len_chunk:]
                    comm.send(index_send, dest=worker, tag=tags.START)  # Send to worker
                else:
                    print('Sending task #', chunk, ' to worker ', worker)
                    index_send = np.zeros(len_chunk + 2, dtype=int)  # Must reformat indexes to 1D array to be sent to worker
                    index_send[0] = chunk  # First index is data chunk #
                    index_send[1] = len_chunk  # Second index is the length of the data chunk
                    # Then send indexes
                    index_send[2:] = indexes[chunk * len_chunk:chunk * len_chunk + len_chunk]
                    comm.send(index_send, dest=worker, tag=tags.START)  # Send to worker
            else:  # If there's no more work --- signal to close the worker
                print('Closing worker #', worker)
                comm.send(None, dest=worker, tag=tags.EXIT)
            chunk += 1
        elif worker_tag == tags.DONE:  # If x-corr data is received from the worker --- i.e. process DONE
            data_chunk = int(data[0])  # Which data chunk received
            data = data[1:]
            if data_chunk == nchunk - 1:  # If last chunk
                results[data_chunk * len_chunk:] = data  # Save to data array
            else:
                results[data_chunk * len_chunk:data_chunk * len_chunk + len_chunk] = data  # Save to data array
            print('Saving data process #', data_chunk, ' from worker ', worker)

        elif worker_tag == tags.EXIT:  # If worker has been closed
            closed_workers += 1

    # Print results
    print(results)
    with open("results.txt", "w") as outfile:
        outfile.write("File: result\n")
        for _i, res in enumerate(results):
            outfile.write("%s: %d\n" % (filelist[_i], res))

    # Timed process ends here
    print('Time to run: ', time.time() - time_start)

elif rank > 0:  # IF WORKER
    while True:
        comm.send(None, dest=0, tag=tags.READY)  # If idle send Ready
        worker_indexes = comm.recv(source=0, status=status)  # Receive indexes for xcorr or exit tag
        worker_tag = status.Get_tag()
        if worker_tag == tags.START:  # If process to xcorr
            chunk_num = worker_indexes[0]  # index 0 is the chunk #
            chunk_len = worker_indexes[1]  # index 1 is the length of the chunk
            worker_indexes = worker_indexes[2:]
            print('Worker #', rank, ' starting process for chunk #', chunk_num)
            result_worker = np.zeros(chunk_len, dtype='object')  # Data array to return

            for ifile, index in enumerate(worker_indexes):
                ev_file = filelist[index]
                evid = os.path.split(ev_file)[1].split(".xml")[0]
                wf_file = glob.glob(os.path.join(qmdir, f"detection_{evid}", "locate", "raw_cut_waveforms", "*.m"))
                if not wf_file:
                    print(f"No waveform file for event ID {evid}")
                    continue
                elif len(wf_file) == 1:
                    wf_file = wf_file[0]
                else:
                    print(f"More than one waveform file for event ID {evid}")
                    print(wf_file)
                    continue
                print('Worker #', rank, ' working on waveform file ', wf_file)

                h5file = os.path.join(output_dir, os.path.split(wf_file)[1].split(".")[0] + ".h5")
                if os.path.exists(h5file) and overwrite:
                    print(f"ASDF file already exists. Deleting to start over.")
                    os.remove(h5file)

                with pyasdf.ASDFDataSet(h5file, mpi=False, compression="gzip-3", mode='a') as ds:
                    # Add station data
                    for f in staxmllist:
                        ds.add_stationxml(f)
                    # Adding waveforms
                    try:
                        event = obspy.read_events(ev_file)[0]
                        ds.add_quakeml(event)
                        ds.add_waveforms(wf_file, tag="raw_recording", event_id=event)
                    except Exception:
                        print("ERROR: Could open ASDF file but adding event data raised an error.")
                        pass
                result_worker[ifile] = 1

            # Now send back
            print('Sending process #', chunk_num, ' back to root from worker #', rank)
            send_back = np.zeros((chunk_len + 1))
            send_back[0] = chunk_num  # Index 0 is chunk #
            send_back[1:] = result_worker
            comm.send(send_back, dest=0, tag=tags.DONE)  # Return to master

        elif worker_tag == tags.EXIT:
            break  # Break out of while loop to exit the process, exit tag sent from master
    comm.send(None, dest=0, tag=tags.EXIT)  # Exited worker

# comm.barrier()
sys.stdout.flush()  # Flush system
# comm.Finalize()
