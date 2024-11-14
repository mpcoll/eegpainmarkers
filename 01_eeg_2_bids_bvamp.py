# -*- coding: utf-8 -*-
# @Author: MP Coll
"""

Author: michel-pierre.coll
Date: 2022-07-27
Project: eegmarkers
Description: Convert raw data to BIDS format FOR NEW PARTICIPANTS WITH THE BV amp

Instructions:
- All eeg files need to be in the sub-xxx/eeg folders and the filenames should
contain the task name and the .mff extension if appropriate.
- Raw folder should be organised this way:

```
raw
├── sub-XXX
    └───eeg
    │     sub-xxx_restingstate.mff
    │     sub-xxx_thermalpassive.mff
    │     sub-xxx_thermalactive.mff
    │     sub-xxx_audioactive.mff
    │     sub-xxx_audiopassive.mff
    │     sub-xxx_disgustactive.mff
    │     sub-xxx_disgustpassive.mff
    │     sub-xxx_chemicalactive.mff
    │     sub-xxx_chemicalpassive.mff
    └───beh
    │      sub-xxx_thermalpassive.csv
    │      ... other psychopy files
    └───physio
        │   sub-xxx_thermalpassive.acq
        │   ... other acqknowledge files
├── socio.csv   (socio questionnaire)
├── iastay1.csv (IASTA y1 questionnaire)
├── iastay2.csv (IASTA y2 questionnaire)
├── bdi.csv     (BDI questionnaire)
├── pcs.csv     (PCS questionnaire)
```

"""


# NOTE !!!!!!!!!!!!!!!!!!!!!! sub-011 was included by creating fake behavioural files for thermal active.

import json
import matplotlib.pyplot as plt
import shutil
from os.path import join as opj
from scipy.signal import decimate

# import bioread
import pandas as pd
import numpy as np
import os
import mne
from scipy.signal import resample
from eegmarkers_config import global_parameters  # noqa

params = global_parameters()

source_dir = opj(params.bidspath, 'sourcedata')
bids_dir = params.bidspath

if not os.path.exists(opj(bids_dir, "derivatives", "eeg_import_figs")):
    os.makedirs(opj(bids_dir, "derivatives", "eeg_import_figs"))

# Load stim intensity file
stim = pd.read_csv(opj(source_dir, "stim_10Hz.txt"), header=None)
# Upsample to 1000 Hz to match EEG
stim = np.append(np.repeat(stim.values[1:], 100, axis=0).flatten(), 0)


"""
Dict of corresponding channels according to acticap slim montage
"""

elec_list_new = {
    "E1": "Fp1",
    "E2": "Fz",
    "E3": "F3",
    "E4": "F7",
    "E5": "FT9",
    "E6": "FC5",
    "E7": "FC1",
    "E8": "C3",
    "E9": "T7",
    "E10": "TP9",
    "E11": "CP5",
    "E12": "CP1",
    "E13": "Pz",
    "E14": "P3",
    "E15": "P7",
    "E16": "O1",
    "E17": "Oz",
    "E18": "O2",
    "E19": "P4",
    "E20": "P8",
    "E21": "TP10",
    "E22": "CP6",
    "E23": "CP2",
    "E24": "Cz",
    "E25": "C4",
    "E26": "T8",
    "E27": "FT10",
    "E28": "FC6",
    "E29": "FC2",
    "E30": "F4",
    "E31": "F8",
    "E32": "Fp2",
    "E33": "AF7",
    "E34": "AF3",
    "E35": "AFz",
    "E36": "F1",
    "E37": "F5",
    "E38": "FT7",
    "E39": "FC3",
    "E40": "C1",
    "E41": "C5",
    "E42": "TP7",
    "E43": "CP3",
    "E44": "P1",
    "E45": "P5",
    "E46": "PO7",
    "E47": "PO3",
    "E48": "POz",
    "E49": "PO4",
    "E50": "PO8",
    "E51": "P6",
    "E52": "P2",
    "E53": "CPz",
    "E54": "CP4",
    "E55": "TP8",
    "E56": "C6",
    "E57": "C2",
    "E58": "FC4",
    "E59": "FT8",
    "E60": "F6",
    "E61": "AF8",
    "E62": "AF4",
    "E63": "F2",
    "E64": "Iz",
    "VREF": "FCz",
}


def stim_json():
    return {"SamplingFrequency": 144, "StartTime": 0, "Columns": []}


def writetojson(outfile, content):
    """
    Helper to write a dictionnary to json
    """
    with open(outfile, "w") as outfile:
        json.dump(content, outfile)


all_rate_pain = []
all_rate_audio = []
# Find participants in source directory
part = [
    p
    for p in os.listdir(source_dir)
    if os.path.isdir(opj(source_dir, p)) and "sub-" in p
]
part = [p for p in part if int(p[4:]) > 54]
part.sort()

# Intialize datasets info
if not os.path.exists(opj(bids_dir, "derivatives", "datasets_info.csv")):
    datasets_info = pd.read_csv(
        opj(bids_dir, "derivatives", "datasets_info_egi.csv"), index_col=0
    )
else:
    datasets_info = pd.read_csv(
            opj(bids_dir, "derivatives", "datasets_info.csv"), index_col=0
        )
        

for p in part:
    if not os.path.exists(opj(bids_dir, p)):
        os.mkdir(opj(bids_dir, p))
        os.mkdir(opj(bids_dir, p, "eeg"))
    # EEG files
    eeg_files = [f for f in os.listdir(opj(source_dir, p, "eeg")) if ".vhdr" in f]

    for f in eeg_files:
        # Read raw data
        raw = mne.io.read_raw_brainvision(opj(source_dir, p, "eeg", f), 
                                          eog=("HEOG", "VEOG"),
                                          misc=("ECG", "GSR"),
                                          preload=True)

        # Get task name and output name
        taskname = (
            f.split("_")[1]
            .replace(".vhdr", "")
            .split(" ")[0]
            .replace("thermode", "thermal")
        )
        outname = "%s_task-%s_eeg.vhdr" % (p, taskname)

        events, event_desc = mne.events_from_annotations(raw)

        event_desc_inv = {
            y: x
            for x, y in event_desc.items()
            if "Rec_start" not in x and "New Segment/" not in x and "D/D  1" not in x
        }

        annot = mne.annotations_from_events(
            events, sfreq=raw.info["sfreq"], event_desc=event_desc_inv
        )
        onset_0 = dict(annot[0])["onset"]
        onset_end = dict(annot[-1])["onset"]

        # Crop to stim duration and make sure duration is ok
        if taskname in ["thermalactive", "thermalpassive"]:

            raw.crop(tmin=onset_0)
            raw.crop(tmax=500)
            # Remove first samp to avoid issues with annotations
            raw = mne.io.RawArray(raw.get_data(), raw.info, first_samp=0)
            duration = raw.times[-1]
            assert np.isclose(duration, 500, atol=0.5)
            assert np.isclose(onset_end - onset_0, 500, atol=1)

        if taskname in ["audioactive", "audiopassive"]:
            raw.crop(tmin=onset_0)
            raw.crop(tmax=490)
            duration = raw.times[-1]
            # Remove first samp to avoid issues with annotations
            raw = mne.io.RawArray(raw.get_data(), raw.info, first_samp=0)
            assert np.isclose(duration, 490, atol=0.5)
            assert np.isclose(onset_end - onset_0, 490, atol=1)

        if taskname in [
            "disgustactive",
            "disgustpassive",
            "chemicalactive",
            "chemicalpassive",
        ]:

            raw.crop(tmin=onset_0, tmax=onset_end)
            duration = raw.times[-1]
            # Remove first samp to avoid issues with annotations
            raw = mne.io.RawArray(raw.get_data(), raw.info, first_samp=0)
            assert np.isclose(duration, 300, atol=1)
            assert np.isclose(onset_end - onset_0, 300, atol=1)

        if "rest" in taskname:
            raw.crop(tmin=onset_0, tmax=onset_end)
            # Remove first samp to avoid issues with annotations
            raw = mne.io.RawArray(raw.get_data(), raw.info, first_samp=0)
            duration = raw.times[-1]
            if p == "sub-061":  # Stopped early
                dur = 280
            elif p == "sub-068":
                dur = 290  # Stopped early
            else:
                dur = 300
            assert np.isclose(duration, dur, atol=1)
            assert np.isclose(onset_end - onset_0, dur, atol=1)

        # Fix anotations

        # Events out
        events_frame = pd.DataFrame(
            data=dict(
                onset=[
                    dict(annot[i])["onset"] - dict(annot[0])["onset"]
                    for i in range(len(annot))
                ],
                duration=0,
                sample=[
                    int(dict(annot[i])["onset"] * raw.info["sfreq"])
                    for i in range(len(annot))
                ],
                trial_type=[dict(annot[i])["description"] for i in range(len(annot))],
                value=[event_desc[annot[i]["description"]] for i in range(len(annot))],
            )
        )

        # Add distance from previous event for sanity check
        events_frame["distance_from_previous"] = [events_frame["onset"][0]] + list(
            np.diff(events_frame["onset"])
        )

        # Save events file
        events_frame.to_csv(
            opj(bids_dir, p, "eeg", outname.replace("_eeg.vhdr", "_events.tsv")),
            sep="\t",
            index=False,
        )

        # Fix but with channel key by adding annotations to all channels
        annot_fix = mne.Annotations(
            onset=[dict(annot[i])["onset"] - onset_0 for i in range(len(annot))],
            description=[dict(annot[i])["description"] for i in range(len(annot))],
            duration=[dict(annot[i])["duration"] for i in range(len(annot))],
        )

        # Set last to trial end
        raw = raw.set_annotations(annot_fix)

        # Load temperature for thermal tasks
        if "thermal" in taskname:
            # If old thermode

            phase = taskname.split("thermal")[1]
            fil = [
                f
                for f in os.listdir(opj(source_dir, p, "beh"))
                if "protocol" in f and phase in f and "csv" in f
            ][0]
            temp_frame = pd.read_csv(opj(source_dir, p, "beh", fil), header=None)
            temp_frame.columns = [
                "time",
                "step",
                "type",
                "temp1",
                "temp2",
                "temp3",
                "temp4",
                "temp5",
            ]

            temp = np.average(
                temp_frame[["temp1", "temp2", "temp3", "temp4", "temp5"]], axis=1
            )
            temp_time = temp_frame["time"].values * 1000

            # Find triggers
            trig_pos = [
                i
                for i in range(len(temp_frame) - 1)
                if "SET_CONST_TEMP" in temp_frame["type"].values[i]
                and "SET_CONST_TEMP" in temp_frame["type"].values[i + 1]
                or "SET_CONST_TEMP" in temp_frame["type"].values[i]
                and "WAIT" in temp_frame["type"].values[i + 1]
                and "WAIT" in temp_frame["type"].values[i - 1]
            ]
            assert len(trig_pos) == 10
            trig_times = temp_time[trig_pos].astype(int)

            # Resample temp to 1000 Hz
            temp_ms = np.interp(np.arange(0, temp_time[-1]), temp_time, temp)

            temp_ms = temp_ms[trig_times[0] :]
            trig_times = trig_times - trig_times[0]

            temp_ms = temp_ms[: int(raw.times[-1] * 1000 + 1)]
            events_frame["onsets_centered_first"] = (
                events_frame["onset"] - events_frame["onset"][0]
            )
            events_frame["trigger_temp"] = list(trig_times) + [0] * (
                len(events_frame) - len(trig_times)
            )

            events_frame.to_csv(
                opj(bids_dir, p, "eeg", outname.replace("_eeg.vhdr", "_events.tsv")),
                sep="\t",
                index=False,
            )

        # Load behavioral data for active tasks
        if "active" in taskname:
            beh = pd.read_csv(
                opj(
                    source_dir,
                    p,
                    "beh",
                    [
                        f
                        for f in os.listdir(opj(source_dir, p, "beh"))
                        if taskname in f and "FullRatings.csv" in f
                    ][0],
                )
            )
            beh.drop([c for c in beh.columns if "Unnamed" in c], axis=1, inplace=True)
            beh["TaskName"] = taskname
            beh["participant_id"] = p
            beh["times_ms"] = np.round(beh["times (s)"] * 1000)
            beh = beh[1:]
            beh["times_ms"].values[-1]
            beh["times_ms"].values[0]

            if "audioactive" in taskname:
                beh = beh[beh["times_ms"] < onset_end * 1000]
            elif "thermalactive" in taskname:
                beh = beh[beh["times_ms"] < onset_end * 1000]
                if "test" in p:
                    beh = beh[beh["times_ms"] < onset_end * 1000]
            else:
                beh = beh[beh["times_ms"] < onset_end * 1000]

            ratings_ms = np.interp(
                np.arange(0, len(raw.times)),
                beh["times_ms"].values.astype(int),
                beh["rating"].values,
            )
            assert len(ratings_ms) == len(raw.times)

            beh.to_csv(
                opj(bids_dir, p, "eeg", outname.replace("_eeg.vhdr", "_stim.tsv.gz")),
                index=False,
                sep="\t",
            )
            stim_json_out = stim_json()

            writetojson(
                opj(bids_dir, p, "eeg", outname.replace("_eeg.vhdr", "_stim.json")),
                stim_json_out,
            )

        rate_info = mne.create_info(
            ["FCz", "rating", "stim", "temp"],
            raw.info["sfreq"],
            ["eeg", "misc", "misc", "misc"],
        )

        if "rest" not in taskname:

            if "active" in taskname:
                if "audio" in taskname or "thermal" in taskname:
                    # Resample ratings to 1000 Hz
                    if "thermal" in taskname:
                        stim_task = np.append(stim, np.zeros(500001 - len(stim)))
                        all_rate_pain.append(ratings_ms)
                        temp_task = np.append(temp_ms, np.zeros(500001 - len(temp_ms)))

                        if "test" in p:
                            stim_task = np.append(stim, np.zeros(490001 - len(stim)))
                            temp_task = np.append(
                                temp_ms, np.zeros(490001 - len(temp_ms))
                            )

                        prate_chan = mne.io.RawArray(
                            np.vstack(
                                [
                                    np.zeros_like(ratings_ms),
                                    ratings_ms,
                                    stim_task,
                                    temp_task,
                                ]
                            ),
                            info=rate_info,
                        )
                    else:
                        stim_task = stim
                        all_rate_audio.append(ratings_ms[:490001])

                        prate_chan = mne.io.RawArray(
                            np.vstack(
                                [
                                    np.zeros_like(ratings_ms),
                                    ratings_ms,
                                    stim_task,
                                    np.zeros_like(ratings_ms),
                                ]
                            ),
                            info=rate_info,
                        )
                else:
                    ratings_1000 = resample(beh["rating"].values, len(raw.times))
                    prate_chan = mne.io.RawArray(
                        np.vstack(
                            [
                                np.zeros_like(ratings_ms),
                                ratings_ms,
                                np.zeros_like(ratings_ms),
                                np.zeros_like(ratings_ms),
                            ]
                        ),
                        info=rate_info,
                    )
            else:
                if "audio" in taskname or "thermal" in taskname:
                    if "thermal" in taskname:
                        stim_task = np.append(stim, np.zeros(500001 - len(stim)))
                        temp_task = np.append(temp_ms, np.zeros(500001 - len(temp_ms)))

                        if "test" in p:
                            stim_task = np.append(stim, np.zeros(490001 - len(stim)))
                            temp_task = np.append(
                                temp_ms, np.zeros(490001 - len(temp_ms))
                            )

                        prate_chan = mne.io.RawArray(
                            np.vstack(
                                [
                                    np.zeros_like(raw.times),
                                    np.zeros_like(raw.times),
                                    stim_task,
                                    temp_task,
                                ]
                            ),
                            info=rate_info,
                        )
                    else:
                        stim_task = stim
                        prate_chan = mne.io.RawArray(
                            np.vstack(
                                [
                                    np.zeros_like(raw.times),
                                    np.zeros_like(raw.times),
                                    stim_task,
                                    np.zeros_like(raw.times),
                                ]
                            ),
                            info=rate_info,
                        )
                else:
                    prate_chan = mne.io.RawArray(
                        np.vstack(
                            [
                                np.zeros_like(raw.times),
                                np.zeros_like(raw.times),
                                np.zeros_like(raw.times),
                                np.zeros_like(raw.times),
                            ]
                        ),
                        info=rate_info,
                    )

        else:
            prate_chan = mne.io.RawArray(
                np.vstack(
                    [
                        np.zeros_like(raw.times),
                        np.zeros_like(raw.times),
                        np.zeros_like(raw.times),
                        np.zeros_like(raw.times),
                    ]
                ),
                info=rate_info,
            )

        raw.load_data().add_channels([prate_chan], force_update_info=True)

        if "active" in taskname:
            plt.figure()
            plt.plot(raw.times, raw.get_data()[-3, :], label='rating')
            plt.plot(raw.times, raw.get_data()[-2, :], label="stimulation")
            if "thermal" in taskname:
                temp_ms_0150 = (
                    (temp_ms - np.min(temp_ms))
                    / (np.max(temp_ms) - np.min(temp_ms))
                    * 150
                )
                temp_ms_0150 = np.append(
                    temp_ms_0150, np.zeros(500001 - len(temp_ms_0150))
                )
                plt.plot(raw.times, temp_ms_0150, label="temp, rescaled")
            count = 0
            for o in events_frame["onset"][:-1]:
                plt.axvline(
                    (o - events_frame["onset"][0]),
                    color="gray",
                    linestyle="--",
                    label="triggers" if count == 0 else None,
                    alpha=0.5,
                )
                count += 1

            plt.legend()
            plt.title(outname)
            plt.savefig(
                opj(
                    bids_dir,
                    "derivatives",
                    "eeg_import_figs",
                    outname.replace("_eeg.vhdr", "_plot.png"),
                ),
                dpi=300,
            )

        if "thermal" in taskname:

            plt.figure()
            plt.plot(temp_ms, label="temp")
            for i in range(len(trig_times)):
                plt.axvline(
                    trig_times[i],
                    color="gray",
                    linestyle="--",
                    label="temp triggers" if i == 0 else None,
                    alpha=0.5,
                )
            plt.title(outname)
            plt.legend()
            plt.savefig(
                opj(
                    bids_dir,
                    "derivatives",
                    "eeg_import_figs",
                    outname.replace("_eeg.vhdr", "__temp_plot1.png"),
                ),
                dpi=300,
            )

            plt.figure()
            plt.title(outname)
            plt.plot(raw.times, raw.get_data()[-2, :], label="stimulation")
            # Rescale temp from 0 to 150
            temp_ms_0150 = (temp_ms - 38) / (np.max(temp_ms) - 38) * 150
            temp_ms_0150 = np.append(temp_ms_0150, np.zeros(500001 - len(temp_ms_0150)))

            plt.plot(raw.times, temp_ms_0150, label="temp, rescaled")
            plt.legend()
            plt.savefig(
                opj(
                    bids_dir,
                    "derivatives",
                    "eeg_import_figs",
                    outname.replace("_eeg.vhdr", "__temp_plot2.png"),
                ),
                dpi=300,
            )

        # Reorder channels
        ch_order = [c for c in list(elec_list_new.values()) if "Iz" not in c] + [
            "HEOG",
            "VEOG",
            "ECG",
            "GSR",
            "rating",
            "stim",
            "temp",
        ]
        
        raw.reorder_channels(ch_order)
        
        # Set all non eeg channels to misc
        raw.set_channel_types(
            {
                "HEOG": "misc",
                "VEOG": "misc",
                "ECG": "misc",
                "GSR": "misc",
                "rating": "misc",
                "stim": "misc",
                "temp": "misc",
            }
        )
        

        # Make a GSR/ECG plot
        plt.figure()
        plt.title(outname)
        norm_gsr = (raw.get_data()[-4, :] - np.min(raw.get_data()[-4, :])) / (
            np.max(raw.get_data()[-4, :]) - np.min(raw.get_data()[-4, :])
        )
        plt.plot(raw.times, raw.get_data()[-4, :], label="GSR")
        plt.ylabel("GSR")
        plt.savefig(
            opj(
                bids_dir,
                "derivatives",
                "eeg_import_figs",
                outname.replace("_eeg.vhdr", "_gsr.png"),
            ),
            dpi=300,
        )
        plt.figure()
        plt.title(outname)
        norm_gsr = (raw.get_data()[-4, :] - np.min(raw.get_data()[-4, :])) / (
            np.max(raw.get_data()[-4, :]) - np.min(raw.get_data()[-4, :])
        )
        plt.plot(norm_gsr, label="Normalized GSR")
        plt.ylabel("Norm GSR")
        plt.savefig(
            opj(
                bids_dir,
                "derivatives",
                "eeg_import_figs",
                outname.replace("_eeg.vhdr", "_normgsr.png"),
            ),
            dpi=300,
        )
        plt.figure(figsize=(10, 2))
        plt.title(outname)
        plt.plot(raw.times[0:30000], raw.get_data()[-5, 0:30000], label="ECG")
        plt.ylabel("ECG")
        plt.savefig(
            opj(
                bids_dir,
                "derivatives",
                "eeg_import_figs",
                outname.replace("_eeg.vhdr", "_ecg30s.png"),
            ),
            dpi=300,
        )

        mne.export.export_raw(
            opj(bids_dir, p, "eeg", outname), raw, overwrite=True
        )
            
        datasets_info.loc[p, taskname + '_dur'] = raw.times[-1]
        datasets_info.loc[p, taskname + '_nchans_all'] = raw.info["nchan"]
        datasets_info.loc[p, taskname + '_nchans_eeg'] = len(raw.copy().pick_types(eeg=True).ch_names)
        datasets_info.loc[p, taskname + '_nchans_others'] = len(raw.copy().pick_types(eeg=False, misc=True).ch_names)
        datasets_info.loc[p, taskname + '_sfreq'] = raw.info["sfreq"]


        # Save channels file
        ch_types = ["EEG"] * 64 + ["EOG"] * 2 + ["ECG"] + ["GSR"] + ["MISC"] * 3
        chan_file = pd.DataFrame(
            {
                "name": raw.ch_names,
                "type": ch_types,
                "units": "µV",
                "reference": "FCz",
                "status": "good",
            }
        )

        chan_file.to_csv(
            opj(bids_dir, p, "eeg", outname.replace("_eeg.vhdr", "_channels.tsv")),
            index=False,
            sep="\t",
        )

        plt.close("all")
        eeg_json_out = {
            "TaskName": "",
            "TaskDescription": "",
            "Instructions": "See Psychopy script for instructions",
            "InstitutionName": "Universite Laval",
            "EEGChannelCount": 64,
            "EOGChannelCount": 2,
            "ECGChannelCount": 1,
            "GSRChannelCount": 1,
            "MISCChannelCount": 3,
            "EEGPlacementScheme": "10-10",
            "EEGReference": "FCz",
            "EEGGround": "FPz",
            "SamplingFrequency": 1000,
            "SoftwareFilters": {
                "Anti-aliasing filter": {"half-amplitude cutoff (Hz)": 500}
            },
            "PowerLineFrequency": 60,
            "Manufacturer": "BrainProducts",
            "ManufacturersModelName": "ActiChamp Plus",
            "CapManufacturer": "BrainVision",
            "CapManufacturersModelName": "actiCAP slim 64 Ch",
            "RecordingType": "continuous",
        }

        if "thermalactive" in outname:
            eeg_json_out["TaskName"] = "thermalactive"
            eeg_json_out["TaskDescription"] = (
                "Thermal stimulation on arm with continous pain ratings on VAS"
            )
        elif "thermalpassive" in outname:
            eeg_json_out["TaskName"] = "thermalpassive"
            eeg_json_out["TaskDescription"] = "Thermal stimulation, passive"
        elif "audioactive" in outname:
            eeg_json_out["TaskName"] = "audioactive"
            eeg_json_out["TaskDescription"] = (
                "Audio stimulation continous unpleasantness ratings on VAS"
            )
        elif "audiopassive" in outname:
            eeg_json_out["TaskName"] = "audiopassive"
            eeg_json_out["TaskDescription"] = "Audio stimulation continous, passive"
        elif "restingstate" in outname:
            eeg_json_out["TaskName"] = "rest"
            eeg_json_out["TaskDescription"] = "Resting state"
        elif "chemicalactive" in outname:
            eeg_json_out["TaskName"] = "chemicalactive"
            eeg_json_out["TaskDescription"] = (
                "Hot sauce on tongue with continous pain ratings on VAS"
            )
        elif "chemicalpassive" in outname:
            eeg_json_out["TaskName"] = "chemicalpassive"
            eeg_json_out["TaskDescription"] = "Hot sauce on tongue, passive"
        elif "disgustpassive" in outname:
            eeg_json_out["TaskName"] = "disgustpassive"
            eeg_json_out["TaskDescription"] = "Karela on tongue, passive"
        elif "disgustactive" in outname:
            eeg_json_out["TaskName"] = "disgustactive"
            eeg_json_out["TaskDescription"] = (
                "Karela on tongue with continous unpleasantness ratings on VAS"
            )
        elif "rest" in outname:
            eeg_json_out["TaskName"] = "resting"
            eeg_json_out["TaskDescription"] = "resting"

        writetojson(
            opj(bids_dir, p, "eeg", outname.replace(".vhdr", ".json")), eeg_json_out
        )

datasets_info.to_csv(opj(bids_dir, "derivatives", "datasets_info.csv"))
# Redo participants.tsv for all participants
part = [
    p
    for p in os.listdir(source_dir)
    if os.path.isdir(opj(source_dir, p)) and "sub-" in p
]
part = [p for p in part]
part.sort()


# Questionnaires
iastay2 = pd.read_csv(opj(source_dir, "iasta_y2.csv"))
iastay2.index = iastay2["#Participant"]

iastay1 = pd.read_csv(opj(source_dir, "iasta_y1.csv"))
iastay1.index = iastay1["#Participant"]

socio = pd.read_csv(opj(source_dir, "socio.csv"))
socio.index = socio["  # Participant  "]

pcs = pd.read_csv(opj(source_dir, "pcs.csv"))
bdi = pd.read_csv(opj(source_dir, "bdi.csv"))
bdi.index = bdi["#Participant"]
pcs.index = pcs[pcs.columns[1]]


part_frame = pd.DataFrame(
    columns=["participant_id", "age", "sex", "gender", "handedness", "test_date"]
)
part_frame["participant_id"] = part
part_frame.index = part

for p in part:

    if int(p[4:]) < 22:
        part_frame.loc[p, "thermode"] = "TSA"
    else:
        part_frame.loc[p, "thermode"] = "TCS"

    if int(p[4:]) < 55:
        part_frame.loc[p, "eeg_system"] = "EGI - NA300"
        part_frame.loc[p, "physio_available"] = "No"
    else:
        part_frame.loc[p, "eeg_system"] = "BrainProducts - Actichamp Plus"
        part_frame.loc[p, "physio_available"] = "Yes"

    part_frame.loc[p, "age"] = int(socio.loc[p, "Quel est votre âge en années? "])
    if str(socio.loc[p, "Quel sexe vous a été assigné à la naissance? "]) == "Masculin":
        part_frame.loc[p, "sex"] = "m"
        part_frame.loc[p, "ismale"] = 1
    else:
        part_frame.loc[p, "sex"] = "f"
        part_frame.loc[p, "ismale"] = 0

    part_frame.loc[p, "test_date"] = socio.loc[
        p, "Date de l'expérimentation (veuillez l'inscrire sous la forme jj-mm-aaaa)"
    ]
    part_frame.loc[p, "gender"] = socio.loc[p, "Quel est votre genre? "]

    part_frame.loc[p, "handedness"] = socio.loc[p, "Quelle est votre main dominante? "]

    # Add all other data to the frame
    for c in range(len(socio.columns)):
        part_frame.loc[p, "qsocio_" + list(socio.columns)[c]] = socio.loc[
            p, list(socio.columns)[c]
        ]

    for c in range(2, len(bdi.columns)):
        part_frame.loc[p, "qbdi_" + str(c - 1)] = int(
            str(bdi.loc[p, list(bdi.columns)[c]])[0]
        )

    for c in range(2, len(iastay2.columns)):
        try:
            part_frame.loc[p, "qiastay2_" + list(iastay2.columns)[c]] = int(
                str(iastay2.loc[p, list(iastay2.columns)[c]])[0]
            )
        except:
            part_frame.loc[p, "qiastay2_" + list(iastay2.columns)[c]] = "nan"

    for c in range(2, len(iastay2.columns)):
        try:
            part_frame.loc[p, "qiastay1_" + list(iastay1.columns)[c]] = int(
                str(iastay1.loc[p, list(iastay1.columns)[c]])[0]
            )
        except:
            part_frame.loc[p, "qiastay1_" + list(iastay1.columns)[c]] = "nan"

    for c in range(2, len(pcs.columns)):
        try:
            part_frame.loc[p, "qpcs_" + list(pcs.columns)[c]] = int(
                str(pcs.loc[p, list(pcs.columns)[c]])[0]
            )
        except:
            part_frame.loc[p, "qpcs_" + list(pcs.columns)[c]] = "nan"


journal = pd.read_csv(opj(source_dir, "journal.csv"), sep=",")
journal.index = journal["Participant"]
for p in journal.index:
    for c in range(1, len(journal.columns)):
        print(c)
        part_frame.loc[p, "participant_id"] = p
        part_frame.loc[p, "journal_" + journal.columns[c]] = journal.loc[
            p, journal.columns[c]
        ]





# Add datasets info to part_frame
for p in part:
    for c in datasets_info.columns:
        part_frame.loc[p, c] = datasets_info.loc[p, c]


part_frame.sort_values(by="participant_id", inplace=True)
part_frame.to_csv(
    opj(bids_dir, "participants.tsv"), sep="\t", index=False, encoding="utf-8"
)



# Data set level file
dataset_description = {
    "Name": "EEG pain markers",
    "BIDSVersion": "1.9.0",
    "Authors": ["Champagne, C", "Coll, MP", "Coll LAB"],
    "EthicsApprovals": ["CIUSS-CN 2023-2487"],
    "License": "CC-BY-4.0",
}
writetojson(opj(bids_dir, "dataset_description.json"), dataset_description)
