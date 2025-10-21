# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 11:20:47 2025

@author: alaguillog
"""

import PySimpleGUI as sg
import configparser
import subprocess
import threading
import re
import os
import signal
import json
import sys

# User preferences file
SETTINGS_FILE = "ReFrag_GUI.json"

def load_settings():
    """Load saved user settings"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_settings(values):
    """Save current user inputs to JSON"""
    try:
        data = {k: v for k, v in values.items() if isinstance(v, (str, int, bool))}
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def load_ini(file_path):
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read(file_path)
    return config

def ini_to_dict(config):
    """Build dict from configparser for showing in the GUI"""
    data = {}
    for section in config.sections():
        for key, value in config.items(section):
            data[f"{section}.{key}"] = value
    return data

def dict_to_ini(data, file_path):
    """Build ini from dict for saving to file"""
    config = configparser.ConfigParser()
    for composite_key, value in data.items():
        section, key = composite_key.split(".", 1)
        if section not in config:
            config[section] = {}
        config[section][key] = value
    with open(file_path, "w") as f:
        config.write(f)

def run_script(values, window):
    """Build command and run ReFrag"""
    cmd = [sys.executable, "ReFrag.py"]

    # Required arguments
    cmd += ["-i", values["-INFILE-"]]
    cmd += ["-r", values["-RAWFILE-"]]
    cmd += ["-d", values["-DMFILE-"]]

    # Optional ones
    if values["-DIA-"]:
        cmd += ["-a", values["-DIA-"]]
    # if values["-SCANRANGE-"]:
    #     cmd += ["-s", values["-SCANRANGE-"]]
    scan_start = values.get("-SCAN_START-", "")
    scan_end = values.get("-SCAN_END-", "")
    scan_range = ""
    if scan_start and scan_end:
        scan_range = f"{scan_start},{scan_end}"  # matches your script's expected format
    elif scan_start:
        scan_range = scan_start
    elif scan_end:
        scan_range = scan_end
    if scan_range:
        cmd += ["-s", scan_range]
    if values["-OUTDIR-"]:
        cmd += ["-o", values["-OUTDIR-"]]
    if values["-CONFIG-"]:
        cmd += ["-c", values["-CONFIG-"]]
    if values["-WORKERS-"]:
        cmd += ["-w", str(values["-WORKERS-"])]
    if values["-VERBOSE-"]:
        cmd += ["-v"]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    window.write_event_value('-PROCESS-', process)

    for line in iter(process.stdout.readline, ''):
        if '%|' in line:
            # tqdm line
            window.write_event_value('-UPDATE-', line)
        else:
            # normal line
            window.write_event_value('-APPEND-', line)

    process.wait()
    window.write_event_value('-DONE-', process.returncode)


# Load saved settings
settings = load_settings()

# GUI layout
italic = (sg.DEFAULT_FONT[0], sg.DEFAULT_FONT[1], "italic")
bold = (sg.DEFAULT_FONT[0], sg.DEFAULT_FONT[1], "bold")
amino_acids = [
    ("Alanine", "A"),
    ("Arginine", "R"),
    ("Asparagine", "N"),
    ("Aspartic Acid", "D"),
    ("Cysteine", "C"),
    ("Glutamic Acid", "E"),
    ("Glutamine", "Q"),
    ("Glycine", "G"),
    ("Histidine", "H"),
    ("Isoleucine", "I"),
    ("Leucine", "L"),
    ("Lysine", "K"),
    ("Methionine", "M"),
    ("Phenylalanine", "F"),
    ("Proline", "P"),
    ("Serine", "S"),
    ("Threonine", "T"),
    ("Selenocysteine", "U"),
    ("Tryptophan", "W"),
    ("Tyrosine", "Y"),
    ("Valine", "V"),
    ("Pyrrolysine", "O"),
    ("Ambiguous E/Q", "Z")
] # TODO: Handle adding custom amino acids
aa_rows = []
for name, code in amino_acids:
    aa_rows.append([
        sg.Text(f"{name} ({code})", size=(25,1)),
        sg.Input(key=f"-{code}_MASS-", size=(20,1), disabled=True),
        sg.Input(key=f"-{code}_FM-", size=(20,1))
    ])
    
iniedit_layout = [
    [sg.Text("INI file"), sg.Input(settings.get("-CONFIG-", ""), key="-CONFIG-"), sg.FileBrowse(), sg.Button("Load INI")],
    [sg.Text("Hover over the name of each parameter to show a brief description.", font=italic)],
    [sg.Column([
    [sg.Text("\nSEARCH PARAMETERS", font=bold)],
    [sg.Text("Batch Size", size=(25,1), tooltip="Size (number of PSMs) of each task that will be submitted to a CPU core."), sg.Input(key="-BATCH_SIZE-", size=(40,1))],
    [sg.Text("Fragment Tolerance (ppm)", size=(25,1)), sg.Input(key="-FRAGMENT_TOLERANCE-", size=(40,1))],
    [sg.Text("Theoretical Δmass Tolerance (Da)", size=(25,1)), sg.Input(key="-DELTAMASS_TOLERANCE-", size=(40,1))],
    [sg.Text("Score Mode", size=(25,1)), sg.Radio("MOD-Hyperscore", "MODE_GROUP", key="-MODE_A-", default=True), sg.Radio("HYB-Hyperscore", "MODE_GROUP", key="-MODE_B-")],
    [sg.Text("Y-series Matching", size=(25,1)), sg.Radio("Exclude y\u00b9", "Y_GROUP", key="-Y_A-", default=True), sg.Radio("Full Series", "Y_GROUP", key="-Y_B-")],
    [sg.Text("Δmass Preference", size=(25,1)), sg.Radio("Experimental", "PREF_GROUP", key="-PREF_A-", default=True), sg.Radio("Theoretical", "PREF_GROUP", key="-PREF_B-")],
    [sg.Text("\nSPECTRUM PROCESSING", font=bold)],
    [sg.Text("Top N", size=(25,1)), sg.Input(key="-FRAGMENT_TOLERANCE-", size=(40,1))],
    [sg.Text("Minimum Intensity Ratio", size=(25,1)), sg.Input(key="-FRAGMENT_TOLERANCE-", size=(40,1))],
    [sg.Text("Bin Top N", size=(25,1)), sg.Input(key="-FRAGMENT_TOLERANCE-", size=(40,1))],
    [sg.Text("Minimum fragment m/z", size=(25,1)), sg.Input(key="-FRAGMENT_TOLERANCE-", size=(40,1))],
    [sg.Text("Maximum fragment m/z", size=(25,1)), sg.Input(key="-FRAGMENT_TOLERANCE-", size=(40,1))],
    [sg.Text("Deisotope", size=(25,1)), sg.Input(key="-FRAGMENT_TOLERANCE-", size=(40,1))],
    [sg.Text("\nSUMMARY PARAMETERS", font=bold)],
    [sg.Text("Protein Column", size=(25,1)), sg.Input(key="-FRAGMENT_TOLERANCE-", size=(40,1))],
    [sg.Text("Decoy Prefix", size=(25,1)), sg.Input(key="-FRAGMENT_TOLERANCE-", size=(40,1))],
    #[sg.Column([], scrollable=True, vertical_scroll_only=True, size=(600,300), key="-INI_INPUTS-")],
    [sg.Text("\nAMINO ACIDS", font=bold)],
    [sg.Text("Adding custom amino acids through the GUI is currently unsupported. It can still be done by editing the INI file directly.", font=italic)],
    [sg.Text("", size=(25,1)), sg.Text("Amino Acid Mass", size=(20,1)), sg.Text("Fixed Modifications", size=(20,1))],
    *aa_rows,
    [sg.Text("\nOTHER MASSES", font=bold)],
    [sg.Text("Proton", size=(25,1)), sg.Input(key="-PROTON_MASS-", size=(20,1), disabled=True)],
    [sg.Text("Hydrogen", size=(25,1)), sg.Input(key="-HYDROGEN_MASS-", size=(20,1), disabled=True)],
    [sg.Text("Oxygen", size=(25,1)), sg.Input(key="-OXYGEN_MASS-", size=(20,1), disabled=True)],
    [sg.Text("\nLOGGING", font=bold)],
    [sg.Text("Create Log", size=(25,1)), sg.Checkbox("", default=settings.get("-CREATE_LOG-", True), key="-CREATE_LOG-")],
    [sg.Text("Create INI", size=(25,1)), sg.Checkbox("", default=settings.get("-CREATE_INI-", True), key="-CREATE_INI-")],
    [sg.Text("\nDEBUG", font=bold)],
    [sg.Text("Debug Scores", size=(25,1)), sg.Checkbox("", default=settings.get("-DEBUG_SCORES-", False), key="-DEBUG_SCORES-")],
    ], scrollable=True, vertical_scroll_only=True, size=(750,600))],
    [sg.Button("Save INI")]
]

run_layout = [
    [sg.Text("MSFragger results file (-i)", size=(25,1), justification='right'), sg.Input(settings.get("-INFILE-", ""), key="-INFILE-", size=(60,1)), sg.FileBrowse()],
    [sg.Text("MS Data file (-r)", size=(25,1), justification='right'), sg.Input(settings.get("-RAWFILE-", ""), key="-RAWFILE-", size=(60,1)), sg.FileBrowse()],
    [sg.Text("DeltaMass file (-d)", size=(25,1), justification='right'), sg.Input(settings.get("-DMFILE-", ""), key="-DMFILE-", size=(60,1)), sg.FileBrowse()],
    [sg.Text("_chN files (-a, comma-separated)", size=(25,1), justification='right'), sg.Input(settings.get("-DIA-", ""), key="-DIA-", size=(60,1))],
    #[sg.Text("Scan range (-s, comma-separated)", size=(25,1), justification='right'), sg.Input(settings.get("-SCANRANGE-", ""), key="-SCANRANGE-", size=(60,1))],
    #[sg.Text("Scan range (-s, comma-separated)", size=(25,1), justification='right'), sg.Input(settings.get("-SCAN_START-", ""), key="-SCAN_START-", size=(10,1), enable_events=True), sg.Text("-", pad=(0,0)), sg.Input(settings.get("-SCAN_END-", ""), key="-SCAN_END-", size=(10,1), enable_events=True)],
    [sg.Text("Scan range (-s)", size=(25,1), justification='right'), sg.Spin([i for i in range(0, 1000000)], initial_value=int(settings.get("-SCAN_START-", 0)), key="-SCAN_START-", enable_events=True, size=(8,1)), sg.Text("-", pad=(0,0)), sg.Spin([i for i in range(0, 1000000)], initial_value=int(settings.get("-SCAN_END-", 0)), key="-SCAN_END-", enable_events=True, size=(8,1))],
    [sg.Text("Output directory (-o)", size=(25,1), justification='right'), sg.Input(settings.get("-OUTDIR-", ""), key="-OUTDIR-", size=(60,1)), sg.FolderBrowse()],
    [sg.Text("Config file (-c)", size=(25,1), justification='right'), sg.Input(settings.get("-CONFIG-", ""), key="-CONFIG-", size=(60,1)), sg.FileBrowse(), sg.Button("Load Config")],
    #[sg.Text("Number of workers (-w)", size=(25,1), justification='right'), sg.Input(settings.get("-WORKERS-", ""), key="-WORKERS-", size=(60,1))],
    [sg.Text("Number of workers (-w)", size=(25,1), justification='right'), sg.Spin([i for i in range(0, os.cpu_count()+1)], initial_value=os.cpu_count(), key="-WORKERS-", size=(8,1))],
    [sg.Text("", size=(25,1)), sg.Checkbox("Verbose (-v)", default=settings.get("-VERBOSE-", False), key="-VERBOSE-")],
    #[sg.ProgressBar(100, orientation='h', size=(50, 20), key='-PROGRESS_BAR-')],
    [sg.Multiline(size=(90, 25), key='-OUTPUT-', autoscroll=True, write_only=True, font=('Courier', 10))],
    [sg.Button("Run", bind_return_key=True), sg.Button("Stop", disabled=True), sg.Button("Exit")]
]
layout = [
    [sg.Text("ReFrag v1.0", font=(sg.DEFAULT_FONT[0], sg.DEFAULT_FONT[1]*2, "bold"))], # TODO get version from script
    [sg.TabGroup([
        [sg.Tab('INI Editor', iniedit_layout), sg.Tab('Run ReFrag', run_layout)]
    ])],
    [sg.Button("Exit")]
]
window = sg.Window("ReFrag GUI", layout)

buffer = []
process = None

# Event loop
while True:
    event, values = window.read()
    if event in (sg.WINDOW_CLOSED, "Exit"):
        if process and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
        # Save settings before closing
        #save_settings(values)
        break

    elif event == "Run":
        if process and process.poll() is None:
            sg.popup_error("A process is already running.")
            continue

        # Save settings when running
        save_settings(values)

        window["-OUTPUT-"].update("")
        # window["-PROGRESS_BAR-"].update(0)
        buffer.clear()
        window["Run"].update(disabled=True)
        window["Stop"].update(disabled=False)

        threading.Thread(target=run_script, args=(values, window), daemon=True).start()
        
    elif event == "Load INI":
        ini_path = values["-CONFIG-"]
        if ini_path:
            config = configparser.ConfigParser(inline_comment_prefixes='#')
            config.read(ini_path)
            window["-FRAGMENT_TOLERANCE-"].update(float(config._sections['Search']['f_tol']))

    elif event == "-PROCESS-":
        process = values[event]

    elif event == "-APPEND-":
        buffer.append(values[event])
        window["-OUTPUT-"].update(''.join(buffer))

    elif event == "-UPDATE-":
        if buffer:
            buffer[-1] = values[event]
        else:
            buffer.append(values[event])
        window["-OUTPUT-"].update(''.join(buffer))

    # elif event == "-PROGRESS-":
    #     window["-PROGRESS_BAR-"].update(values[event])
    #     window.refresh()

    elif event == "Stop":
        if process and process.poll() is None:
            try:
                if os.name == 'nt':
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except Exception:
                try:
                    process.terminate()
                except Exception:
                    pass
            sg.popup("ReFrag stopped.")
        window["Stop"].update(disabled=True)
        window["Run"].update(disabled=False)

    elif event == "-DONE-":
        code = values[event]
        if code == 0:
            sg.popup("ReFrag finished successfully!")
        else:
            sg.popup("ReFrag finished with an error or was stopped.")
        window["Run"].update(disabled=False)
        window["Stop"].update(disabled=True)
        process = None
        
window.close()
