import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import glob
from collections import defaultdict
from scipy.signal import find_peaks

note_ranges = {
    'E3': (155, 175),
    'F3': (170, 190),
    'F#3/Gb3': (180, 200),
    'G3': (185, 210),
    'G#3/Ab3': (200, 220),
    'A3': (208, 232),
    'A#3/Bb3': (225, 245),
    'B3': (235, 260),
    'C4': (255, 275),
    'C#4/Db4': (270, 290),
    'D4': (275, 315),
    'D#4/Eb4': (305, 325),
    'E4': (310, 350)
}

def analyze_audio(filename):
    y, sr = librosa.load(filename, sr=None)
  
    n_fft = 2**14
    S = np.abs(librosa.stft(y, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    S_mean = np.mean(S, axis=1)
    
    peaks, _ = find_peaks(S_mean, height=0.05*np.max(S_mean))
    peak_freqs = freqs[peaks]
    peak_mags = S_mean[peaks]
    
    sorted_idx = np.argsort(peak_mags)[::-1]
    sorted_peaks = peak_freqs[sorted_idx]
    
    fundamental_freq = None
    for peak_freq in sorted_peaks:
        for note, (freq_min, freq_max) in note_ranges.items():
            if freq_min <= peak_freq <= freq_max:
                fundamental_freq = peak_freq
                print(f"Found {note} fundamental frequency: {fundamental_freq:.2f} Hz")
                break
        if fundamental_freq:
            break
    
    if not fundamental_freq:
        for peak_freq in sorted_peaks:
            if peak_freq < 1000:
                fundamental_freq = peak_freq
                print(f"Using first peak under 1000 Hz: {fundamental_freq:.2f} Hz")
                break
    
    if not fundamental_freq:
        fft_result = np.fft.rfft(y)
        fft_freqs = np.fft.rfftfreq(len(y), 1/sr)
        mask = fft_freqs < 1000
        fundamental_freq = fft_freqs[mask][np.argmax(np.abs(fft_result[mask]))]
    
    max_amplitude = np.max(np.abs(y))
    max_amplitude_sample = np.argmax(np.abs(y))
    
    amplitude_threshold = max_amplitude * 0.05
    
    decay_sample = None
    for i in range(max_amplitude_sample, len(y)):
        if np.abs(y[i]) < amplitude_threshold:
            if np.all(np.abs(y[i:]) < amplitude_threshold):
                decay_sample = i
                break
    
    if decay_sample is not None:
        time_to_decay = (decay_sample - max_amplitude_sample) / sr
    else:
        time_to_decay = None
        print("⚠️ Warning: Could not find a stable decay point below 5% threshold")
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(y))/sr, y)
    plt.axhline(y=amplitude_threshold, color='r', linestyle='--', label='5% Threshold')
    plt.axhline(y=-amplitude_threshold, color='r', linestyle='--')
    plt.axvline(x=max_amplitude_sample/sr, color='g', linestyle='--', label='Max Amplitude')
    if decay_sample is not None:
        plt.axvline(x=decay_sample/sr, color='b', linestyle='--', label='Decay Point (5%)')
    plt.title('Audio Waveform Analysis')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(freqs, S_mean)
    plt.axvline(x=fundamental_freq, color='r', linestyle='--', label=f'Fundamental: {fundamental_freq:.2f} Hz')

    plt.scatter(peak_freqs[sorted_idx[:5]], S_mean[peaks][sorted_idx[:5]], color='green', label='Top 5 Peaks')

    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, 1000)
    plt.legend()

    plt.tight_layout()
    
    if decay_sample is not None:
        decay_rate = (max_amplitude - amplitude_threshold) / time_to_decay
    else:
        decay_rate = None

    return {
        "Fundamental Frequency (Hz)": fundamental_freq,
        "Max Amplitude (Normalized)": max_amplitude,
        "Max Amplitude Position (s)": max_amplitude_sample / sr,
        "5% Amplitude Threshold": amplitude_threshold,
        "Decay Position (s)": decay_sample / sr if decay_sample is not None else None,
        "Time to Decay (s)": time_to_decay,
        "Decay Rate (amplitude/sec)": decay_rate
    }

def process_audio_files(directory_path, output_csv):
    audio_files = glob.glob(os.path.join(directory_path, "*.wav"))
    
    note_results = defaultdict(list)
    
    for filename in audio_files:
        print(f"Processing: {os.path.basename(filename)}")
        
        try:
            results = analyze_audio(filename)
            
            note = None
            for note_name, (freq_min, freq_max) in note_ranges.items():
                if freq_min <= results["Fundamental Frequency (Hz)"] <= freq_max:
                    note = note_name
                    break
            
            note_results[note].append({
                "Filename": os.path.basename(filename),
                "Fundamental Frequency (Hz)": results["Fundamental Frequency (Hz)"],
                "Max Amplitude (Normalized)": results["Max Amplitude (Normalized)"],
                "5% Amplitude Threshold": results["5% Amplitude Threshold"],
                "Time to Decay (s)": results["Time to Decay (s)"],
                "Decay Rate (amplitude/sec)": results["Decay Rate (amplitude/sec)"]
            })
            
            plt.savefig(f"{os.path.splitext(filename)[0]}_analysis.png")
            plt.close()
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            "Filename", 
            "Note", 
            "Fundamental Frequency (Hz)", 
            "Max Amplitude (Normalized)", 
            "5% Amplitude Threshold", 
            "Time to Decay (s)",
            "Decay Rate (amplitude/sec)"
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for note, trials in note_results.items():
            for trial in trials:
                writer.writerow({
                    "Filename": trial["Filename"],
                    "Note": note,
                    "Fundamental Frequency (Hz)": trial["Fundamental Frequency (Hz)"],
                    "Max Amplitude (Normalized)": trial["Max Amplitude (Normalized)"],
                    "5% Amplitude Threshold": trial["5% Amplitude Threshold"],
                    "Time to Decay (s)": trial["Time to Decay (s)"],
                    "Decay Rate (amplitude/sec)": trial["Decay Rate (amplitude/sec)"]
                })
    
    avg_csv = os.path.splitext(output_csv)[0] + "_averages.csv"
    with open(avg_csv, 'w', newline='') as csvfile:
        fieldnames = [
            "Note", 
            "Number of Trials",
            "Avg Fundamental Frequency (Hz)", 
            "Avg Max Amplitude (Normalized)", 
            "Avg 5% Amplitude Threshold", 
            "Avg Time to Decay (s)",
            "Avg Decay Rate (amplitude/sec)"
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for note, trials in note_results.items():
            if not note:
                continue
                
            num_trials = len(trials)
            
            if num_trials == 0:
                continue
                
            avg_freq = np.mean([t["Fundamental Frequency (Hz)"] for t in trials if t["Fundamental Frequency (Hz)"] is not None])
            avg_max_amp = np.mean([t["Max Amplitude (Normalized)"] for t in trials if t["Max Amplitude (Normalized)"] is not None])
            avg_threshold = np.mean([t["5% Amplitude Threshold"] for t in trials if t["5% Amplitude Threshold"] is not None])
            
            decay_times = [t["Time to Decay (s)"] for t in trials if t["Time to Decay (s)"] is not None]
            decay_rates = [t["Decay Rate (amplitude/sec)"] for t in trials if t["Decay Rate (amplitude/sec)"] is not None]
            
            avg_decay_time = np.mean(decay_times) if decay_times else None
            avg_decay_rate = np.mean(decay_rates) if decay_rates else None
            
            writer.writerow({
                "Note": note,
                "Number of Trials": num_trials,
                "Avg Fundamental Frequency (Hz)": avg_freq,
                "Avg Max Amplitude (Normalized)": avg_max_amp,
                "Avg 5% Amplitude Threshold": avg_threshold,
                "Avg Time to Decay (s)": avg_decay_time,
                "Avg Decay Rate (amplitude/sec)": avg_decay_rate
            })
    
    print(f"Analysis complete. Individual results saved to {output_csv}")
    print(f"Average results saved to {avg_csv}")


audio_directory = "/Users/mardia/Desktop/PHYSICS/AUDIOS/"
output_csv_file = "/Users/mardia/Desktop/PHYSICS/audio_analysis_results.csv"
process_audio_files(audio_directory, output_csv_file)
