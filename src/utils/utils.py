import os
import random
import numpy as np
import librosa
import torch
import soundfile as sf
import matplotlib.pyplot as plt



def create_dir(dir: str) -> None:
    if os.path.isdir(dir):
        print(dir, 'already exists ...')
    else:
        print('Making a new directory: ', dir)
        os.makedirs(dir)

def flac2wav(flac_path:str, wav_path:str) -> None:
    """Convert flac to wav file"""
    
    flac_files = [
        f for f in os.listdir(flac_path)
        if os.path.isfile(os.path.join(flac_path, f)) and f.endswith('.flac')
        ]

    for file in flac_files:

        print(f'Converting {file}')

        flac_file_path = os.path.join(flac_path, file)
        wav_file_name = os.path.splitext(file)[0] + '.wav'
        wav_file_path = os.path.join(wav_path, wav_file_name)
        
        flac_audio, samplerate = sf.read(flac_file_path)
        
        sf.write(wav_file_path, flac_audio, samplerate)
        
        
    print('Done converting all FLAC files to WAV.\n')

def get_melss(wav_file: str, output_path: str) -> None:
    """Generate and save a mel spectrogram image from a WAV file."""
    # Load audio
    x, sr = librosa.load(wav_file, sr=None, res_type='kaiser_fast')

    # Create a small, axis-free plot
    fig = plt.figure(figsize=[1, 1])
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Compute mel spectrogram
    melss = librosa.feature.melspectrogram(y=x, sr=sr)
    db_melss = librosa.power_to_db(melss, ref=np.max)

    # Plot and save
    librosa.display.specshow(db_melss, sr=sr, y_axis='linear', x_axis='time')
    plt.savefig(output_path, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()

def create_melspec_dataset(wav_path:str, mel_path:str):
    """ Create the Mel Spectogram Images """

    wav_files = [
        f for f in os.listdir(wav_path)
        if os.path.isfile(os.path.join(wav_path, f)) and f.endswith('.wav')
        ]
    
    for file in wav_files:
        print(f"Getting Mel Spectogram of {file}")

        wav_file = os.path.join(wav_path, file)
        output_file = os.path.join(mel_path, os.path.splitext(file)[0] + '.jpg')
        print(output_file)
        get_melss(wav_file, output_file)

    print("All mel spectrograms generated.")

def five_sec_chunks(wav_path:str, root_chunk_path:str, chunk_duration: float = 5.0) -> None: 

    wav_files = [
        f for f in os.listdir(wav_path)
        if os.path.isfile(os.path.join(wav_path, f)) and f.endswith('.wav')
        ]
    
    for file in wav_files:
        print(f"Clipping {file}")

        wav_file_path = os.path.join(wav_path, file)
        
        # Load the audio file
        audio_data, sample_rate = sf.read(wav_file_path)
        total_samples = len(audio_data)
        chunk_samples = int(chunk_duration * sample_rate)

        
        # Calculate number of full chunks
        num_full_chunks = total_samples // chunk_samples

        base_name = os.path.splitext(os.path.basename(file))[0]

       

        for i in range(num_full_chunks):
            start_sample = i * chunk_samples
            end_sample = start_sample + chunk_samples
            chunk_data = audio_data[start_sample:end_sample]

            chunk_filename = f"{base_name}_part{i+1}.wav"
            
            chunk_path = os.path.join(root_chunk_path, chunk_filename)

            
            sf.write(chunk_path, chunk_data, sample_rate)
            print(f"Saved: {chunk_path}")

            

    print("Done splitting audio. Discarded last chunk if shorter than {chunk_duration} seconds.")



    






