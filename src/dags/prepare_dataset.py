
"""
    This script make the following actions:
        - Converting flac files to wav files.
        - Splitting wav files in 5 second chunk files.
        - Getting the Mel Spectogram for each chunk files.
"""

from src.utils.utils import flac2wav, five_sec_chunks, create_melspec_dataset
import os
import argparse

def main(mode:str, label:str):

    root_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

    flac_path = str(root_dir + f'/bird_CLEF_competition/data/{mode}_audio/{label}/flac/')
    wav_path = str(root_dir + f'/bird_CLEF_competition/data/{mode}_audio/{label}/wav/')
    mel_path = str(root_dir + f'/bird_CLEF_competition/data/{mode}_images/{label}/')
    chunk_path = str(root_dir + f'/bird_CLEF_competition/data/{mode}_audio/{label}/wav_chunks/')

    if label == 'voice':
        flac2wav(flac_path, wav_path)

    five_sec_chunks(wav_path, chunk_path)
    create_melspec_dataset(wav_path, mel_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process audio data for training or testing")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True,
                        help="Mode to run the script: train or test")
    parser.add_argument('--label', type=str, choices=['voice', 'no_voice'], required=True,
                        help="Label to run the script: voice or no_voice")
    args = parser.parse_args()
    main(args.mode, args.label)