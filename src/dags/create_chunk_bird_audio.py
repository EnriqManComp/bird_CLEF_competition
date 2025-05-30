"""
    This script make the following actions:        
        - Splitting ogg files in 5 second chunk files.
        - Create the chunk metadata
"""

from src.utils.utils import five_sec_chunks
import os
import sqlalchemy


def main():

    root_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

    train_data_path = str(root_dir + f'/bird_CLEF_competition/data/birdCLEFDataset/birdclef-2025/train_audio/')
    chunk_path = str(root_dir + f'/bird_CLEF_competition/data/birdCLEFDataset/birdclef-2025-chunks/train_audio/')

    for dirpath, dirnames, _ in os.walk(train_data_path):
        
        for dirname in dirnames:
            print(f"  Subfolder: {dirname}")
            subfolder_train_data = str(os.path.join(dirpath, dirname))

            os.makedirs(os.path.join(chunk_path, dirname), exist_ok=True)

            five_sec_chunks(
                audio_path=subfolder_train_data,
                root_chunk_path=str(os.path.join(chunk_path, dirname)),
                ext='ogg'
                )
        

if __name__ == '__main__':
    main()