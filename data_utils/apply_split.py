import os
from tqdm import tqdm
import pickle
import shutil

speakers = ['seth', 'oliver', 'conan', 'chemistry']
source_data_root = "../expressive_body-V0.7"
data_root = "D:/Downloads/SHOW_dataset_v1.0/ExpressiveWholeBodyDatasetReleaseV1.0"

f_read = open('split_more_than_2s.pkl', 'rb')
f_save = open('none.pkl', 'wb')
data_split = pickle.load(f_read)
none_split = []

train = val = test = 0

for speaker_name in speakers:
    speaker_root = os.path.join(data_root, speaker_name)

    videos = [v for v in data_split[speaker_name]]

    for vid in tqdm(videos, desc="Processing training data of {}......".format(speaker_name)):
        for split in data_split[speaker_name][vid]:
            for seq in data_split[speaker_name][vid][split]:

                seq = seq.replace('\\', '/')
                old_file_path = os.path.join(data_root, speaker_name, vid, seq.split('/')[-1])
                old_file_path = old_file_path.replace('\\', '/')
                new_file_path = seq.replace(source_data_root.split('/')[-1], data_root.split('/')[-1])
                try:
                    shutil.move(old_file_path, new_file_path)
                    if split == 'train':
                        train = train + 1
                    elif split == 'test':
                        test = test + 1
                    elif split == 'val':
                        val = val + 1
                except FileNotFoundError:
                    none_split.append(old_file_path)
                    print(f"The file {old_file_path} does not exists.")
                except shutil.Error:
                    none_split.append(old_file_path)
                    print(f"The file {old_file_path} does not exists.")

print(none_split.__len__())
pickle.dump(none_split, f_save)
f_save.close()

print(train, val, test)


