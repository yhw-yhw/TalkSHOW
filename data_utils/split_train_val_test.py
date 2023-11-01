import os 
import json 
import shutil

if __name__ =='__main__':
    id_list = "chemistry conan oliver seth"
    id_list = id_list.split(' ')

    old_root = '/home/usename/talkshow_data/ExpressiveWholeBodyDatasetReleaseV1.0'
    new_root = '/home/usename/talkshow_data/ExpressiveWholeBodyDatasetReleaseV1.0/talkshow_data_splited'

    with open('train_val_test.json') as f:
        split_info = json.load(f)
    phase_list = ['train', 'val', 'test']
    for phase in phase_list:
        phase_path_list = split_info[phase]
        for p in phase_path_list:
            old_path = os.path.join(old_root, p)
            if not os.path.exists(old_path):
                print(f'{old_path} not found, continue' )
                continue
            new_path = os.path.join(new_root, phase, p)
            dir_name = os.path.dirname(new_path)
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            shutil.move(old_path, new_path)

