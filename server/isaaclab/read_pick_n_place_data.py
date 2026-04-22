import h5py

file_path = "/home/hongchix/main/data/aug_0901/type_aug_03_pose_aug_all_hdf_dataset.hdf5"
with h5py.File(file_path, "r") as f:
    print("Top-level keys:", list(f.keys()))
    print()

    data_group = f['data']

    print("Data group attributes:")
    for attr_name in data_group.attrs:
        attr_value = data_group.attrs[attr_name]
        print(f"  {attr_name}: {attr_value}")
    print()

    episode_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')], 
                            key=lambda x: int(x.split('_')[1]))
    
    num_episodes = len(episode_keys)
    print(f"Total number of episodes: {num_episodes}")
    print()