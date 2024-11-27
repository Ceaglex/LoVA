from moviepy.editor import VideoFileClip
from PIL import Image
import glob
import os
import torch
import pickle
from PIL import Image
import clip
from tqdm import tqdm
from accelerate import Accelerator


import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self,
                 mp4_dir, 
                 pickle_dir,
                 preprocess,
                 duration = 10,
                 recalculate = True,
                 fps = 22):
        self.fps = fps
        self.duration = duration
        self.pickle_dir = pickle_dir
        self.recalculate = recalculate
        self.mp4_files = glob.glob(os.path.join(mp4_dir, '*.mp4'))    ######
        self.preprocess = preprocess

    def __len__(self):
        return len(self.mp4_files)

    def __getitem__(self, idx):
        mp4_file = self.mp4_files[idx]
        try:
            pickle_file = os.path.join(self.pickle_dir, os.path.basename(mp4_file)[:-4] + ".pickle")

            if self.recalculate or not os.path.exists(pickle_file):
                video = VideoFileClip(mp4_file)
                
                video = video.set_fps(self.fps)
                frames = []
                with torch.no_grad():
                    for idx, frame in enumerate(video.iter_frames()):
                        if idx > (self.duration+0.5)*self.fps:
                            print(f"Duration: {video.duration}  Frame Idx:{idx},  {mp4_file} is too long to encode by CLIP! ")
                            video.duration = (self.duration+0.5)
                            break
                        frame = Image.fromarray(frame)
                        frame = self.preprocess(frame).unsqueeze(0)
                        frames.append(frame)
                
                feature_dict = {'video_path': mp4_file, 'pickle_path':pickle_file, 'fps' : video.fps, 'duration' : video.duration, 'frame_num':len(frames)}
                frames = torch.cat(frames)  

                # print(f"{feature_dict['duration']}, {frames.shape}, {frames.shape[0]/feature_dict['duration']}")
                return feature_dict, frames
            
            else:
                print(f"{pickle_file} already exists!")   
                return {}, torch.tensor(0)

        except Exception as e:
            print(f"{mp4_file} encountered with error : ", e)
            with open('log.txt', 'a') as file:
                file.write(f"FILE:{mp4_file}\n REASON:{e}\n\n")
            return {}, torch.tensor(0)

    



def main(mp4_dir,
        pickle_dir,
        duration = 10, 
        frame_size = 20,
        fps = 22, 
        device = 0,
        recalculate = True):
    
    accelerator = Accelerator()
    device = accelerator.device

    with torch.no_grad():
        model, preprocess = clip.load("ViT-L/14", device=device, download_root='../../weight/CLIP')
        dataset = AudioDataset(mp4_dir=mp4_dir, preprocess=preprocess, duration = duration, pickle_dir=pickle_dir, fps = fps, recalculate = recalculate)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle=False)

        model, dataloader = accelerator.prepare(model, dataloader)
        print(len(dataloader))
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        
        for feature_dict, frames in tqdm(dataloader, desc = f'GPU {device}'):
            # dataloader frames [bs, frame_num, 3, 224, 224]
            # feature_dict[key] = [value]

            try:
                if len(feature_dict) == 0:
                    continue

                for key, value in feature_dict.items():
                    feature_dict[key] = float(value[0]) if isinstance(value[0], torch.Tensor) else value[0]

                frames = frames.squeeze(0).to(device)
                image_features = []
                for i in range(len(frames)//frame_size + 1):
                    start_id = i*frame_size
                    end_id = (i+1)*frame_size
                    if start_id >= len(frames):
                        break
                    image_features.append(model.encode_image(frames[start_id:end_id]))
                

                image_features = torch.concat(image_features)
                feature_dict['feature'] = image_features

                with open(feature_dict['pickle_path'], 'wb') as handle:
                    pickle.dump(feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            except Exception as e:
                print("Extracting feature fails because of error : ", e)
                with open('log.txt', 'a') as file:
                    file.write(f"FILE:{feature_dict['video_path']}\n REASON:{e}\n\n")
                        



if __name__ == "__main__":
    train_test = 'train'
    dataset = 'unav100'
    duration = 10
    mp4_dir = f'../../../{dataset}/dataset/{train_test}'
    pickle_dir = f'../../dataset/feature/{train_test}/{dataset}/{duration}'
    # mp4_dir = f'../../../{dataset}/dataset/{train_test}/{duration}'
    # pickle_dir = f'../../dataset/feature/{train_test}/{dataset}/{duration}'

    frame_size = 20
    main(mp4_dir = mp4_dir, pickle_dir = pickle_dir, duration = duration, frame_size = frame_size)

