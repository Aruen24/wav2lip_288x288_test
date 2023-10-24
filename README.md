# This is a 288x288 wav2lip model version.
The original repo: https://github.com/Rudrabha/Wav2Lip
Some Features I will implement here
- [x] input size 288x288
- [x] PRelu
- [x] LeakyRelu
- [x] Gradient penalty
- [x] Wasserstein Loss
- [] wav2lip_384
- [] wav2lip_512
- [] syncnet_192
- [] syncnet_384
- [] 2TUnet instead of simple unet in wav2lip original: https://arxiv.org/abs/2210.15374
- [] MSG-UNet: https://github.com/laxmaniron/MSG-U-Net
- [] SAM-UNet: https://github.com/1343744768/Multiattention-UNet
<br />
I trained my own model on AVSPEECH dataset and then transfer learning with my private dataset. 

##说明
shell```conda create -n wav2lip python==3.7
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple


librosa==0.7.0
numpy==1.17.1
opencv-contrib-python>=4.2.0.34
opencv-python==4.1.0.25
torch==1.1.0
torchvision==0.3.0
tqdm==4.45.0
numba==0.48


pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117


# module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline'
pip install "opencv-python-headless<4.3"

#长视频转换成短视频
python get_small_videos.py


# Place the LRS2 filelists (train, val) .txt files in the /Data2/wyw/LRS2/filelists/ folder
python process_data_lrs.py
#owndata
python process_data_own.py


#Preprocess the dataset for fast training
python preprocess.py --data_root /Data2/wyw/LRS2/main --preprocessed_root /Data2/wyw/LRS2/lrs2_preprocessed/
#own data
CUDA_VISIBLE_DEVICES=0,1,2,3 python preprocess.py --data_root /Data2/wyw/data_288/video --preprocessed_root /Data2/wyw/data_288/own_preprocessed/ --batch_size 8 --ngpu 4

# 1、Training the expert discriminator 鉴别器  loss到0.25较好
修改hparams.py第七行 filelists/换成/Data2/wyw/LRS2/filelists/ 或者/Data2/wyw/data_288/filelists/

修改audio.py中100行，换成return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin, fmax=hp.fmax)

#修改color_syncnet_train.py文件267行model run in parallel
#CUDA_VISIBLE_DEVICES=0,1,2,3 python color_syncnet_train.py --data_root  /Data2/wyw/LRS2/lrs2_preprocessed/ --checkpoint_dir ./syncnet_checkpoints
#python color_syncnet_train.py --data_root  /Data2/wyw/LRS2/lrs2_preprocessed/ --checkpoint_dir ./syncnet_checkpoints

#batchsize设置成256训练的会快些，修改hparams.py中81行batch_size=64,91行syncnet_batch_size=256，94、95行syncnet_eval_interval=5000,syncnet_checkpoint_interval=5000
CUDA_VISIBLE_DEVICES=0,1,2,3 python color_syncnet_train.py --data_root  /Data2/wyw/data_288/own_preprocessed/ --checkpoint_dir ./syncnet_checkpoints_own
# 加速https://github.com/Rudrabha/Wav2Lip/pull/265/commits/dc9e2655916272e3bee01aeafbe7939d11f5cbd5
CUDA_VISIBLE_DEVICES=0,1,2,3 python color_syncnet_train_speed_up.py --data_root  /Data2/wyw/data_288/own_preprocessed/ --checkpoint_dir ./syncnet_checkpoints_own_relu_bs_256_speed --checkpoint_path ./syncnet_checkpoints_own_relu_bs_256/checkpoint_step000115000.pth
报错Assertion input_val >= zero && input_val <= one failed
将color_syncnet_train.py文件 133行
logloss = nn.BCELoss()改成logloss = nn.BCEWithLogitsLoss()


将models/conv2.py文件中Conv2d中的PRelu改成Relu，Conv2dTranspose中prelu改成relu,其他的不改，训练模型保存在syncnet_checkpoints_conv2d_relu中

# Training the Wav2Lip models 生成器
#python wav2lip_train.py --data_root /Data2/wyw/LRS2/lrs2_preprocessed/ --checkpoint_dir ./checkpoints --syncnet_checkpoint_path ./syncnet_checkpoints
python wav2lip_train.py --data_root /Data2/wyw/data_288/own_preprocessed/ --checkpoint_dir ./checkpoints_own --syncnet_checkpoint_path ./syncnet_checkpoints_own

wav2lip_train.py文件179行logloss = nn.BCELoss()改成logloss = nn.BCEWithLogitsLoss()



# Preprocessed-CMLR-Dataset-For-Wav2Lip
https://github.com/zzj1111/Preprocessed-CMLR-Dataset-For-Wav2Lip


# 数据要求
1，严格数据质量
1.1 预外理后人脸大小尽量大于288x288(或者尽量靠近288*288)
1.2 每人人录制视频3-10分钟
1.3 1080P，至少720P
1.4 不能出现背景杂音(音量放到最大去听保证没有杂音)
1.5 25fos
1.6 每个视频处理为3秒
```


## Citing

To cite this repository:

```bibtex
@misc{Wav2Lip,
  author={Rudrabha},
  title={Wav2Lip: Accurately Lip-syncing Videos In The Wild},
  year={2020},
  url={https://github.com/Rudrabha/Wav2Lip}
}
```

