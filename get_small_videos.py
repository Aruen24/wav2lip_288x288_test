# _*_ coding:utf-8 _*_

import os
import cv2
import sys
import re
import shutil
import subprocess
from datetime import datetime, timedelta
import moviepy.editor as mpe

# 选取数据和数据的处理要求:
# 1.1 预处理后人脸大小尽量大于288x288 （对处理后的视频帧逐帧筛选来判断该视频是否满足该条件）
# 1.2 每个人录制视频3-10分钟
# 1.3 1080P，至少720P
# 1.4 不能出现背景杂音(音量放到最大去听保证没有杂音) (对处理后的视频进行逐个筛选来判断该视频是否满足该条件)
# 1.5 25fps, 音频采样为16K
# 1.6 每个视频处理为3秒

# 这里是针对一个视频的处理，将每个视频处理成为连续的3s，然后再手动去筛选和清洗视频

# 参考链接: https://zhuanlan.zhihu.com/p/142396915 python调用FFmpeg把视频剪辑成多段
#          https://blog.csdn.net/mooneve/article/details/132576731  使用ffmpeg截取视频片段
#          https://blog.csdn.net/weixin_42182599/article/details/127496704 1024程序员节｜FFmpeg 调整声道数、采样率、码率

#
src_videos_path = "../data/readbook.mp4"
# src_videos_path_25fps_16k_2 = '../data/anrang_25fps_16k_2.mp4'
dst_videos_path = "../dst/"

# 先对src_videos_path进行25fps、16K采样、双通道转换
# cmd = 'ffmpeg ' + '-r 25 ' + '-i ' + src_videos_path + ' ' + '-ar ' + '16000 ' + '-ac 2 ' + src_videos_path_25fps_16k_2
# subprocess.call(cmd, shell=True)

TIME_FROMAT = '%H:%M:%S'

def do_cut(file_input, file_output, file_output_25fps, file_output_25fps_16k, s1_time, s2_time):
    start_time = s1_time.strftime(TIME_FROMAT)

    end_time = s2_time.strftime(TIME_FROMAT)

    print("start_time: ", start_time)
    print('end_time: ', end_time)

    # cmd = 'ffmpeg -i ' + file_input + ' -ss ' + start_time + ' -to ' + end_time + ' -c:v copy -c:a copy ' + file_output
    # cmd = 'ffmpeg' + ' -i ' + file_input + ' -ss ' + start_time + ' -to ' + end_time + ' -c:v copy -c:a copy ' + file_output
    # cmd = 'ffmpeg' + ' -i ' + file_input + ' -ss ' + start_time + ' -t ' + end_time + ' -c copy ' + file_output # 不推荐这种写法
    # cmd = 'ffmpeg' + ' -i ' + file_input + ' -c:v libx264 ' + '-crf 18 ' + '-ss ' + start_time + ' -t 3 ' + ' ' + file_output  # 这种写法可以缓解卡着不动情况
    cmd = 'ffmpeg' + ' -i ' + file_input + ' -c:v libx264 ' + '-crf 18 ' + '-ss ' + start_time + ' -t 3 ' + '-r 25 ' + '-ar ' + '16000 ' + '-ac 2 ' + file_output  # 这种写法可以缓解卡着不动情况
                                                                                                                                                                   # 切为3秒，25fps, 16K采样， 双通道
                                                                                                                                                                   # 或者有一种可能就是先把完整的视频转换为25fps、16k采样、双通道
    subprocess.call(cmd, shell=True)

    # 转换为25fps
    # cmd1 = 'ffmpeg ' + '-r 25 ' + '-i ' + file_output + ' ' + file_output_25fps
    # subprocess.call(cmd1, shell=True)

    # 转换为16k
    # cmd2 = 'ffmpeg ' + '-y ' + '-i ' + file_output_25fps + ' -ar ' + '16000 ' + file_output_25fps_16k
    # subprocess.call(cmd2, shell=True)

    # 删除25fps和16k

    # 将视频转换为fps=25,且采样率为16k

def do_edit(src_videos_path, dst_videos_path, start_time, end_time, slice_duration):
    if not os.path.exists(dst_videos_path):
        os.makedirs(dst_videos_path)
    video_name = os.path.basename(src_videos_path)

    if not os.path.exists(os.path.join(dst_videos_path, video_name)):
        os.makedirs(os.path.join(dst_videos_path, video_name[:-4]))

    s_time = datetime.strptime(start_time, TIME_FROMAT)
    e_time = datetime.strptime(end_time, TIME_FROMAT)

    n_slice = (int)((e_time - s_time).total_seconds() / slice_duration)

    s1_time = s_time

    for i in range(0, n_slice):
        s2_time = s1_time + timedelta(seconds=slice_duration)
        file_output = os.path.join(dst_videos_path, video_name[:-4], str(i) + '.mp4')
        file_output_25fps = os.path.join(dst_videos_path, video_name[:-4], str(i) + '_' + '25fps.mp4')
        file_output_25fps_16k = os.path.join(dst_videos_path, video_name[:-4], str(i) + '_' + '25fps_16k.mp4')
        do_cut(src_videos_path, file_output, file_output_25fps, file_output_25fps_16k, s1_time, s2_time)
        s1_time = s2_time

if __name__ == "__main__":
    # 计算语音的秒长，并将其转化为时分秒格式
    clip = mpe.AudioFileClip(src_videos_path)
    duration = clip.duration
    clip.close()
    audio_time_secs = int(duration)
    # 将秒转换为时:分:秒格式
    m, s = divmod(audio_time_secs, 60)
    h, m = divmod(m, 60)
    last = "%02d:%02d:%02d" % (h, m, s)
    print(last)
    do_edit(src_videos_path, dst_videos_path, "00:00:00", last, 3)

    # 删除
    # os.remove(src_videos_path_25fps_16k_2)










