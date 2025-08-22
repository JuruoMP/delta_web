#!/usr/bin/env python3
import os
import argparse
import requests
import zipfile
from tqdm import tqdm

# 模型下载链接 (支持多种尺寸: tiny, base, small, medium)
# 注意: 多语言模型(不带.en后缀)本身支持中文识别
model_urls = {
    'tiny': {
        'en': "https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
        'multilingual': "https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"
    },
    'base': {
        'en': "https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        'multilingual': "https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
    },
    'small': {
        'en': "https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
        'multilingual': "https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
    },
    'medium': {
        'en': "https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin",
        'multilingual': "https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin"
    }
}

# 默认模型保存目录
default_model_dir = "./models"
# 默认模型尺寸
default_model_size = "base"


def download_file(url, save_path):
    """
    下载文件并显示进度条
    :param url: 下载链接
    :param save_path: 保存路径
    """
    try:
        # 发送GET请求，流式处理
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()  # 如果状态码不是200，抛出异常

        # 获取文件总大小
        total_size = int(response.headers.get('content-length', 0))

        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 下载文件并显示进度条
        with open(save_path, 'wb') as file, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)

        print(f"成功下载: {save_path}")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='下载Whisper模型')
    parser.add_argument('--model_dir', type=str, default=default_model_dir,
                        help=f'模型保存目录 (默认: {default_model_dir})')
    parser.add_argument('--en', action='store_true', help='仅下载英文模型')
    parser.add_argument('--zh', action='store_true', help='仅下载多语言模型(支持中文)')
    parser.add_argument('--size', type=str, default=default_model_size,
                        choices=['tiny', 'base', 'small', 'medium'],
                        help=f'模型尺寸 (默认: {default_model_size}, 可选: tiny, base, small, medium)')
    args = parser.parse_args()

    # 创建模型目录
    os.makedirs(args.model_dir, exist_ok=True)

    # 确定要下载的模型类型
    download_en = not args.zh  # 默认下载英文模型，除非指定--zh
    download_multilingual = not args.en  # 默认下载多语言模型(支持中文)，除非指定--en

    # 获取选定尺寸的模型URL
    size = args.size

    # 下载英文模型
    if download_en:
        en_model_url = model_urls[size]['en']
        en_model_filename = os.path.basename(en_model_url)
        en_model_path = os.path.join(args.model_dir, en_model_filename)
        print(f"下载英文 {size} 模型...")
        download_file(en_model_url, en_model_path)

    # 下载多语言模型(支持中文)
    if download_multilingual:
        multilingual_model_url = model_urls[size]['multilingual']
        multilingual_model_filename = os.path.basename(multilingual_model_url)
        multilingual_model_path = os.path.join(args.model_dir, multilingual_model_filename)
        print(f"下载多语言 {size} 模型(支持中文)...")
        download_file(multilingual_model_url, multilingual_model_path)

    print(f"{size} 尺寸模型下载完成！")


if __name__ == '__main__':
    main()