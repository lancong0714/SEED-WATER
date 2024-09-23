# 此py是为了将数据随机分配为训练集和测试集
import os
import random
import shutil


def random_sample_and_move(source_dir, target_dir):
    # 创建目标目录，如果不存在
    os.makedirs(target_dir, exist_ok=True)

    # 遍历源目录中的每个子文件夹
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        # 确保是目录
        if os.path.isdir(folder_path):
            # 获取当前文件夹中的所有文件
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

            # 随机选择文件
            if files:  # 确保文件夹不为空
                selected_files = random.sample(files, 2)  #k=选择需要随机分配的文件个数

                # 创建新的子文件夹路径
                new_folder_path = os.path.join(target_dir, folder_name)
                os.makedirs(new_folder_path, exist_ok=True)

                # 分别移动选定的文件到新的子文件夹
                for selected_file in selected_files:
                    source_file_path = os.path.join(folder_path, selected_file)
                    target_file_path = os.path.join(new_folder_path, selected_file)
                    shutil.move(source_file_path, target_file_path)  # 使用move将文件移动

                    print(f"Moved {source_file_path} to {target_file_path}")


# 使用示例
source_directory = r".\seed EEM\9.18\train"  # 源目录路径
target_directory = r".\seed EEM\9.18\test-2"  # 目标目录路径

random_sample_and_move(source_directory, target_directory)