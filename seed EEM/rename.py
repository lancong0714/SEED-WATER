import os


def rename_images_in_subdirectories(directory):
    # 遍历指定目录及其所有子目录
    for root, dirs, files in os.walk(directory):
        # 遍历当前目录下的所有文件
        for file in files:
            # 检查文件名是否以 '.xlsx.png' 结尾
            if file.endswith('.xlsx.png'):
                # 创建新的文件名,去掉 '.xlsx'
                new_name = file.replace('.xlsx', '')
                # 构造完整的旧文件路径和新文件路径
                old_file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, new_name)
                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {old_file_path} to {new_file_path}')

        # 遍历当前目录下的所有子目录
        for dir in dirs:
            # 递归调用函数,处理子目录
            rename_images_in_subdirectories(os.path.join(root, dir))


# 调用函数,传入目标目录
rename_images_in_subdirectories('D:\HZAU\exp\EEM\seed EEM\9.18')