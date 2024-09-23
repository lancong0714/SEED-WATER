# 此py是为了绘画三维荧光光谱图谱

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


class FluoDataType:
    def __init__(self, file_path, bg_file_path):
        data = pd.read_excel(file_path).values
        data = data[np.where(data == "Data points")[0][0] + 1:, :]

        self.ex_begin = int(data[0][1])
        self.ex_end = int(data[0][-1])
        self.ex_step = int(data[0][2] - data[0][1])
        self.em_begin = int(data[1][0])
        self.em_end = int(data[-1][0])
        self.em_step = int(data[2][0] - data[1][0])

        self.data = np.array(data[1:, 1:], dtype=np.float32)

        self.sub_background(bg_file_path)
        self.sub_scatter()
        self.unify_data()

    def get_ex(self):
        return self.ex_begin, self.ex_end, self.ex_step

    def get_em(self):
        return self.em_begin, self.em_end, self.em_step

    def sub_background(self, bg_file_path):
        bgdata = pd.read_excel(bg_file_path).values
        bgdata = bgdata[np.where(bgdata == "Data points")[0][0] + 2:, 1:]

        if bgdata.shape != self.data.shape:
            raise ValueError("Background data shape does not match fluorescence data shape.")

        self.data = np.array(self.data - bgdata, dtype=np.float32)

    def sub_scatter(self):
        ex = np.array(range(self.ex_begin, self.ex_end + 1, self.ex_step))
        em = np.array(range(self.em_begin, self.em_end + 1, self.em_step))

        self.data[self.data < 0] = 0  # Remove negative values

        # Remove scattering effects
        for j in range(len(ex)):
            for i in range(len(em)):
                if (ex[j] - 10 <= em[i] <= ex[j] + 10 and self.data[i][j] >= 500) or \
                        (2 * ex[j] - 30 <= em[i] <= 2 * ex[j] + 30 and self.data[i][j] >= 500):
                    self.data[i][j] = 0

    def draw_contour(self, pic_name, pic_long=12.8, pic_height=7.2, save_path=None):
        plt.rcParams['figure.figsize'] = (pic_long, pic_height)

        ex = np.array(range(self.ex_begin, self.ex_end + 1, self.ex_step))
        em = np.array(range(self.em_begin, self.em_end + 1, self.em_step))

        ex, em = np.meshgrid(ex, em)

        plt.figure()
        plt.contourf(ex, em, self.data)
        plt.contour(ex, em, self.data)

        plt.tick_params(labelsize=16)
        plt.title(pic_name, fontdict={"family": "Times New Roman"}, fontsize=32)
        plt.xlabel('Ex', fontdict={"family": "Times New Roman"}, fontsize=28)
        plt.ylabel('Em', fontdict={"family": "Times New Roman"}, fontsize=28)

        if save_path is not None:
            plt.savefig(save_path)
            print("已保存" + pic_name + "图片，保存路径为：" + save_path)
        else:
            plt.show()

        plt.close()

    def unify_data(self, ex_begin=200, ex_end=400, em_begin=270, em_end=500):
        ex = np.array(range(self.ex_begin, self.ex_end + 1, self.ex_step))
        em = np.array(range(self.em_begin, self.em_end + 1, self.em_step))

        i_begin = next((i for i in range(len(em)) if em[i] == em_begin), None)
        i_end = next((i for i in range(len(em)) if em[i] == em_end), None)

        j_begin = next((j for j in range(len(ex)) if ex[j] == ex_begin), None)
        j_end = next((j for j in range(len(ex)) if ex[j] == ex_end), None)

        if i_begin is not None and i_end is not None and j_begin is not None and j_end is not None:
            self.data = self.data[i_begin:i_end + 1, j_begin:j_end + 1]
            self.ex_begin = ex_begin
            self.ex_end = ex_end
            self.em_begin = em_begin
            self.em_end = em_end
        else:
            raise ValueError("指定的波长范围超出数据范围，请检查输入参数。")


def process_fluorescence_data(directory, save_directory, bg_file_path):
    main_files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]

    if not main_files:
        print("没有找到符合条件的文件，请检查目录和文件扩展名。")

    for main_file in tqdm(main_files, desc="Processing files"):
        main_file_path = os.path.join(directory, main_file)

        try:
            fluo_data = FluoDataType(main_file_path, bg_file_path)
            pic_name = f'Contour Plot for {os.path.basename(main_file)}'
            save_path = os.path.join(save_directory, f'{os.path.basename(main_file)}.png')
            fluo_data.draw_contour(pic_name, save_path=save_path)

        except Exception as e:
            print(f"处理文件 {main_file} 时出错: {e}")


if __name__ == '__main__':
    data_directory = r".\seed EEM\data\9.18-data\main"
    save_directory = r".\seed EEM\data\9.18-data\pic\9.18"
    background_file = r".\seed EEM\data\9.18-data\h20.xlsx"

    process_fluorescence_data(data_directory, save_directory, background_file)