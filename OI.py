import numpy as np
from osgeo import gdal
import rasterio
from scipy.spatial.distance import cdist
from scipy.linalg import lstsq
from affine import Affine
from scipy.optimize import nnls

#def read_tif(file_path):
 #   with rasterio.open(file_path) as src:
 #       return src.read(1), src.transform

def read_tif(path: str):
    """读取栅格文件，返回数组和仿射变换参数"""
    dataset = gdal.Open(path)
    if dataset is None:
        raise ValueError(f"文件无法打开: {path}")
    array = dataset.ReadAsArray()           # 读取为NumPy数组
    transform = dataset.GetGeoTransform()   # 获取仿射变换参数
    return array, transform

# 读取影像数据shanxi
#F_result, transform_F = read_tif(r"H:\12532\20230725\starfm\starfm0918.tif")
#O_result, transform_O = read_tif(r"H:\12532\20230725\rf\lst_rf.tif")
#bilinear, _ = read_tif(r"G:\area_diff\shanxi\lst\fine\0725lst_90m.tif")
#bilinear, transform_bilinear = read_tif(r"H:\12532\20230725\lst\correct\0725lst_correct_990m_90m.tif")
#lucc, transform_lucc = read_tif(r"H:\area_diff\shanxi\downscale_0725\kernels_90m\lucc.tif")
#transform = transform_lucc

# 读取原始粗分辨率数据（990m）shenyang
#t1_coarse, _ = read_tif(r"G:\11931\LST\correct\0922lst_correct_990m.tif")
#lucc, transform_lucc = read_tif(r"H:\11931\downscale_0922\kernels_90m\lucc.tif")
#bilinear, transform_bilinear= read_tif(r"H:\11931\LST\correct\0922lst_correct_990m_90m.tif")
# 读取其他数据
#F_result, transform_F = read_tif(r"H:\11931\starfm\starfm.tif")
#O_result, transform_O = read_tif(r"H:\11931\downscale_0922\result\lst_rf.tif")
#transform = transform_lucc

#beijing
# 读取原始粗分辨率数据（990m）北京
#t1_coarse, _ = read_tif(r"G:\11931\LST\correct\0922lst_correct_990m.tif")
#lucc, transform_lucc = read_tif(r"H:\12332\20220902\downscale_0902\kernels_90m\lucc.tif")
#bilinear, transform_bilinear = read_tif(r"H:\12332\20220902\lst\correct\0902lst_correct_990m_90m.tif")
# 读取其他数据
#F_result, transform_F = read_tif(r"H:\12332\20220902\starfm\starfm0419.tif")
#O_result, transform_O = read_tif(r"H:\12332\20220902\downscale_0902\result\lst_rf.tif")
#transform = transform_lucc

# 读取原始粗分辨率数据（990m）河北下面
#lucc, transform_lucc = read_tif(r"H:\12433\downscale_0925\kernels_90m\lucc.tif")
#bilinear, transform_bilinear = read_tif(r"H:\12433\20220925\lst\correct\0925lst_correct_990m_90m.tif")
# 读取其他数据
#F_result, transform_F = read_tif(r"H:\12433\20220925\starfm\starfm.tif")
#O_result, transform_O = read_tif(r"H:\12433\downscale_0925\result\lst_rf.tif")
#transform = transform_lucc

#河南12436
#t1_coarse, _ = read_tif(r"G:\11931\LST\correct\0922lst_correct_990m.tif")
#lucc, transform_lucc = read_tif(r"H:\12436\downscale_0429\kernels_90m\lucc.tif")
#bilinear, transform_bilinear = read_tif(r"H:\12436\20230429\0429_correct_990m_90m.tif")
# 读取其他数据
#F_result, transform_F = read_tif(r"H:\12436\starfm\starfm_0707.tif")
#O_result, transform_O = read_tif(r'H:\12436\downscale_0429\result\lst_rf.tif')
#transform = transform_lucc


lucc, transform_lucc = read_tif(r"H:\area_diff\shanxi\downscale_0725\kernels_90m\lucc.tif")
bilinear, transform_bilinear = read_tif(r"H:\simulation\12532\lst\0725lst_990m_90m.tif")
# 读取其他数据
F_result, transform_F = read_tif(r"H:\extra\20230514\starfm\starfm.tif")
O_result, transform_O = read_tif(r'H:\simulation\12532\RF\lst_rf.tif')
transform = transform_lucc
 #参数设置
window_size = 7
half_window = window_size // 2
L_f = 500  # 背景场协方差尺度
L_o = 200 # 观测场协方差尺度
alpha = 0.1  # 正则化参数
beta = 0.5  # 观测场权重因子
alpha_reg = 1e-3  # 正则化系数
diff = O_result - F_result
max_delta = np.percentile(np.abs(diff), 99)  # 取95%分位数
#max_delta = 5.0   # 最大增量约束

rows, cols = F_result.shape
final_result = np.zeros_like(F_result, dtype=np.float32)
contribution = np.zeros_like(F_result, dtype=np.float32)
for i in range(rows):
    for j in range(cols):
        # 提取窗口
        row_min = max(0, i - half_window)
        row_max = min(rows, i + half_window + 1)
        col_min = max(0, j - half_window)
        col_max = min(cols, j + half_window + 1)

        F_result_window = F_result[row_min:row_max, col_min:col_max]
        lucc_window = lucc[row_min:row_max, col_min:col_max]
        O_result_window = O_result[row_min:row_max, col_min:col_max]
        bilinear_window = bilinear[row_min:row_max, col_min:col_max]

        # 筛选相似像素（同土地覆盖类型）
        center_class = lucc[i, j]
        mask = (lucc_window == center_class)
        local_indices = np.argwhere(mask)
        num_similar = len(local_indices)

        if num_similar < 10:
            final_result[i, j] = O_result[i, j]
            continue
        elif num_similar > 25:
            # 计算与中心像素的O差
            O_center = O_result[i, j]
            O_values = O_result_window[local_indices[:, 0], local_indices[:, 1]]
            O_diff = np.abs(O_values - O_center)

            # 选择差值最小的25个
            sorted_indices = np.argsort(O_diff)
            local_indices = local_indices[sorted_indices[:25]]
            num_similar = 25
        # 转换为全局坐标并计算距离
 #       global_rows = row_min + local_indices[:, 0]
 #       global_cols = col_min + local_indices[:, 1]
 #       points = np.column_stack((global_rows, global_cols))

        # 计算所有观测点对间的距离（用于协方差）
  #      distances = cdist(points, points)

        # 计算中心点到各观测点的距离
  #      center = np.array([[i, j]])
  #      center_distances = cdist(center, points)[0]
        # 确保transform是6元素元组

        if len(transform) > 6:
            transform = transform[:6]  # 截断到前6个元素

        # 创建Affine对象


        transform_affine = Affine.from_gdal(*transform)
        center_x, center_y = transform_affine * (j, i)
        cols_window = col_min + local_indices[:, 1]
        rows_window = row_min + local_indices[:, 0]
        x = transform_affine.c + cols_window * transform_affine.a + rows_window * transform_affine.b
        y = transform_affine.f + cols_window * transform_affine.d + rows_window * transform_affine.e
        points = np.column_stack((x, y))
        distances = cdist(points, points)
        center_distances = cdist([[center_x, center_y]], points)[0]

        # 计算lambda_j (|B-O| / |B-F|)
        B = bilinear_window[local_indices[:, 0], local_indices[:, 1]]
        O = O_result_window[local_indices[:, 0], local_indices[:, 1]]
        F = F_result_window[local_indices[:, 0], local_indices[:, 1]]
        lambda_j = (np.abs(B - O) + alpha) / (np.abs(B - F) + alpha)

        # 计算协方差矩阵
        mu_f = np.exp(-distances **2 / (2 * L_f **2))
 #       mu_o = np.exp(-distances ** 2 / (2 * L_o ** 2))
        R_diag =  (lambda_j ** 2) + alpha_reg
        spatial_corr = np.exp(-distances ** 2 / (2 * L_o ** 2))
        mu_o = np.diag(R_diag) * spatial_corr  # 对角线调整 + 空间相关
        A = mu_f + mu_o
        # 构建矩阵A和向量b (包含正则化)
 #       A = mu_f + mu_o * np.outer(lambda_j, lambda_j)
        A_reg = A + alpha_reg * np.eye(A.shape[0])
        b = np.exp(-center_distances ** 2 / (2 * L_f**2))

        # 求解带约束的最小二乘
        W, _  = nnls(A_reg, b)
    #    W = np.maximum(W, 0.0)

        # 安全归一化处理
        sum_W = np.sum(W)
        if sum_W > 1e-6:
            W /= sum_W
        else:
            W = np.ones_like(W) / len(W)

        # 增量约束
        delta = (1 - beta) * (O - F)
        delta_clipped = np.clip(delta, -max_delta, max_delta)

        # 直接赋值（无物理范围约束）
        final_result[i,j] = F_result[i,j] + np.dot(W, delta_clipped)
 #       contribution[i,j]=np.dot(W, delta_clipped)
# 从输入文件复制 geotransform 和 crs
with rasterio.open(r"H:\area_diff\shanxi\downscale_0725\kernels_90m\lucc.tif") as src:
    transform = src.transform
    crs = src.crs
    profile = src.profile

# 写入文件时使用相同的 transform 和 crs
with rasterio.open(
    r'H:\extra\20230514\oi\OI.tif',
    'w',
    driver='GTiff',
    height= final_result.shape[0],
    width= final_result.shape[1],
    count=1,
    dtype= final_result.dtype,
    transform=transform,
    crs=crs
) as dst:
    dst.write(final_result, 1)


