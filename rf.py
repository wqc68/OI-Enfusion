from osgeo import gdal
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# 读取TIFF文件并转换为float型
def read_tif(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)  # 读取单波段数据
        return data.astype(np.float32)

  # 加载990分辨率地表温度和因子数据
lst_990 = read_tif(r"H:\12532\20230725\lst\coarse\0725lst_990m.tif")
dem_990= read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_990m\dem.tif')
slope_990 = read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_990m\slope.tif')
ndvi_990= read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_990m\ndvi.tif')
lat_990= read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_990m\lat.tif')
lon_990 = read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_990m\lon.tif')
b2_990  = read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_990m\b2.tif')
lucc_990  = read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_990m\lucc.tif')
mndwi_990= read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_990m\mndwi.tif')
b3_990 = read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_990m\b3.tif')
b5_990 = read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_990m\b5.tif')
savi_990 = read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_990m\savi.tif')
b6_990= read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_990m\b6.tif')
#ndbi_990= read_tif(r'G:\12532\downscale_0918\kernels_90m\ndbi.tif')

# 将所有因子拼接成一个二维矩阵（每个因子都是一维的，形状相同）
X = np.stack([ dem_990, slope_990, lucc_990, ndvi_990, lat_990,lon_990,mndwi_990, savi_990, b2_990, b3_990, b5_990, b6_990], axis=-1)
#X = np.stack([ dem_990, slope_990, lucc_990, ndvi_990, lat_990,mndwi_990, savi_990, b2_990, b3_990, b5_990, b6_990,ndbi_990], axis=-1)
# 对应的目标变量（LST）
y = lst_990.flatten()  # 展平成一维
X = X.reshape(-1, X.shape[-1])  # 展平成2D数组，行数为像素点数，列数为特征数

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 初始化随机森林回归器
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 在训练集上训练模型
rf_model.fit(X_train, y_train)

# 在验证集上进行预测和评估
y_val_pred = rf_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)

print(f"验证集 RMSE: {val_rmse}")
print(f"验证集 R²: {val_r2}")

# 参数网格
param_grid = {
    'n_estimators': [500],
    'max_depth': [300],
    'min_samples_split':[2],
    'min_samples_leaf': [1]
}

# 网格搜索
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

# 输出最佳参数
print("最佳参数:", grid_search.best_params_)
# 使用最佳模型进行预测
best_rf_model = grid_search.best_estimator_
y_test_pred = best_rf_model.predict(X_test)

# 评估测试集
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print(f"测试集 RMSE: {test_rmse}")
print(f"测试集 R²: {test_r2}")

# 确保数据类型为float32或float64
lst_990 = lst_990.astype(np.float32)
dem_990 = dem_990.astype(np.float32)
slope_990 = slope_990.astype(np.float32)
ndvi_990 = ndvi_990.astype(np.float32)
lat_990 = lat_990.astype(np.float32)
lon_990 = lon_990.astype(np.float32)
b2_990 = b2_990.astype(np.float32)
lucc_990 =lucc_990.astype(np.float32)
mndwi_990 = mndwi_990.astype(np.float32)
b3_990 = b3_990.astype(np.float32)
b5_990 =b5_990.astype(np.float32)
savi_990 =savi_990.astype(np.float32)
b6_990 =b6_990.astype(np.float32)
#ndbi_990 =ndbi_990.astype(np.float32)

# 将数据展平并组合为DataFrame
data_990 = pd.DataFrame({
    "LST": lst_990.flatten(),
    "DEM": dem_990.flatten(),
    "Slope": slope_990.flatten(),
    "NDVI": ndvi_990.flatten(),
    "LAT": lat_990.flatten(),
    "LON": lon_990.flatten(),
    "b2": b2_990.flatten(),
    "lucc": lucc_990.flatten(),  
    "mndwi": mndwi_990.flatten(),
    "b3": b3_990.flatten(),
    "b5": b5_990.flatten(),  
    "savi": savi_990.flatten(),
    "b6": b6_990.flatten(), 
#    "ndbi": ndbi_990.flatten()
})

# 去除无效值
data_990 = data_990.dropna()

# Step 5: 逐个读取90m因子数据并进行预测
# 6. 生成30m分辨率的预测结果
dem_90m= read_tif(r"H:\area_diff\shanxi\downscale_0725\kernels_90m\dem.tif")
slope_90m = read_tif(r"H:\area_diff\shanxi\downscale_0725\kernels_90m\slope.tif")
ndvi_90m = read_tif(r"H:\area_diff\shanxi\downscale_0725\kernels_90m\ndvi.tif")
lat_90m = read_tif(r"H:\area_diff\shanxi\downscale_0725\kernels_90m\lat.tif")
lon_90m = read_tif(r"H:\area_diff\shanxi\downscale_0725\kernels_90m\lon.tif")
b2_90m = read_tif(r"H:\area_diff\shanxi\downscale_0725\kernels_90m\b2.tif")
lucc_90m = read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_90m\lucc.tif')
mndwi_90m= read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_90m\mndwi.tif')
b3_90m = read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_90m\b3.tif')
b5_90m = read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_90m\b5.tif')
savi_90m= read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_90m\savi.tif')
b6_90m= read_tif(r'H:\area_diff\shanxi\downscale_0725\kernels_90m\b6.tif')
#ndbi_90m= read_tif(r'G:\11931\downscale_0922\kernels_90m\ndbi.tif')

# 将90m因子拼接成输入特征矩阵
X_90m = np.stack([ dem_90m, slope_90m, lucc_90m,lat_90m,lon_90m, ndvi_90m, mndwi_90m, savi_90m, b2_90m, b3_90m,b5_90m, b6_90m], axis=-1)
#X_90m = np.stack([ dem_90m, slope_90m, lucc_90m,lat_90m, ndvi_90m, mndwi_90m, savi_90m, b2_90m, b3_90m,b5_90m, b6_90m,ndbi_90m], axis=-1)
X_90m = X_90m.reshape(-1, X_90m.shape[-1])

# 使用训练好的模型预测90m分辨率地表温度
lst_90m_pred = best_rf_model.predict(X_90m)

# 将预测结果重构为与输入数据相同的尺寸
lst_90m_pred = lst_90m_pred.reshape(lat_90m.shape)

# 从输入文件复制 geotransform 和 crs
with rasterio.open(r'H:\area_diff\shanxi\downscale_0725\kernels_90m\b6.tif') as src:
    transform = src.transform
    crs = src.crs

# 写入文件时使用相同的 transform 和 crs
with rasterio.open(
    r'H:\12532\11\lst.tif',
    'w',
    driver='GTiff',
    height=lst_90m_pred.shape[0],
    width=lst_90m_pred.shape[1],
    count=1,
    dtype=lst_90m_pred.dtype,
    transform=transform,
    crs=crs
) as dst:
    dst.write(lst_90m_pred, 1)
