# 传感器模板

本目录存放 AFSIM sensor 模板。所有模板均从官方 Demo 交叉验证。

## 模板列表

| 文件 | 类型 | 用途 | 验证来源 |
|------|------|------|----------|
| `geometric_sensor.txt` | WSF_GEOMETRIC_SENSOR | 几何传感器（不考虑物理） | engage.md line 51 |
| `radar_sensor.txt` | WSF_RADAR_SENSOR | 主动/被动雷达 | sensor_demos.md line 39, 1380 |
| `antenna_pattern.txt` | antenna_pattern | 天线方向图（6种类型） | sensor_demos.md line 31+ |

## 使用方式

1. 选择对应类型的模板文件
2. 复制到 `platform_type` 块内，替换 `<占位符>`
3. 在 Warlock/Wizard 中验证

## antenna_pattern 方向图类型速查

| 类型 | 关键词 | 典型参数 |
|------|--------|----------|
| 常数 | `constant` | `constant 0 db` |
| 圆形 | `circular_pattern` | beamwidth, peak_gain |
| 矩形 | `rectangular_pattern` | azimuth_beamwidth, elevation_beamwidth |
| 恒定 | `constant_pattern` | azimuth_beamwidth, elevation_beamwidth |
| 正弦 | `sine_pattern` | azimuth_beamwidth, elevation_beamwidth |
| 均匀 | `uniform_pattern` | beamwidth |
