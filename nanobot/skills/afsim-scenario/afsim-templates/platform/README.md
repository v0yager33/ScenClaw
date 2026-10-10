# 平台模板

本目录存放 AFSIM platform_type 和全局配置模板。所有模板均从官方 Demo 交叉验证。

## 模板列表

| 文件 | 用途 | 验证来源 |
|------|------|----------|
| `air_platform.txt` | 防空导弹系统（传感器+处理器+武器） | engage.md |
| `space_platform.txt` | 卫星轨道 mover | satellite_demos.md |
| `global_config.txt` | 全局仿真配置 | satellite_demos.md |

## 命名规范

`{功能}_{名称}.txt`，例如 `air_fighter.txt`、`ground_sam.txt`、`ship_carrier.txt`

## space_mover 轨道参数速查

| 参数 | 说明 | 单位 |
|------|------|------|
| `semi_major_axis` | 半长轴 | km |
| `eccentricity` | 离心率（0=圆） | - |
| `inclination` | 轨道倾角 | deg |
| `raan` | 升交点赤经 | deg |
| `anomaly` | 近地点幅角 | deg |
| `revs_per_day` | 每天转数 | - |

## 全局配置速查

| 参数 | 说明 |
|------|------|
| `start_date` / `start_time` | 仿真开始时间（卫星仿真必须） |
| `end_time` | 仿真结束时间 |
| `include` / `include_once` | 包含外部文件 |
| `event_output` / `event_pipe` | 事件输出 |
| `dis_interface` | DIS 接口 |
