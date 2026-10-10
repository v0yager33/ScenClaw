# 脚本与通信模板

本目录存放 AFSIM script 和通信模板。所有模板均从官方 Demo 交叉验证。

## 脚本模板

| 文件 | 用途 | 验证来源 |
|------|------|----------|
| `observer_events.txt` | 事件钩子（WeaponTerminated 等） | script_demos.md line 35-56 |

## 通信模板

| 文件 | 用途 | 验证来源 |
|------|------|----------|
| `comm_transceiver.txt` | 通信收发器（内部链路/卫星通信） | engage.md line 26, comm.md |

## observer 事件类型速查

| 事件 | 触发时机 |
|------|----------|
| `WEAPON_TERMINATED` | 武器终止（超时/诱饵） |
| `WEAPON_HIT` | 武器命中目标 |
| `WEAPON_FIRED` | 武器发射 |
| `WEAPON_MISSED` | 武器未命中 |
| `PLATFORM_KILLED` | 平台被摧毁 |
| `PLATFORM_BROKEN` | 平台受损 |
| `DETECTION` | 新目标被探测 |
| `TRACK_ESTABLISHED` | 航迹建立 |
