# AFSIM 快速参考

> 黄金法则 + 最小原型 + 渐进开发流程。详细 Demo 见 `demos-index.md`。

---

## 黄金法则

> **不要凭空自造 AFSIM 语法，要基于验证过的模板和官方 Demo 输出！**

### 4 个禁止

1. 从零开始写（语法错误率 90%+）
2. 看 Demo 自己写（细节会错）
3. 同时改多处（无法定位错误）
4. 不验证就继续写（错误累积）

### 4 个必须

1. 从官方 Demo 开始（复制粘贴，不自己写）
2. 每次只改一点，立即验证
3. 保留黄金基准（能运行的版本）
4. 报错 → errors-ref 匹配 → 教训激活 → 修正

---

## 最小原型 1：飞机平台（推荐起点）

**43 行，最简单，适合 99% 的项目。**

```text
radar_signature FIGHTER_RADAR_SIGNATURE
  constant 10 m^2
end_radar_signature

platform_type TARGET_TYPE WSF_PLATFORM
   side blue
   icon F-18E
   radar_signature FIGHTER_RADAR_SIGNATURE
   mover WSF_AIR_MOVER
      debug
      print_route true
      maximum_radial_acceleration 150 g
      maximum_linear_acceleration 150 g
   end_mover
end_platform_type

platform test_aircraft TARGET_TYPE
   side blue
   route
      position 30:00:00n 120:00:00e altitude 10000 ft msl
         speed 300 kts
      position 30:30:00n 120:30:00e altitude 10000 ft msl
         speed 300 kts
   end_route
end_platform

random_seed 1
end_time 30 sec
log_file output/minimal_test.log
```

---

## 最小原型 2：固定地面平台

**~15 行，适合地面/舰船固定平台。**

```text
platform_type LAUNCHER_TYPE WSF_PLATFORM
   side red
   icon SA-10_Launcher
   indestructible
end_platform_type

platform launcher1 LAUNCHER_TYPE
   side red
   position 40:00:00n 116:00:00e altitude 0 ft msl
end_platform
```

---

## 最小原型 3：带传感器

```text
sensor find WSF_GEOMETRIC_SENSOR
   ignore_same_side
   frame_time 10 sec
   reports_location
   on
   track_quality 0.5
   debug
end_sensor

sensor ttr LRSAM_TTR
   on
end_sensor
```

---

## 核心基类型速查

| 类型 | 用途 |
|------|------|
| `WSF_PLATFORM` | 所有平台基类 |
| `WSF_AIR_MOVER` | 飞机 mover |
| `WSF_SURFACE_MOVER` | 舰船 mover |
| `WSF_GROUND_MOVER` | 地面 mover |
| `WSF_SPACE_MOVER` | 太空/卫星 mover（开普勒根数轨道） |
| `WSF_GUIDED_MOVER` | 导弹 mover |
| `WSF_GEOMETRIC_SENSOR` | 几何传感器 |
| `WSF_RADAR_SENSOR` | 雷达传感器 |
| `WSF_ESM_SENSOR` | 电子支援措施传感器（纯接收测向） |
| `WSF_RADAR_SIGNATURE` | 雷达反射截面积（需 `end_radar_signature` 闭合） |
| `WSF_OPTICAL_SIGNATURE` | 光学特征（可选基类型，Demo 中常无基类型） |
| `WSF_TASK_PROCESSOR` | 任务处理器 |
| `WSF_TRACK_PROCESSOR` | 航迹处理器 |
| `WSF_COMM_TRANSCEIVER` | 通信收发器 |
| `WSF_GUIDANCE_COMPUTER` | 制导计算机 |

---

## radar_signature / optical_signature 多种格式

### radar_signature（必须基类型 + 闭合）

```text
# 固定 RCS
radar_signature <name> WSF_RADAR_SIGNATURE
   constant 1.0 m^2
end_radar_signature

# 频率依赖 RCS（PLT 文件）
radar_signature <name>
   plt_file plt_radar_sig.txt default horizontal
end_radar_signature

# 双基地雷达
radar_signature <name>
   constant .0001 m^2
   bistatic_signature
      azimuth -180 deg
         elevation -90 deg
end_radar_signature
```

### optical_signature（两种格式皆可）

```text
# 格式 A：无基类型（Demo 中更常见）
optical_signature <name>
   constant 1000 m^2
end_optical_signature

# 格式 B：有基类型
optical_signature <name> WSF_OPTICAL_SIGNATURE
   intensity 100
end_optical_signature
```

---

## 渐进开发流程

```
选择原型 → Warlock 验证 ✅ → 改一点 → 验证 ✅
    → 改一点 → 验证 ✅ → ... → 完成
```

---

## 常见语法陷阱

| 陷阱 | 错误 | 正确 |
|------|------|------|
| 行尾注释 | `speed 300 kts # comment` | 注释在独立行 |
| 块不闭合 | 缺少 `end_mover` 等 | 每个块必须有 `end_xxx` |
| 坐标格式 | `30 120` | `30:00:00n 120:00:00e` |
| 速度单位 | `speed 300` | `speed 300 kts` 或 `speed 150 m/s` |
| 高度单位 | `altitude 10000` | `altitude 10000 ft msl` |
