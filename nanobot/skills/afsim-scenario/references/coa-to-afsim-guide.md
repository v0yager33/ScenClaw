# COA → AFSIM 映射指南

> 本文档详细描述如何将 MissionOrch-LC 输出的 COA Markdown 方案
> 转化为 AFSIM 想定代码。包含算法描述和完整代码示例。

---

## 1. Entity → platform_type + platform

### 算法

```
对 COA Entity Manifest 中的每个 entity:
  1. 按 platform_code 查 platform-mapping.yaml
  2. 收集同 platform_code 的 entities → 共享一个 platform_type 定义
  3. 为每个 entity 实例生成 platform + route
```

### 示例: B_AWACS (KJ-500 x 1)

**COA 输入:**
```
| B_AWACS | KJ-500 | 1 | BLUE | 预警指挥 | EAGLE_EYE | <BASE:B-Mainland> | 无 |
```

**AFSIM 输出:**
```text
radar_signature AWACS_RADAR_SIGNATURE WSF_RADAR_SIGNATURE
   constant 50 m^2
end_radar_signature

platform_type KJ_500_TYPE WSF_PLATFORM
   side blue
   icon E-3_AWACS
   radar_signature AWACS_RADAR_SIGNATURE
   mover WSF_AIR_MOVER
      maximum_radial_acceleration 3 g
      maximum_linear_acceleration 2 g
   end_mover
   sensor awacs_radar WSF_GEOMETRIC_SENSOR
      frame_time 10 sec
      maximum_range 500 km
      reports_location
      ignore_same_side
      on
   end_sensor
   sensor link16_comm WSF_COMM_TRANSCEIVER
      frequency 1030 MHz
      maximum_range 600 km
      internal_link
   end_sensor
   processor track_fusion WSF_TRACK_PROCESSOR
      show_state_transitions
   end_processor
end_platform_type

platform eagle_eye KJ_500_TYPE
   side blue
   position 26:00:00n 119:30:00e altitude 35000 ft msl
      speed 450 kts
end_platform
```

### 示例: B_EW (J-16D x 2)

**COA 输入:**
```
| B_EW | J-16D | 2 | BLUE | 电子战 | GHOST_1, GHOST_2 | <BASE:B-Mainland> | 电子战吊舱 + 2xPL-15 |
```

**AFSIM 输出:**
```text
radar_signature J16D_RADAR_SIGNATURE WSF_RADAR_SIGNATURE
   constant 5 m^2
end_radar_signature

platform_type J_16D_TYPE WSF_PLATFORM
   side blue
   icon Su-30
   radar_signature J16D_RADAR_SIGNATURE
   mover WSF_AIR_MOVER
      maximum_radial_acceleration 9 g
      maximum_linear_acceleration 5 g
   end_mover
   sensor esm_sensor WSF_ESM_SENSOR
      frame_time 1 sec
      receive_only
      reports_bearing
      reports_location
      on
   end_sensor
   weapon pl_15 WSF_GUIDED_MOVER
      quantity 2
   end_weapon
end_platform_type

platform ghost_1 J_16D_TYPE
   side blue
   position 26:00:00n 119:30:00e altitude 30000 ft msl
      speed 550 kts
end_platform

platform ghost_2 J_16D_TYPE
   side blue
   position 26:00:00n 119:30:00e altitude 30000 ft msl
      speed 550 kts
end_platform
```

---

## 2. Phase Actions → route 航路点

### 行动关键词映射表

| COA 行动 | AFSIM route 元素 | 说明 |
|----------|-----------------|------|
| `TAKEOFF <BASE:X>` | 第一个 position = BASE 坐标 | 初始位置 |
| `MOVE_TO <WAYPOINT:X>` | position = WAYPOINT 坐标 + speed | 航路点 |
| `MOVE_TO <ORBIT:X>` | position = ORBIT 坐标 + speed | 进入轨道 |
| `PATROL <ORBIT:X>` | 两个交替 position 形成巡逻线 | 需生成两个相邻点 |
| `LOITER <ORBIT:X>` | 单个 position（停留） | 仿真时间控制 |
| `TRANSIT <ORBIT:X>` | position = 中间航路点 | 过渡飞行 |
| `EGRESS <WAYPOINT:X>` | position = 撤离点 | 脱离战场 |
| `RTB <BASE:X>` | 最后一个 position = 基地坐标 | 返航 |
| `REFUEL` | position = 加油区坐标 | 空中加油 |
| `ENGAGE` | 不生成 route 点 | 武器交战由 weapon 定义处理 |
| `JAM` | 不生成 route 点 | 电子战行为由 script 处理 |
| `STRIKE` | 不生成 route 点 | 投弹由 weapon 定义处理 |
| `INTERCEPT` | position = 拦截阵位 | 接敌机动 |

### 示例: B_SQ_01 (J-16) 全程 route

**COA 行动序列:**
```
Phase 1: TAKEOFF <BASE:B-Mainland>, FORM_UP <WAYPOINT:WP-MARSHAL>
Phase 2: TRANSIT <ORBIT:ORB-REFUEL>, REFUEL
Phase 3: MOVE_TO <WAYPOINT:WP-IP> (距目标150-180km)
Phase 4: MOVE_TO (前推至120-150km), ENGAGE
Phase 5: LOITER <ORBIT:ORB-HOLD>
Phase 6: EGRESS <WAYPOINT:WP-EXFIL>
Phase 7: REFUEL, RTB <BASE:B-Mainland>
```

**AFSIM route 输出:**
```text
platform thunder_11 J_16_TYPE
   side blue
   route
      position 26:00:00n 119:30:00e altitude 30000 ft msl
         speed 550 kts
      position 26:05:00n 119:40:00e altitude 30000 ft msl
         speed 550 kts
      position 26:30:00n 120:00:00e altitude 30000 ft msl
         speed 450 kts
      position 27:00:00n 121:00:00e altitude 30000 ft msl
         speed 550 kts
      position 27:30:00n 122:00:00e altitude 30000 ft msl
         speed 550 kts
      position 27:45:00n 122:30:00e altitude 30000 ft msl
         speed 550 kts
      position 27:30:00n 122:00:00e altitude 30000 ft msl
         speed 550 kts
      position 27:00:00n 121:30:00e altitude 30000 ft msl
         speed 500 kts
      position 26:00:00n 119:30:00e altitude 30000 ft msl
         speed 450 kts
   end_route
end_platform
```

---

## 3. Geo Reference → position 坐标计算

### 坐标解算算法

COA 地理参考格式: `<TYPE:ID>` + hint（如 "距目标岛礁480-520km"）

**Step 1: 确定参考中心点**

从 Geo Reference 表中找到核心目标区域坐标。例如:
```
目标岛礁中心 ≈ 27:30:00n 122:30:00e（假设）
```

**Step 2: 距离→经纬度偏移**

大圆距离近似公式（适用于 <1000km）:

```
Δlat (度) = 距离(km) / 111.32
Δlon (度) = 距离(km) / (111.32 × cos(lat_rad))
```

**Step 3: 方位角推断**

从 hint 文本中的关键词推断方位角:

| 关键词 | 方位角 | 说明 |
|--------|--------|------|
| "前方" / "东" | 90° | 正东方向 |
| "侧翼" / "北" | 0° | 正北方向 |
| "后方" / "西" | 270° | 正西方向 |
| "南" | 180° | 正南方向 |
| 无方向词 | 均匀分布 | 围绕中心点均匀展开 |

**Step 4: 坐标计算**

```python
import math

def geo_offset(center_lat, center_lon, distance_km, bearing_deg):
    """从中心点 + 距离 + 方位角计算目标坐标"""
    lat_rad = math.radians(center_lat)
    bearing_rad = math.radians(bearing_deg)

    # 角度制近似（<1000km 精度足够）
    delta_lat = distance_km * math.cos(bearing_rad) / 111.32
    delta_lon = distance_km * math.sin(bearing_rad) / (111.32 * math.cos(lat_rad))

    return center_lat + delta_lat, center_lon + delta_lon

def to_afsim_coord(lat, lon):
    """转换为 AFSIM d:m:s 坐标格式"""
    lat_dir = "n" if lat >= 0 else "s"
    lon_dir = "e" if lon >= 0 else "w"

    lat_abs = abs(lat)
    lon_abs = abs(lon)

    lat_d = int(lat_abs)
    lat_m = int((lat_abs - lat_d) * 60)
    lat_s = int(((lat_abs - lat_d) * 60 - lat_m) * 60)

    lon_d = int(lon_abs)
    lon_m = int((lon_abs - lon_d) * 60)
    lon_s = int(((lon_abs - lon_d) * 60 - lon_m) * 60)

    return f"{lat_d}:{lat_m:02d}:{lat_s:02d}{lat_dir} {lon_d}:{lon_m:02d}:{lon_s:02d}{lon_dir}"
```

### 示例: ORB-AWACS (距目标岛礁480-520km)

```
中心点: 27:30:00n 122:30:00e
距离: 500 km（取中值）
方位角: 270°（在目标西侧）

计算:
  delta_lat = 500 × cos(270°) / 111.32 = 0
  delta_lon = 500 × sin(270°) / (111.32 × cos(27.5°)) = -500 / 98.78 = -5.06°

结果: (27.5°, 122.5° - 5.06°) = (27.5°, 117.44°)
AFSIM: 27:30:00n 117:26:24e
```

---

## 4. Weapons → weapon_type 定义

### 武器映射表

| COA 武器名 | AFSIM weapon_type 名 | mover 参数 | 备注 |
|-----------|---------------------|-----------|------|
| 鹰击-91 / YJ-91 | yj_91 | WSF_GUIDED_MOVER | 反辐射导弹 |
| PL-15 | pl_15 | WSF_GUIDED_MOVER | 超视距空空导弹 |
| PL-10 | pl_10 | WSF_GUIDED_MOVER | 近距格斗弹 |
| 48N6E3 (S-400) | 48n6e3 | WSF_GUIDED_MOVER | 远程地空导弹 |
| 精确制导炸弹 | guided_bomb | WSF_GUIDED_MOVER | 卫星/INS 制导 |
| R-77-1 | r77_1 | WSF_GUIDED_MOVER | 俄制中距弹 |
| R-73 | r73 | WSF_GUIDED_MOVER | 俄制近距弹 |

### 示例: 鹰击-91 定义

```text
weapon_type yj_91 WSF_GUIDED_MOVER
   speed 900 m/s
   maximum_range 150 km
   seeker WSF_GEOMETRIC_SENSOR
      receive_only
      frame_time 0.1 sec
      maximum_range 30 km
   end_seeker
   warhead 80 kg
end_weapon_type
```

---

## 5. 效果链 → 仿真终止条件

COA 效果链中的"衡量指标"可映射为 AFSIM 的仿真终止观察条件:

| COA 效果衡量 | AFSIM 对应 |
|-------------|-----------|
| "辐射源停机 >10 分钟" | 通过 event_output 监控 sensor 状态 |
| "瘫痪指数 ≥ 80%" | script 中计数 + 条件终止 |
| "拦截成功率 100%" | event_pipe 记录交战结果 |

一般使用 `end_time` 控制总仿真时长，辅以 `event_output` 记录关键事件。

---

## 6. 决策点 → processor 状态机

COA 决策点可映射为 AFSIM processor 的多状态逻辑:

### 示例: DP1 (雷达静默 → 切换坐标攻击)

```text
processor dp1_logic WSF_TASK_PROCESSOR
   show_state_transitions
   evaluation_interval 30 sec

   state SEARCH
      evaluation_interval 30 sec
      next_state COORD_ATTACK
   end_state

   state COORD_ATTACK
      evaluation_interval 10 sec
   end_state
end_processor
```

---

## 7. 完整端到端示例

给定一个简化的 COA（1 阶段、2 实体）:

**COA MD 输入:**
```markdown
# COA方案：简化SEAD测试

## 实体清单
| unit_id | platform_code | quantity | side | role | callsign | base | weapons |
| B_EW | J-16D | 1 | BLUE | 电子战 | GHOST | BASE-A | PL-15x2 |
| SAM | S-400 | 1 | RED | 防空 | RED-SAM | - | 48N6E3x4 |

## 地理参考
| ref_id | type | side | hint |
| <BASE:BASE-A> | BASE | BLUE | 26:00:00n 119:30:00e |
| <TARGET:SAM-SITE> | TARGET | RED | 27:30:00n 122:30:00e |

## COA矩阵（1 阶段）
| 实体 | 阶段1: 交战 |
| B_EW | 1. MOVE_TO 距目标400km 2. JAM 3. ESM侦收 |
| SAM | 1. 开机搜索 2. 跟踪目标 3. 发射导弹 |
```

**生成的 AFSIM 想定:**

```text
# === Signatures ===
radar_signature J16D_RADAR_SIGNATURE WSF_RADAR_SIGNATURE
   constant 5 m^2
end_radar_signature

radar_signature S400_RADAR_SIGNATURE WSF_RADAR_SIGNATURE
   constant 10 m^2
end_radar_signature

# === Platform Types ===
platform_type J_16D_TYPE WSF_PLATFORM
   side blue
   icon Su-30
   radar_signature J16D_RADAR_SIGNATURE
   mover WSF_AIR_MOVER
      maximum_radial_acceleration 9 g
      maximum_linear_acceleration 5 g
   end_mover
   sensor esm_sensor WSF_ESM_SENSOR
      frame_time 1 sec
      receive_only
      reports_bearing
      reports_location
      on
   end_sensor
   weapon pl_15
      quantity 2
   end_weapon
end_platform_type

platform_type S_400_TYPE WSF_PLATFORM
   side red
   icon SA-21_Launcher
   radar_signature S400_RADAR_SIGNATURE
   mover WSF_GROUND_MOVER
   end_mover
   sensor engagement_radar WSF_GEOMETRIC_SENSOR
      frame_time 1 sec
      maximum_range 250 km
      reports_location
      ignore_same_side
      on
   end_sensor
   weapon 48n6e3
      quantity 4
   end_weapon
   processor track_proc WSF_TRACK_PROCESSOR
      show_state_transitions
   end_processor
end_platform_type

# === Platform Instances ===
platform ghost J_16D_TYPE
   side blue
   route
      position 26:00:00n 119:30:00e altitude 30000 ft msl
         speed 550 kts
      position 27:00:00n 120:30:00e altitude 30000 ft msl
         speed 550 kts
   end_route
end_platform

platform red_sam S_400_TYPE
   side red
   position 27:30:00n 122:30:00e altitude 0 ft msl
end_platform

# === Global Config ===
random_seed 1
end_time 3600 sec
log_file output/simplified_sead.log
thread_count 4
```
