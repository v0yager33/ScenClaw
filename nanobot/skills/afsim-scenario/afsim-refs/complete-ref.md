# AFSIM 完整语法参考

> 按块类型分节的完整语法参考。仅在用户询问具体语法细节时读取。

---

## platform_type

```text
platform_type PLATFORM_NAME WSF_PLATFORM
   side <side_name>                    # 红/蓝/绿方
   icon <icon_name>                    # 图标名称，如 F-18E, SA-10_Launcher
   radar_signature <signature_name>     # 雷达反射截面积名称
   optical_signature <signature_name>   # 光学特征名称（可选）
   indestructible                      # 不可摧毁（可选，地面平台常用）
   mover <mover_type>                 # 运动类型
      <mover_parameters>
   end_mover
   sensor <sensor_name> <sensor_type> # 传感器定义
      <sensor_parameters>
   end_sensor
   weapon <weapon_name> <weapon_type> # 武器定义
      <weapon_parameters>
   end_weapon
   processor <processor_name> <processor_type> # 处理器定义
      <processor_parameters>
   end_processor
end_platform_type
```

---

## platform（实例）

```text
platform INSTANCE_NAME PLATFORM_TYPE_NAME
   side <side_name>
   position <lat> <lon> altitude <alt> <ref>  # d:m:s N/S e/w, ft/m, msl/agl
   heading <degrees>                # 航向（可选）
   speed <value> <unit>            # 速度（可选）
   route
      position <lat> <lon> altitude <alt> <ref>
         speed <value> <unit>
      position <lat> <lon> altitude <alt> <ref>
         speed <value> <unit>
   end_route
end_platform
```

---

## route（路径）

```text
route ROUTE_NAME
   waypoint <lat> <lon> altitude <alt> <ref> <speed> <unit>
   waypoint <lat> <lon> altitude <alt> <ref> <speed> <unit>
end_route
```

或使用 `run` 块：

```text
run
   flight_route
       (X, Y, Z Position) (Speed)
          0.0     0.0 100.0 m 200.0 m/sec
       2000.0     0.0 100.0 m 200.0 m/sec
   end_flight_route
end_run
```

---

## sensor

### WSF_GEOMETRIC_SENSOR

```text
sensor <name> WSF_GEOMETRIC_SENSOR
   location <x> <y> <z> <unit>     # 传感器在平台上的位置
   ignore_same_side                 # 忽略同阵营目标
   frame_time <value> <unit>        # 帧时间
   reports_location                 # 报告目标位置
   on                              # 默认开启
   track_quality <value>            # 航迹质量 0-1
   debug                            # 调试输出（可选）
end_sensor
```

### WSF_RADAR_SENSOR

```text
sensor <name> WSF_RADAR_SENSOR
   location <x> <y> <z> <unit>
   on
   receiver
      antenna_pattern <pattern_name>
      antenna_height <value> <unit>
      frequency <value> <unit>
      bandwidth <value> <unit>
   end_receiver
end_sensor
```

---

## weapon

```text
weapon <name> <weapon_type>
   quantity <n>                     # 数量
   active <true|false>              # 默认激活
end_weapon
```

内置武器类型：`LRSAM`, `SRSAM`, `AAM`, `ASM`, `BOMB`, `TORPEDO` 等。

---

## processor

### WSF_TASK_PROCESSOR

```text
processor <name> WSF_TASK_PROCESSOR
   show_state_transitions           # 显示状态转换（调试）
   evaluation_interval <state> <value> <unit>
   state <state_name>
      next_state <next_state_name>
         <condition_and_action>
      end_next_state
   end_state
end_processor
```

### WSF_TRACK_PROCESSOR

```text
processor <name> WSF_TRACK_PROCESSOR
end_processor
```

---

## signature

### radar_signature

```text
radar_signature <name> WSF_RADAR_SIGNATURE
   constant <value> <unit>          # 固定 RCS，如 10 m^2
   # 或使用频率相关 RCS
end_radar_signature
```

### optical_signature

```text
optical_signature <name> WSF_OPTICAL_SIGNATURE
   intensity <value>
end_optical_signature
```

---

## 全局配置

```text
random_seed <value>                 # 随机种子
end_time <value> <unit>            # 仿真结束时间
frame_time <value> <unit>           # 全局帧时间
log_file <path>                     # 日志文件路径
output_rate default                  # 输出速率配置
   time <value> <unit> period <value> <unit>
end_output_rate
```

---

## include

```text
include weapons/lrsam.txt
include radars/lrsam_ttr.txt
```

---

## script

```text
script void <function_name>(<params>)
   <code>
end_script

script void <event_handler>(<event_params>)
   <code>
end_script
```

常用事件钩子：
- `WeaponTerminated(WsfWeaponEngagement aWeaponEngagement)`
- `PlatformKilled(WsfPlatform aPlatform)`
- `WeaponHit(WsfWeaponEngagement aWeaponEngagement, WsfPlatform aTargetPlatform)`
- `WeaponFired(WsfWeaponEngagement aWeaponEngagement, WsfTrack aTargetTrack)`
