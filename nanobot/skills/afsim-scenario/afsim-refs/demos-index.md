# AFSIM Demo 速查表

> 按功能分类的 60+ 官方 Demo 索引。复制代码时请保持格式不变。

---

## 搜索关键词速查

| 需求 | 搜索关键词 | 查看文件 |
|------|-----------|----------|
| 飞机飞行 | platform_type, WSF_AIR_MOVER, route, position | engage, simple_scenario, air_to_air |
| 导弹/空战 | WSF_GUIDED_MOVER, missile, weapon | engage, air_to_air, hellfire |
| 传感器 | WSF_GEOMETRIC_SENSOR, WSF_RADAR_SENSOR, sensor | sensor_demos, iads, ciws |
| 雷达 | radar_signature, WSF_RADAR_SIGNATURE, RCS | signature_demos, radar, sensor |
| 舰船 | WSF_SURFACE_MOVER, ship, carrier | ship_ad, acoustic |
| 地面/武器 | WSF_GROUND_MOVER, launcher, SAM | launcher, fires, tbm_demos |
| 脚本逻辑 | processor, WSF_TASK_PROCESSOR, script | script_demos, behavior_tree |
| 路径规划 | route, route_waypoint, bearing | route_finder_demos, traffic_demos |
| 空间/卫星 | WSF_SPACE_MOVER, satellite, orbit | space_operations, satellite_demos |
| 电子战 | electronic_warfare, EW, jam | electronic_warfare, comm |
| 通信网络 | comm_network, WSF_COMM_TRANSCEIVER | comm, iads_c2_demos |
| 行为树 | behavior_tree, state, next_state | behavior_tree, iads |
| 声学 | acoustic, sonar, WSF_ACOUSTIC_SENSOR | acoustic |
| 激光 | laser, designator, WSF_LASER_SENSOR | laser_designator |
| 天气/环境 | weather, atmosphere, terrain | simple_scenario, terrain_following |

---

## 按功能分类

### 飞机 / 飞行 / 空战

| Demo | 文件 | 行号 | 特点 |
|------|------|------|------|
| engage 最小原型 | engage项目汇总.md | 330-345 | 43行，最简单飞机原型 |
| air_to_air 空战 | air_to_air项目汇总.md | 全文 | 完整空战场景，含导弹 |
| hel 直升机 | hel项目汇总.md | 全文 | 直升机平台 |
| brawler 缠斗 | brawler项目汇总.md | 全文 | 空中格斗场景 |
| outer_air_battle 远距拦截 | outer_air_battle项目汇总.md | 全文 | 超视距空战 |

### 导弹 / 武器 / 打击

| Demo | 文件 | 行号 | 特点 |
|------|------|------|------|
| hellfire 地狱火导弹 | hellfire项目汇总.md | 全文 | 精确制导武器 |
| fires 火力打击 | fires项目汇总.md | 全文 | 地面火力 |
| tbm_demos 战术弹道导弹 | tbm_demos项目汇总.md | 全文 | TBM 攻防 |
| ballistic 弹道 | ballistic项目汇总.md | 全文 | 弹道导弹 |
| ballistic_missile_shootdown 拦截 | ballistic_missile_shootdown项目汇总.md | 全文 | 导弹拦截 |
| gun_engagement 航炮 | gun_engagement项目汇总.md | 全文 | 机炮格斗 |
| suppressor 压制 | suppressor项目汇总.md | 全文 | SEAD/压制 |

### 传感器 / 雷达

| Demo | 文件 | 行号 | 特点 |
|------|------|------|------|
| sensor_demos | sensor_demos项目汇总.md | 全文 | 传感器综合示例 |
| signature_demos 雷达散射 | signature_demos项目汇总.md | 全文 | RCS 定义 |
| radar_sensor | sensor_demos项目汇总.md | 全文 | 雷达传感器 |
| ciws 近防系统 | ciws项目汇总.md | 全文 | 末端防御 |
| heatmap 热成像 | heatmap项目汇总.md | 全文 | 传感器可视化 |

### 地面 / 防空 / IADS

| Demo | 文件 | 行号 | 特点 |
|------|------|------|------|
| iads 综合防空 | iads项目汇总.md | 全文 | 完整 IADS 系统 |
| iads_c2_demos 指挥控制 | iads_c2_demos项目汇总.md | 全文 | C2 架构 |
| launcher 发射车 | launcher项目汇总.md | 全文 | 地面发射平台 |
| orwaca_iads 防空 | orwaca_iads项目汇总.md | 全文 | 防空系统 |

### 舰船 / 海上

| Demo | 文件 | 行号 | 特点 |
|------|------|------|------|
| ship_ad 舰船 | ship_ad项目汇总.md | 全文 | 舰船平台 |
| acoustic 声学 | acoustic项目汇总.md | 全文 | 声纳/潜艇 |

### 空间 / 卫星 / 天基

| Demo | 文件 | 行号 | 特点 |
|------|------|------|------|
| satellite_demos 卫星 | satellite_demos项目汇总.md | 全文 | 卫星运行 |
| space_operations 空间作战 | space_operations项目汇总.md | 全文 | 空间对抗 |
| cislunar 地月空间 | cislunar项目汇总.md | 全文 | 深空场景 |

### 路由 / 运动 / 机动

| Demo | 文件 | 行号 | 特点 |
|------|------|------|------|
| route_finder_demos 路径规划 | route_finder_demos项目汇总.md | 全文 | 动态路径 |
| traffic_demos 交通流 | traffic_demos项目汇总.md | 全文 | 航路管理 |
| kinematic_mover 运动学 | kinematic_mover项目汇总.md | 全文 | 平台机动 |
| terrain_following 地形跟随 | terrain_following项目汇总.md | 全文 | 低空突防 |
| alv_routing  ALV | alv_routing项目汇总.md | 全文 | 自主导航 |

### 脚本 / 逻辑 / 行为树

| Demo | 文件 | 行号 | 特点 |
|------|------|------|------|
| script_demos | script_demos项目汇总.md | 全文 | 脚本综合示例 |
| behavior_tree | behavior_tree项目汇总.md | 全文 | 状态机/行为树 |
| example_scripts | example_scripts项目汇总.md | 全文 | 脚本范例 |

### 通信 / 电子战

| Demo | 文件 | 行号 | 特点 |
|------|------|------|------|
| comm 通信 | comm项目汇总.md | 全文 | 通信网络 |
| electronic_warfare 电子战 | electronic_warfare项目汇总.md | 全文 | ECM/ESM |
| exchange_proc 数据链 | exchange_proc项目汇总.md | 全文 | 数据交换 |

### 多分辨率 / 推演

| Demo | 文件 | 行号 | 特点 |
|------|------|------|------|
| multiresolution_demos | multiresolution_demos项目汇总.md | 全文 | 聚合解聚 |
| wargame 兵棋推演 | wargame项目汇总.md | 全文 | 高级推演 |

### 特效 / 可视化 / 调试

| Demo | 文件 | 行号 | 特点 |
|------|------|------|------|
| draw 绘图 | draw项目汇总.md | 全文 | 自定义绘制 |
| visual_part 可视化 | visual_part项目汇总.md | 全文 | 可视化效果 |
| sensor_plot 传感器绘图 | sensor_plot项目汇总.md | 全文 | 传感器显示 |
| coverage_demos 覆盖范围 | coverage_demos项目汇总.md | 全文 | 探测覆盖 |

### 其他

| Demo | 文件 | 行号 | 特点 |
|------|------|------|------|
| base_types 基础类型 | base_types项目汇总.md | 全文 | 基础类型定义 |
| base_types_nx 扩展类型 | base_types_nx项目汇总.md | 全文 | NX 扩展类型 |
| bearing_only 仅方位 | bearing_only项目汇总.md | 全文 | 仅测向传感器 |
| cyber 网络战 | cyber项目汇总.md | 全文 | 网络攻防 |
| distributed_operations 分布式 | distributed_operations项目汇总.md | 全文 | 分布式作战 |
| logistics 后勤 | logistics项目汇总.md | 全文 | 补给/后勤 |
| parachute 降落伞 | parachute项目汇总.md | 全文 | 空降 |
| p6dof 六自由度 | p6dof项目汇总.md | 全文 | 6DoF 物理 |
| six_dof 六自由度 | six_dof项目汇总.md | 全文 | 6DoF 运动 |
| six_dof_with_brawler | six_dof_with_brawler项目汇总.md | 全文 | 6DoF + 格斗 |
| shooter 射手 | shooter项目汇总.md | 全文 | 武器发射 |
| noise_cloud 噪声云 | noise_cloud项目汇总.md | 全文 | 干扰云 |
| swarm 蜂群 | swarm项目汇总.md | 全文 | 无人机蜂群 |
| sosm 太空作战 | sosm项目汇总.md | 全文 | 太空态势 |
| timeline 时间线 | timeline项目汇总.md | 全文 | 时间线事件 |
| timeline_lead 时间线引导 | timeline_lead项目汇总.md | 全文 | 事件触发 |
| oth_radar 超视距雷达 | oth_radar项目汇总.md | 全文 | OTH 雷达 |
| new_guidance 新型制导 | new_guidance项目汇总.md | 全文 | 先进制导 |
| l16_j11 L16/J11 | l16_j11项目汇总.md | 全文 | 特定平台 |
| chaff 箔条 | chaff项目汇总.md | 全文 | 干扰物 |

---

## Demo 语法验证记录（知识完善计划产出）

以下 Demo 已通过交叉验证，语法已整合到 skill 文档：

|| Demo | 验证内容 | 贡献 |
|------|------|--------|------|
| engage | 基础交战 | sensor 格式、processor 状态机、comm internal_link |
| sensor_demos | 传感器参数 | receive_only、transmit_only、antenna_pattern、scan_mode、maximum_range |
| base_types | 基础类型 | optical_signature 无基类型写法、radar_signature 多种格式 |
| script_demos | 脚本/observer | event 钩子（WeaponTerminated 等）、realtime、dis_interface |
| comm | 通信系统 | WSF_COMM_TRANSCEIVER、internal_link、transmitter/receiver 块 |
| satellite_demos | 卫星轨道 | space_mover 开普勒根数、start_date/end_time 全局配置 |

详细模板见 `templates/` 目录。
