# COA方案：简化SEAD对抗测试

## 方案概述

本方案为简化版 SEAD（压制敌防空）对抗测试场景。蓝方出动电子战机和反辐射打击编队，
对红方岛礁上的 S-400 远程防空系统实施侦察-锁定-摧毁闭环。场景分三个阶段：
电子战诱敌、反辐射打击、效果评估。

## 实体清单

| unit_id | platform_code | quantity | side | role | callsign | base | weapons | notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B_AWACS | KJ-500 | 1 | BLUE | 预警指挥 | EAGLE_EYE | <BASE:B-Mainland> | 无 | 提供广域态势感知 |
| B_EW | J-16D | 2 | BLUE | 电子战 | GHOST_1, GHOST_2 | <BASE:B-Mainland> | 电子战吊舱 + 2xPL-15 | 频率隔离干扰 |
| B_SQ_01 | J-16 | 4 | BLUE | SEAD主攻 | THUNDER_11~14 | <BASE:B-Mainland> | 4x鹰击-91 + 4xPL-15 | 反辐射打击主力 |
| R_SAM_01 | S-400 | 1 | RED | 远程防空 | RED-SAM-1 | - | 4x48N6E3 | 岛礁东部高地 |
| R_SAM_02 | HQ-22 | 2 | RED | 中程防空 | RED-HQ22-1, RED-HQ22-2 | - | 6xHQ-22 | 岛礁北部 |
| R_RADAR | LR_RADAR | 1 | RED | 远程预警 | RED-EW | - | 无 | 岛礁制高点 |

## 地理参考

| ref_id | type | name | side | hint | notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| <BASE:B-Mainland> | BASE | 我方空军基地 | BLUE | 26:00:00n 119:30:00e | 全部平台起降基地 |
| <ORBIT:ORB-AWACS> | ORBIT | 预警机轨道 | BLUE | 距目标岛礁480km | 西侧偏南 |
| <WAYPOINT:WP-JAM> | WAYPOINT | 电子战压制点 | BLUE | 距目标岛礁400km | 西侧 |
| <WAYPOINT:WP-IP> | WAYPOINT | SEAD攻击进入点 | BLUE | 距目标岛礁150km | 西侧 |
| <ORBIT:ORB-HOLD> | ORBIT | 待机区 | BLUE | 距目标岛礁250km | 西侧 |
| <TARGET:T-S400> | TARGET | S-400系统 | RED | 27:30:00n 122:30:00e | 岛礁东部高地 |
| <TARGET:T-HQ22> | TARGET | 红旗-22系统 | RED | 27:35:00n 122:25:00e | 岛礁北部 |
| <TARGET:T-RADAR> | TARGET | 远程预警雷达 | RED | 27:32:00n 122:28:00e | 制高点 |

## COA矩阵

| 作战实体 | 阶段1: 电子战诱敌 *([触发→] ON_START)* | 阶段2: 反辐射打击 *([触发→] 辐射源确认)* | 阶段3: 效果评估 *([触发→] 打击完成)* |
| :--- | :--- | :--- | :--- |
| **预警指挥** (B_AWACS) [PLATFORM:KJ-500 x 1] | 1. TAKEOFF <BASE:B-Mainland> 2. MOVE_TO <ORBIT:ORB-AWACS> | 1. PATROL <ORBIT:ORB-AWACS> 2. SHARE_TRACK 引导打击编队 | 1. PATROL <ORBIT:ORB-AWACS> 2. 评估打击效果 |
| **电子战** (B_EW) [PLATFORM:J-16D x 2] | 1. TAKEOFF <BASE:B-Mainland> 2. MOVE_TO <WAYPOINT:WP-JAM> 3. JAM 对敌雷达频段干扰 | 1. JAM 持续压制 2. ESM侦收定位 | 1. ESM确认辐射源消失 2. EGRESS |
| **SEAD主攻** (B_SQ_01) [PLATFORM:J-16 x 4] | 1. TAKEOFF <BASE:B-Mainland> 2. MOVE_TO <ORBIT:ORB-HOLD> | 1. MOVE_TO <WAYPOINT:WP-IP> 2. ENGAGE 发射鹰击-91 | 1. LOITER <ORBIT:ORB-HOLD> 2. RTB <BASE:B-Mainland> |
| **S-400** (R_SAM_01) [PLATFORM:S-400 x 1] | 1. 开机搜索 2. 跟踪目标 | 1. 发射导弹 2. 持续照射 | 1. 被压制/摧毁 |
| **红旗-22** (R_SAM_02) [PLATFORM:HQ-22 x 2] | 1. 待命 | 1. 开机补防 | 1. 被评估 |
| **预警雷达** (R_RADAR) [PLATFORM:LR_RADAR x 1] | 1. 持续搜索 | 1. 持续搜索 | 1. 被压制 |

## 效果链

| 效果ID | 效果描述 | 衡量指标 | 达成实体 |
| :--- | :--- | :--- | :--- |
| E1 | 瘫痪S-400系统 | S-400火控雷达停止辐射>10分钟 | B_SQ_01, B_EW |
| E2 | 压制红旗-22 | 至少1套雷达关机 | B_SQ_01, B_EW |
| E3 | 瘫痪预警体系 | 远程预警雷达信号消失 | B_EW |

## 决策点

| 决策点ID | 所属阶段 | 触发条件 | 选项 |
| :--- | :--- | :--- | :--- |
| DP1 | Phase_1-2 | S-400全程静默 | ①威慑发射诱导 ②坐标攻击 |

## 关键风险

| 风险类别 | 风险描述 | 缓解措施 |
| :--- | :--- | :--- |
| 战术 | S-400静默对抗 | DP1梯度升级 |
| 频谱冲突 | 干扰覆盖数据链 | 频率隔离 |
