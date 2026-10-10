# weapon-mapping.md — COA 武器名 → AFSIM weapon_type 映射表

> Agent 在 COA→AFSIM 转化时使用此表将武器名称映射为 AFSIM weapon_type 定义参数。

## 映射规则

```
COA 武器描述 → 提取武器型号 → 查下表 → 生成 AFSIM weapon_type 定义
```

## 空空导弹

| COA 武器名 | AFSIM name | mover | speed | max_range | seeker_range | warhead | 备注 |
|-----------|-----------|-------|-------|-----------|-------------|---------|------|
| PL-15 | pl_15 | WSF_GUIDED_MOVER | 1200 m/s | 200 km | 30 km | 30 kg | 超视距主动雷达制导 |
| PL-10 | pl_10 | WSF_GUIDED_MOVER | 900 m/s | 30 km | 15 km | 20 kg | 近距红外成像制导 |
| PL-12 | pl_12 | WSF_GUIDED_MOVER | 1100 m/s | 100 km | 25 km | 25 kg | 中距半主动/主动 |
| AIM-120D | aim_120d | WSF_GUIDED_MOVER | 1200 m/s | 180 km | 30 km | 22 kg | AMRAAM |
| AIM-9X | aim_9x | WSF_GUIDED_MOVER | 900 m/s | 35 km | 18 km | 9 kg | Sidewinder |
| R-77-1 | r77_1 | WSF_GUIDED_MOVER | 1100 m/s | 110 km | 25 km | 22 kg | 俄制中距主动 |
| R-73 | r73 | WSF_GUIDED_MOVER | 800 m/s | 30 km | 15 km | 8 kg | 俄制近距红外 |
| R-37M | r37m | WSF_GUIDED_MOVER | 1500 m/s | 300 km | 40 km | 60 kg | 超远程空空 |

## 空地/反辐射导弹

| COA 武器名 | AFSIM name | mover | speed | max_range | seeker_range | warhead | 备注 |
|-----------|-----------|-------|-------|-----------|-------------|---------|------|
| 鹰击-91 / YJ-91 | yj_91 | WSF_GUIDED_MOVER | 900 m/s | 150 km | 30 km | 80 kg | 反辐射导弹 |
| AGM-88 HARM | agm_88 | WSF_GUIDED_MOVER | 900 m/s | 150 km | 40 km | 68 kg | 美制反辐射 |
| AGM-158 JASSM | agm_158 | WSF_GUIDED_MOVER | 280 m/s | 370 km | 0 km | 450 kg | 隐身巡航 (无 seeker) |
| 精确制导炸弹 | guided_bomb | WSF_GUIDED_MOVER | 300 m/s | 70 km | 0 km | 250 kg | 卫星/INS 制导 |
| LD-3000 | ld_3000 | WSF_GUIDED_MOVER | 350 m/s | 60 km | 0 km | 200 kg | 激光制导炸弹 |

## 地空导弹

| COA 武器名 | AFSIM name | mover | speed | max_range | seeker_range | warhead | 备注 |
|-----------|-----------|-------|-------|-----------|-------------|---------|------|
| 48N6E3 (S-400) | 48n6e3 | WSF_GUIDED_MOVER | 1800 m/s | 400 km | 50 km | 150 kg | S-400 远程弹 |
| 40N6 (S-400) | 40n6 | WSF_GUIDED_MOVER | 2200 m/s | 400 km | 60 km | 180 kg | S-400 超远程弹 |
| HQ-22 | hq22_missile | WSF_GUIDED_MOVER | 1200 m/s | 170 km | 30 km | 100 kg | 红旗-22 中程 |
| HQ-9B | hq9b | WSF_GUIDED_MOVER | 1500 m/s | 250 km | 40 km | 130 kg | 红旗-9B 远程 |
| MIM-104 Patriot | mim_104 | WSF_GUIDED_MOVER | 1700 m/s | 160 km | 35 km | 90 kg | 爱国者 |
| THAAD | thaad | WSF_GUIDED_MOVER | 2800 m/s | 200 km | 100 km | 0 kg | 动能拦截 (无战斗部) |

## 反舰导弹

| COA 武器名 | AFSIM name | mover | speed | max_range | seeker_range | warhead | 备注 |
|-----------|-----------|-------|-------|-----------|-------------|---------|------|
| 鹰击-12 / YJ-12 | yj_12 | WSF_GUIDED_MOVER | 800 m/s | 400 km | 40 km | 300 kg | 超音速反舰 |
| 鹰击-83 / YJ-83 | yj_83 | WSF_GUIDED_MOVER | 300 m/s | 180 km | 20 km | 165 kg | 亚音速反舰 |
| Harpoon | harpoon | WSF_GUIDED_MOVER | 280 m/s | 130 km | 20 km | 220 kg | 美制反舰 |

## AFSIM weapon_type 模板

```text
weapon_type {{NAME}} WSF_GUIDED_MOVER
   speed {{SPEED}}
   maximum_range {{MAX_RANGE}}
   seeker WSF_GEOMETRIC_SENSOR
      receive_only
      frame_time 0.1 sec
      maximum_range {{SEEKER_RANGE}}
   end_seeker
   warhead {{WARHEAD}}
end_weapon_type
```

### 特殊处理

- **无 seeker 武器** (制导炸弹/巡航导弹): 省略 seeker 块
- **动能拦截弹** (THAAD): warhead 设为 0 kg
- **多模 seeker**: 可叠加多个 seeker 块 (雷达 + 红外)

## COA 武器名识别正则

```python
WEAPON_PATTERNS = [
    r"PL-(\d+)",           # PL-15, PL-10
    r"鹰击-(\d+)",          # 鹰击-91
    r"YJ-(\d+)",           # YJ-91
    r"R-(\d+)",            # R-77, R-73
    r"48N6\w*",            # 48N6E3
    r"HQ-(\d+)",           # HQ-22, HQ-9
    r"AIM-(\d+\w*)",       # AIM-120D, AIM-9X
    r"AGM-(\d+\w*)",       # AGM-88
    r"精确制导炸弹",         # 通用精确制导
    r"制导滑翔弹",           # 制导滑翔弹
]
```
