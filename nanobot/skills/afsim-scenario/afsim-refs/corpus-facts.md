# AFSIM 语法事实（官方 demo 语料实测）

> 数据来源：AFSIM 2.9 官方 `examples/demos` 语料的本地快照
> 4749 个 `.txt`、1121 个目录、40+ 领域（acoustic / ballistic / cyber / electronic_warfare /
> iads / satellite_demos / six_dof / space_operations …）。
>
> **本文件的权威性高于 `quick-ref.md` 和 `errors-ref.md`。** 那两份是二手总结，
> 已发现至少两处与官方语料矛盾（见下"文档勘误"）。判断 AFSIM 语法时，
> 顺序是：官方 demo 语料 > 官方文档 > 项目内二手总结。

## 1. 坐标格式：四种都合法，且经纬度可混用

实测 `position` 行分布：

| 形态 | 出现次数 | 例子 |
| --- | --- | --- |
| `d:m:s` / `d:m:s` | 26813 | `position 30:00:00n 120:00:00e` |
| 十进制度 / 十进制度 | **13280** | `position 1.283345n 103.859047e` |
| `d:m` / `d:m` | 268 | `position 40:30n 121:15w` |
| 整数度 / 整数度 | 140 | `position 30n 120e` |
| 混用（如 `d:m:s` / `d:m`） | ~90 | `position 27:30:00n 121:15w` |

秒可带小数：`00:00:36.035n`。

另有**直角 / ECI 位置**，没有半球标识而是"数值 + 长度单位"重复三组：

```text
position -1.2e+07 m -3.4e+06 m -5.0e+06 m
```

## 2. `side` 是任意标识符，不是 red/blue 枚举

实测取值（部分）：

```text
red 1235   blue 743   green 113   US 76    ORB 57   indigo 47
ITSO 34    CIS 28     ESA 19      pink 14  yellow 11
dark_purple 10   white 7   purple 7   light_blue 6
dark_blue 3   black 3   Red 3   violet 2   orange 2
```

区分大小写（`red` 与 `Red` 同时存在）。任何检查器都不该把 side 限制成红蓝两方。

## 3. 脚本块的块体是另一门语言

`script` / `execute` / `on_init` / `on_update` / `on_initialize` /
`on_new_execute` / `on_track_drop` / `on_new_fail` / `script_variables` /
`precondition` 这些块的**内容是 AFSIM 脚本（类 C）**，不是想定命令。

`aea_iads/processors/red_sam_battery_tactics.txt` 里有这样一行脚本：

```text
platform = sub;
```

按想定命令解析就会得出"未闭合的 platform 块"。**任何静态检查都必须整段跳过脚本体**，
只保留起止行用于块配对。

## 4. 块起始的歧义形态

| 写法 | 是不是块起始 | 说明 |
| --- | --- | --- |
| `platform NAME TYPE` | 是 | 平台实例 |
| `platform target` | **不是** | `track ... end_track` 块内的字段，只有两个词元 |
| `edit platform NAME` | 是 | 三个词元但带 `edit` 前缀 |
| `add sensor NAME TYPE` | 是 | `add` / `edit` 都会开块；`delete` 不会 |
| `event_output` | 是 | 裸形态 |
| `event_output file x.evt` | 是 | **带参数也是块起始** |
| `event_output true` | **不是** | `engage` 类输出块内的布尔字段 |
| `launch_computer MRM_LC end_launch_computer` | 是（单行自闭合） | 起止同行 |
| `radar_signature MRM_SIG` | 不是 | platform_type 内的引用 |
| `antenna_pattern SEEKER_ANTENNA` | 不是 | sensor 内的引用 |
| `independent_variable alt units ft` | 不是 | 6DOF 数据表的变量声明 |

`event_output` 的两种带参形态无法靠词元个数区分，只能靠"文档里还有没有未认领的
`end_event_output`"来判定（配额法）。

## 5. 块关键字的规模

官方语料出现 **120+ 种** `end_*`。任何固定白名单都不可能穷举，因此块配对检查
应当：只对登记过的块做严格判定，未登记的 `end_*` 静默跳过；允许引用形态的关键字
在配额耗尽时按字段处理。

## 6. 跨文件引用

想定普遍拆成多文件，`platform_type` 定义与 `platform` 实例常常分居两个文件，
由第三个主文件把它们一起 `include`。`satellite_demos`、`iads`、`space_operations`
下的平台清单文件**连 `include` 指令都没有**（由父文件负责引入）。

单文件验证器无法解析这种引用，"类型未定义"只能作为提示，不能判成错误——除非
该文件自己定义了 `platform_type` 且完全没有 `include`。

## 7. 单位

| 参数 | 语料实测 |
| --- | --- |
| altitude | `ft msl` 84、`ft` 101（air_to_air 全用 ft） |
| speed | `ft/s` 85、`m/s` 14 |
| maximum_range | `nmi`、`m`、`km` 都有 |

`m/s` 有语料先例，安全。`altitude ... m` 在 air_to_air 语料里没有先例，但 AFSIM
支持公制长度单位。

## 文档勘误

以下两条是项目内二手文档的**错误**说法，已被官方语料证伪：

1. **`quick-ref.md` 说行尾注释是错误**（"注释在独立行"）。
   实际：AFSIM 支持 `#` 与 `//` 行尾注释，air_to_air 语料里大量使用，如
   `weapon fox3 MEDIUM_RANGE_RADAR_MISSILE # fox 3 (MRM)`。

2. **`errors-ref.md` E005 把 `position 30.0N 120.0E` 列为非法**，容易被读成
   "十进制度非法"。实际十进制度用了 13280 次，完全合法；该反例真正的问题在
   **大写半球标识**，另一个反例 `position 30 120` 的问题在**缺半球标识**。

引用这两份文档时要交叉验证语料。
