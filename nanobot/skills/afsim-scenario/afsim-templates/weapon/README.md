# 武器与特征模板

本目录存放 AFSIM 武器和目标特征模板。所有模板均从官方 Demo 交叉验证。

## 武器模板

| 文件 | 用途 | 验证来源 |
|------|------|----------|
| `signature.txt` | radar_signature / optical_signature | base_types.md, engage.md |

## 内置武器类型

`LRSAM`, `SRSAM`, `AAM`, `ASM`, `BOMB`, `TORPEDO` 等

## signature 格式速查

| 类型 | 格式 | 备注 |
|------|------|------|
| 固定 RCS | `constant 1.0 m^2` | 最常用 |
| 频率依赖 | `plt_file xxx.txt` | 使用 PLT 文件 |
| 双基地 | `bistatic_signature` 块 | 双基地雷达 |
| 固定光学 | `constant 1000 m^2` | 无基类型 |
| 光学期望 | `WSF_OPTICAL_SIGNATURE` + `intensity` | 有基类型 |
