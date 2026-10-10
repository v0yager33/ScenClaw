# 处理器模板

本目录存放 AFSIM processor 模板。所有模板均从官方 Demo 交叉验证。

## 模板列表

| 文件 | 类型 | 用途 | 验证来源 |
|------|------|------|----------|
| `task_processor.txt` | WSF_TASK_PROCESSOR | 多状态机任务逻辑 | engage.md line 30-100 |
| `track_processor.txt` | WSF_TRACK_PROCESSOR | 航迹管理 | engage.md line 67 |

## 使用方式

1. 选择对应类型的模板文件
2. 复制到 `platform_type` 块内，替换 `<占位符>`
3. 调整状态名称和转换逻辑
4. 在 Warlock/Wizard 中验证

## 状态机开发流程

```
DETECTED → TRY_TO_ACQUIRE → TRY_TO_TRACK → TRY_TO_FIRE
  (探测)     (捕获目标)      (稳定跟踪)     (发射武器)
```

每个状态的 `evaluation_interval` 控制评估频率，`next_state` 定义转换条件。
