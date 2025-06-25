# 双重检测方法：位置 + 颜色直方图

## 概述

`car_detector.py` 现在使用双重检测方法来更准确地判断车辆是否在移动：

1. **几何位置检测**：基于边界框中心点的欧几里得距离
2. **颜色直方图检测**：基于HSV颜色空间的直方图相似性

## 工作原理

### 1. 几何位置检测
```python
# 计算边界框中心点
new_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
last_center = ((last_pos[0] + last_pos[2]) // 2, (last_pos[1] + last_pos[3]) // 2)

# 计算欧几里得距离
position_distance = np.sqrt((new_center[0] - last_center[0])**2 + (new_center[1] - last_center[1])**2)
position_stationary = position_distance <= POSITION_THRESHOLD  # 20像素
```

### 2. 颜色直方图检测
```python
# 提取ROI并转换为HSV
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# 计算H和S通道的直方图（忽略V通道以抵抗光照变化）
hist = cv2.calcHist([hsv_roi], [0, 1], None, [32, 32], [0, 180, 0, 256])

# 使用相关性方法比较直方图
similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
histogram_stationary = similarity >= HISTOGRAM_THRESHOLD  # 0.85
```

### 3. 组合决策
```python
# 两种方法都必须同意车辆静止
combined_stationary = position_stationary and histogram_stationary
```

## 配置参数

```python
# 位置检测参数
POSITION_THRESHOLD = 20  # 像素，最大移动距离

# 颜色直方图参数
HISTOGRAM_THRESHOLD = 0.85  # 相似性阈值
HISTOGRAM_BINS = 32  # 直方图bin数量
HISTOGRAM_METHOD = cv2.HISTCMP_CORREL  # 比较方法
```

## 优势

### 1. 更高的准确性
- **位置检测**：检测明显的物理移动
- **颜色检测**：检测车辆外观变化（如车门开关、人员进出）

### 2. 更强的鲁棒性
- **抗抖动**：边界框轻微抖动不会误判为移动
- **抗光照**：使用HSV颜色空间，对光照变化不敏感
- **抗阴影**：忽略V通道，减少阴影影响

### 3. 更稳定的检测
- **双重验证**：两种方法必须同时满足
- **减少误报**：降低因检测噪声导致的误判

## 可视化信息

在视频输出中，您可以看到详细的检测信息：

```
Car #1: 0.95 | Stationary: 15/30 | Pos:5px Hist:0.92
```

- `Pos:5px`：位置移动距离（像素）
- `Hist:0.92`：颜色直方图相似性（0-1）

## 测试

运行测试脚本验证双重检测方法：

```bash
python test_dual_detection.py
```

## 性能考虑

- **计算开销**：颜色直方图计算增加了少量CPU开销
- **内存使用**：每个车辆需要存储一个颜色直方图
- **精度提升**：显著提高了停车检测的准确性

## 故障排除

如果遇到问题：

1. **检查直方图计算**：确保ROI区域有效
2. **调整阈值**：根据实际场景调整 `HISTOGRAM_THRESHOLD`
3. **查看调试信息**：使用 `get_detection_method_info()` 方法 