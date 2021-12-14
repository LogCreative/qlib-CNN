# qlib-CNN

## 运行范例

直接运行范例代码，得到结果。

```
python task1/workflow_by_code.py
```

![](task1/pass.png)

## pytorch 实现 CNN

基础版本的CNN论文展现了下面的三层卷积网络。

![](img/oldcnn.png)

首先需要运行 [数据获取](task2/get_data.ipynb) 的代码。

[代码](task2/pytorch_cnn.py) 为 CNN 的实现情况，层由 `layers` 定义。运行时需要将其移入 qlib 包的 `qlib/contrib/model/` 中，训练框架参考 [pytorch_nn](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_nn.py)。

使用
```cmd
qrun task2/workflow.yaml
```
运行模型，该 yaml 文件参考了 [MLP](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/MLP/workflow_config_mlp_Alpha158.yaml)。

GTX 1050 Ti 的显存大小不足以运行该 CNN 模型代码，需要使用更高显存的 GPU。

### 分析

TCN 已经被[实现](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_tcn_ts.py)，参照对应的 [YAML 文件](task3/workflow_config_tcn_Alpha158.yaml)，将尝试直接运行该模型。由于本地算力不足，使用 Google Colab 运行之见 [代码](task3/workflow_tcn.ipynb)。

> `TSDatasetSampler` 近期有 API 变动，需要手动进行数据转换（未见 Issue 更新）。

![](img/report.png)

![](img/return.png)

![](img/scoreIC.png)

![](img/IC.png)