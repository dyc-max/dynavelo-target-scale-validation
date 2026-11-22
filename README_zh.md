# DynaVelo 靶标尺度诊断与验证代码包

作者：DuanYuChen。自有代码MIT；Zenodo中的自有权重与结果CC BY 4.0。

本包包含实际训练版本匹配的脚本、敏感度与归一化代码、GRNBoost2及外部
差异分析代码、环境记录、源数据字典、校验工具及四图绘制入口。

[Zenodo 1.1.0](https://doi.org/10.5281/zenodo.24437365) 提供模型权重、完整矩阵和结果。
[S11 复现包](https://github.com/dyc-max/dynavelo-target-scale-validation/releases/download/v1.1.0/DynaVelo_Section_S11_Reproducibility.zip) 提供脚本、Tables S11a–S11c 及机器可读结果。
`07_source_tables_and_scripts.zip` 收录冻结的图表源数据、补充结果表和数值溯源脚本。
`06_external_validation.zip` 由六个部分组成，请先按[传输说明](https://zenodo.org/records/24437365/files/EXTERNAL_VALIDATION_TRANSPORT.md?download=1) 重组，再校验归档。

快速使用、命令与数据范围见英文README。`archive/project`保留科学脚本原字节，
其中原Windows路径和历史环境依赖尚在，不能当作已测试的跨平台一键训练包。
打包期间仅进行文件复制、压缩、哈希、语法及元数据检查，未重跑科学分析。

主矩阵为192细胞聚合、每种子1,191×3,006。1,884细胞读出仅覆盖既有
20个RNA及8个motif输入，不虚构全群体全1,191输入的导数矩阵。
补充材料记录了 DC 靶集比较的范围和 T 细胞结果的原始数据坐标。
