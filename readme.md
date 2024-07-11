# 使用代码前请先安装依赖
使用本项目前，请确保通过以下命令配置虚拟环境和安装所需依赖包：
```shell
pip install vietualenv
virtualenv venv
cd ./venv/Scripts
activate
cd ../..
pip install -r requirements.txt
```
# 环境要求
python == 3.10.11

# 目录结构：
本项目的目录结构组织如下：/
├── datas： 存放数据
├── general： 存放通用工具：目前有hsv提取工具
├── vision_detection： 视觉核心代码
├── axes_transfer：空间变换核心代码

# 快速开始
```shell
视觉检测：python main.py
hsv校准：python ./general/hsv_calibration.py
空间变换矩阵求解：python ./axes_transfer/axes_transfer.py
```