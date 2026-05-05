# PyGeoNet app layer

把整个 `app` 文件夹放到 PyGeoNet 项目根目录下，目录结构应为：

```text
pygeonet-2/
  download.py
  normalize.py
  model.py
  search.py
  standard.py
  utils/
  app/
    app.py
    templates/
    static/
```

## 启动方式

推荐在 `pygeonet-2` 根目录执行：

```bash
pip install -r requirements.txt
pip install -r app/requirements-app.txt
python -m app.app
```

也可以执行：

```bash
python app/app.py
```

浏览器打开：

```text
http://127.0.0.1:5000/
```

健康检查：

```text
http://127.0.0.1:5000/health
```

## 本版修复

- 修复 `No module named 'utils.search'`。
- 原因是 app 层原来带有 `app/utils/`，当用 `python app/app.py` 启动时，Python 可能把 `app/utils` 错当成 PyGeoNet 根目录下的 `utils`，从而导致 `search.py` 里的 `from utils.search.tool import ...` 失败。
- 本版删除了 app 层的 `utils/` 包，把绘图辅助文件移动为 `app/plot_tool.py`，避免和 PyGeoNet 原函数包的 `utils/` 冲突。
- 同时在 `app.py` 中强制把 PyGeoNet 根目录放到 `sys.path` 最前面，并清理错误加载的 `utils` 包。

## 设计说明

- 本文件夹只作为 Flask 平台层。
- `download.py`、`normalize.py`、`model.py`、`search.py`、`standard.py` 等原始函数包不需要修改。
- 平台通过动态导入调用 PyGeoNet 根目录中的原始函数。
- 运行结果统一保存在 `app/runtime/outputs/` 下，并由页面接口自动打包下载。
