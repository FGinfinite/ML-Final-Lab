# 导出环境

## conda

```shell
conda list -e > requirements.txt
```
## pip

```shell
pip freeze > requirements.txt
```

# 生成环境

```shell
conda create --name <env> --file requirements.txt
```

# 安装环境

## conda

```shell
conda list -e > requirements.txt
```

## pip

```shell
pip install -r requirements.txt
```