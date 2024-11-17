# MLKit Apple Exp

## セットアップ

### 依存関係

```shell
uv add torch==2.3.0 diffusers transformers accelerate
```

```shell
uv add huggingface_hub[cli]
```

### huggingface CLI のログイン

https://huggingface.co/settings/tokens にアクセスし，トークンを発行する

-   参考 https://huggingface.co/docs/huggingface_hub/guides/cli

```shell
huggingface-cli login
```

`~/.cache/huggingface/token`にトークンが保存される
