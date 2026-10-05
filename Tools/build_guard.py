"""构建期安全检查：防止把开发机上的用户配置打进发布包。

源码运行时默认开启便携模式（FilePathConfig 的 _portable_requested），配置会写到
仓库根的 Resource/config.json，里面含全部 API key 与 AWS 密钥。而打包脚本用
--add-data=Resource:Resource 整目录复制，RuntimeSetup 又用 Resource/config.json
给新用户做种子配置 —— 也就是说，只要打包前本机跑过一次应用，维护者的全部密钥
就会进入公开的发布包，并被分发成所有新用户的默认配置。

.gitignore 只挡住了 CI（CI 从干净 checkout 构建），挡不住本地构建，所以这里在
两个打包脚本里都做硬失败。

接口刻意接收「项目根目录」而不是「Resource 目录」：两个打包脚本的 ROOT 类型并不
一致（pyinstall.py 走 os.path 得到 str，pyinstall_macos.py 走 pathlib 得到 Path），
让调用方自己做路径拼接就写出过 `ROOT / "Resource"` 这种 str/str 表达式，只在 CI
才炸。把拼接收进这里，调用方只传 ROOT，这类错误就不存在了。
"""

from pathlib import Path

# 便携模式配置及其派生文件（Config.py 的 .corrupt / .tmp 落盘路径）
FORBIDDEN_RESOURCE_FILES = (
    "config.json",
    "config.json.corrupt",
    "config.json.tmp",
)


def assert_no_user_config(project_root) -> None:
    """项目根目录下若存在用户配置就中止构建，并报出具体路径。"""
    resource_dir = Path(project_root) / "Resource"
    present = [
        resource_dir / name
        for name in FORBIDDEN_RESOURCE_FILES
        if (resource_dir / name).exists()
    ]
    if not present:
        return

    listing = "\n".join(f"  - {path}" for path in present)
    raise SystemExit(
        "Refusing to build: user configuration found inside the packaged resource "
        "directory.\n"
        f"{listing}\n"
        "This file holds API keys and would be shipped to every user, and seeded as "
        "their default configuration. Move it aside (or delete it) and build again. "
        "It is gitignored, so CI builds are unaffected - this only bites local builds."
    )
