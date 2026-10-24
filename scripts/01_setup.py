"""远端后台安装：clone YuE2 + 按官方钉死版本安装依赖。

后台执行，不阻塞 colab exec；日志写在 /content/setup.log。
安装完成后顺手卸载 torchvision / torchaudio：它们是为 Colab 预装的 torch 2.11
编译的，YuE2 钉死 torch 2.10，留着会让 transformers 导入直接失败。
"""
import subprocess

CMD = (
    "set -x; cd /content && rm -rf YuE && "
    "git clone --depth 1 https://github.com/multimodal-art-projection/YuE.git && "
    "cd /content/YuE && python -m pip install --upgrade pip && "
    "python -m pip install . && "
    "python -m pip uninstall -y torchvision torchaudio"
)

with open("/content/setup.log", "w") as log:
    proc = subprocess.Popen(
        ["bash", "-lc", CMD],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

print("setup started, pid =", proc.pid)
print("log: /content/setup.log  （用 tail -f /content/setup.log 观察，结束时看 yue2-infer）")
