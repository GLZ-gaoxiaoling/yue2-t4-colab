"""用官方 abc_tools 校验合并后的纯音乐谱是否可解析。"""
import sys

sys.path.insert(0, "/content/yue2-music/yue2-music/scripts")

import abc_tools

text = open("/content/cover/score/instrumental.abc", encoding="utf-8").read()
score = abc_tools.parse(text)
print("parsed OK")
print(abc_tools.report(score))
