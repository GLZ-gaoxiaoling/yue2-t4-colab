"""把 SheetSage2 的旋律谱（Vocal + Ins 双轨）合并成纯音乐单旋律谱。

规则对齐 YuE2 instrumental skill：Vocal 的音符全部移到 Ins，Vocal 只留休止；
同一小节两轨都有音符时 Vocal 优先（保留旋律），Vocal 只有休止的小节保留原 Ins 内容。
合并后按 4/4 补齐每小节时长，避免 abc_tools 的 meter 校验失败。
"""
import re
import sys
from fractions import Fraction
from pathlib import Path

NOTE = re.compile(r"[A-Ga-g]")
MULTI_REST = re.compile(r"^Z(\d*)$")
VOICE_DECL = re.compile(r"^V:\s*(Vocal|Ins)\b")
UNIT_TOKEN = re.compile(r"(\^+|_+|=)?([A-Ga-gzZx])([',]*)(\d+(?:/\d+)?)?(\.*)")
BAR_UNITS = Fraction(16)  # M:4/4 + L:1/16


def expand_multi_rest(token):
    """把 Z / Z4 这类多小节休止展开成独立小节。"""
    match = MULTI_REST.match(token.strip())
    if match:
        return ["z16"] * int(match.group(1) or 1)
    return [token]


def strip_chords(body):
    return re.sub(r'"[^"]*"', "", body)


def has_notes(body):
    return bool(NOTE.search(strip_chords(body)))


def bar_units(body):
    """按 L:1/16 计算小节总时长（单位：十六分音符）。"""
    total = Fraction(0)
    for match in UNIT_TOKEN.finditer(strip_chords(body)):
        pitch, digits, dots = match.group(2), match.group(4), match.group(5)
        if pitch in {"Z"}:
            continue
        length = Fraction(digits) if digits else Fraction(1)
        if dots:
            length *= 1 + sum(Fraction(1, 2 ** i) for i in range(1, len(dots) + 1))
        total += length
    return total


def render_text(value):
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def fit_bar(body):
    """把小节补到 4/4（不足补休止，超长原样保留）。"""
    units = bar_units(body)
    if units < BAR_UNITS:
        return body + "z" + render_text(BAR_UNITS - units)
    return body


def parse_blocks(lines):
    header, blocks, comment, voice, bars = [], [], None, None, []
    declared = set()

    def flush():
        nonlocal voice, bars, comment
        if voice and bars:
            blocks.append((comment, voice, bars))
        comment, voice, bars = None, None, []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("%"):
            flush()
            comment = line
            continue
        match = VOICE_DECL.match(line)
        if match:
            if line not in declared:       # 声部声明属于头部，原样保留
                declared.add(line)
                header.append(line)
                continue
            flush()
            voice = match.group(1)
            continue
        if line.startswith("V:"):
            flush()
            voice = None
            continue
        if voice is None:
            header.append(line)
            continue
        for token in [part for part in line.split("|") if part]:
            bars.extend(expand_multi_rest(token))
    flush()
    return header, blocks


def main(source, target):
    lines = Path(source).read_text(encoding="utf-8").splitlines()
    header, blocks = parse_blocks(lines)

    pairs, index = [], 0
    while index < len(blocks):
        comment, voice, bars = blocks[index]
        if voice == "Vocal" and index + 1 < len(blocks) and blocks[index + 1][1] == "Ins":
            pairs.append((comment, bars, blocks[index + 1][2]))
            index += 2
        else:
            pairs.append((comment, bars if voice == "Vocal" else [],
                          bars if voice == "Ins" else []))
            index += 1

    out = list(header)
    moved = kept = 0
    for comment, vocal_bars, ins_bars in pairs:
        size = max(len(vocal_bars), len(ins_bars))
        vocal_bars = vocal_bars + ["z16"] * (size - len(vocal_bars))
        ins_bars = ins_bars + ["z16"] * (size - len(ins_bars))

        new_vocal, new_ins = [], []
        for vocal_bar, ins_bar in zip(vocal_bars, ins_bars):
            if has_notes(vocal_bar):
                units = bar_units(vocal_bar)
                new_ins.append(fit_bar(strip_chords(vocal_bar)))
                moved += 1
            else:
                units = bar_units(vocal_bar) or BAR_UNITS
                new_ins.append(fit_bar(strip_chords(ins_bar)) if has_notes(ins_bar) else "z16")
                kept += 1
            new_vocal.append("z16")

        if comment:
            out.append(comment)
        out.append("V: Vocal")
        out.append("|".join(new_vocal) + "|")
        out.append("V: Ins")
        out.append("|".join(new_ins) + "|")

    Path(target).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"bars: vocal-moved={moved} ins-kept={kept} total={moved + kept}")

    check = Path(target).read_text(encoding="utf-8")
    assert "V: Vocal" in check and "V: Ins" in check, "缺少双声部"
    body = [line for line in check.splitlines() if line and not line.startswith(("X:", "T:", "M:", "L:", "Q:", "V:", "%", "K:"))]
    assert not any('"' in line for line in body), "小节里残留和弦符号"
    print(f"wrote {target}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
