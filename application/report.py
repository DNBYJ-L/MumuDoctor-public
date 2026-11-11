#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自查报告 PDF 生成（Pillow 手绘，零额外依赖）

环境里没有 reportlab，且安装不了新包，因此用 Pillow 直接排版 A4 页面再导出 PDF。
代价是文字是位图、不可选中；换来的是在任何机器上都能出生、不依赖系统 PDF 工具链。

版式固定三部分：身体数据 / 分析结论 / 建议，外加页脚免责声明。
"""

import io
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

from checkin import CheckinReport, SeriesStats, followup_plan as trend_followup_plan
from selfcheck import (
    SelfCheckResult,
    department_advice,
    followup_plan,
    note_for,
)

DPI = 200
PAGE_W, PAGE_H = 1654, 2339  # A4 @200dpi
MARGIN_X = 120
MARGIN_TOP = 130
MARGIN_BOTTOM = 230  # 正文到此为止，页脚单独锚定在下方
CONTENT_W = PAGE_W - MARGIN_X * 2

INK = (27, 27, 24)
INK2 = (62, 62, 56)
MUTED = (133, 133, 122)
LINE = (229, 225, 213)
ACCENT = (28, 91, 74)
DANGER = (158, 53, 36)
WARN = (138, 90, 0)
WHITE = (255, 255, 255)

STATUS_COLOR = {
    "ok": ACCENT,
    "watch": WARN,
    "low": WARN,
    "high": DANGER,
    "urgent": DANGER,
}

_FONT_CANDIDATES = {
    # (路径, ttc 内的字面索引)。Noto Sans CJK 的 .ttc 里 0=JP、2=SC，
    # 不显式指定索引会拿到日文字面来渲染中文。
    "regular": [
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),
        (r"C:\Windows\Fonts\Deng.ttf", 0),
        (r"C:\Windows\Fonts\simhei.ttf", 0),
        (r"C:\Windows\Fonts\msyh.ttc", 0),
        ("/System/Library/Fonts/PingFang.ttc", 0),
    ],
    "bold": [
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 2),
        (r"C:\Windows\Fonts\Dengb.ttf", 0),
        (r"C:\Windows\Fonts\msyhbd.ttc", 0),
        (r"C:\Windows\Fonts\simhei.ttf", 0),
    ],
    "light": [
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),
        (r"C:\Windows\Fonts\Dengl.ttf", 0),
        (r"C:\Windows\Fonts\Deng.ttf", 0),
    ],
}


@lru_cache(maxsize=16)
def _font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    for path, index in _FONT_CANDIDATES.get(weight, _FONT_CANDIDATES["regular"]):
        if not Path(path).exists():
            continue
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    # 走到这里说明系统没有中文字体，中文会渲染成方块。
    # 服务器上执行：apt-get install -y fonts-noto-cjk
    return ImageFont.load_default()


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return 0x2E80 <= code <= 0x9FFF or 0xFF00 <= code <= 0xFFEF or 0x3000 <= code <= 0x303F


def _tokens(text: str) -> List[str]:
    """把句子切成可断行的单元：汉字逐字，西文按词。"""
    out: List[str] = []
    buf = ""
    for ch in text:
        if _is_cjk(ch) or ch in "（）【】、，。；：！？":
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
        elif ch == " ":
            if buf:
                out.append(buf)
                buf = ""
            out.append(" ")
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: float) -> List[str]:
    lines: List[str] = []
    line = ""
    for token in _tokens(text):
        candidate = line + token
        if line and font.getlength(candidate) > max_w:
            lines.append(line.rstrip())
            line = "" if token == " " else token
        else:
            line = candidate
    if line.strip():
        lines.append(line.rstrip())
    return lines or [""]


class _Builder:
    """带分页的简易排版器。"""

    def __init__(self) -> None:
        self.pages: List[Image.Image] = []
        self.page = None
        self.draw = None
        self.y = MARGIN_TOP
        self.new_page()

    def new_page(self) -> None:
        self.page = Image.new("RGB", (PAGE_W, PAGE_H), WHITE)
        self.draw = ImageDraw.Draw(self.page)
        self.pages.append(self.page)
        self.y = MARGIN_TOP

    def ensure(self, height: float) -> None:
        if self.y + height > PAGE_H - MARGIN_BOTTOM:
            self.new_page()

    def text(self, text: str, font, color=INK, line_h: int = 50, indent: int = 0) -> None:
        for line in _wrap(text, font, CONTENT_W - indent):
            self.ensure(line_h)
            self.draw.text((MARGIN_X + indent, self.y), line, font=font, fill=color)
            self.y += line_h

    def gap(self, px: int) -> None:
        self.y += px

    def rule(self, color=LINE, width: int = 2) -> None:
        self.ensure(10)
        self.draw.line([(MARGIN_X, self.y), (PAGE_W - MARGIN_X, self.y)], fill=color, width=width)
        self.y += width


def _header(b: _Builder, generated_at: datetime, profile: str = "") -> None:
    b.draw.text((MARGIN_X, b.y), "木木医生", font=_font(26, "bold"), fill=ACCENT)
    b.draw.text(
        (MARGIN_X + 260, b.y + 4), "个人健康自查", font=_font(24), fill=MUTED
    )
    b.y += 70

    b.draw.text((MARGIN_X, b.y), "健康自查报告", font=_font(64, "bold"), fill=INK)
    b.y += 100

    meta = f"生成时间 {generated_at.strftime('%Y-%m-%d %H:%M')}"
    if profile:
        meta = f"{profile}　·　" + meta
    b.draw.text((MARGIN_X, b.y), meta, font=_font(24), fill=MUTED)
    b.y += 34
    b.draw.text((MARGIN_X, b.y), "本报告由 AI 依据自填数据生成，仅供参考，不构成诊断", font=_font(22), fill=MUTED)
    b.y += 44
    b.rule()
    b.gap(40)


def _banner(b: _Builder, result: SelfCheckResult) -> None:
    color = {"ok": ACCENT, "watch": WARN, "alert": DANGER, "urgent": DANGER}[result.level]
    label_font = _font(26)
    value_font = _font(34, "bold")
    note_font = _font(26)

    label = result.level_label
    note = ("请尽快就医：" + "、".join(result.urgent_items)) if result.urgent_items else ""

    value_w = b.draw.textlength(label, font=value_font)
    note_w = b.draw.textlength(note, font=note_font) if note else 0
    # 标题较长时说明文字另起一行，避免叠字
    inline = bool(note) and (34 + value_w + 36 + note_w <= CONTENT_W - 40)
    box_h = 96 if (inline or not note) else 142

    b.ensure(box_h + 40)
    b.draw.rectangle(
        [(MARGIN_X, b.y), (PAGE_W - MARGIN_X, b.y + box_h)], outline=color, width=2
    )
    b.draw.rectangle([(MARGIN_X, b.y), (MARGIN_X + 8, b.y + box_h)], fill=color)
    b.draw.text((MARGIN_X + 34, b.y + 20), "整体评估", font=label_font, fill=MUTED)
    b.draw.text((MARGIN_X + 34, b.y + 50), label, font=value_font, fill=color)

    if note:
        if inline:
            b.draw.text(
                (MARGIN_X + 34 + value_w + 36, b.y + 56), note, font=note_font, fill=color
            )
        else:
            b.draw.text((MARGIN_X + 34, b.y + 100), note, font=note_font, fill=color)

    b.y += box_h + 44


def _section_title(b: _Builder, index: str, title: str, keep: int = 150) -> None:
    """keep：标题后至少留出的高度，避免标题孤零零留在页底。"""
    b.ensure(keep)
    b.draw.text((MARGIN_X, b.y), index, font=_font(26, "bold"), fill=ACCENT)
    b.draw.text((MARGIN_X + 46, b.y), title, font=_font(36, "bold"), fill=INK)
    b.y += 62


def _table(b: _Builder, result: SelfCheckResult) -> None:
    cols = [
        ("项目", 360),
        ("数值", 360),
        ("参考范围", 420),
        ("状态", CONTENT_W - 360 - 360 - 420),
    ]
    xs = [MARGIN_X]
    for _, w in cols[:-1]:
        xs.append(xs[-1] + w)

    cell_font = _font(27)
    head_font = _font(25, "bold")
    line_h = 40

    def draw_row(cells: List[Tuple[str, tuple]], bold=False, shade=None) -> None:
        wrapped = [_wrap(c[0], head_font if bold else cell_font, cols[i][1] - 24) for i, c in enumerate(cells)]
        row_h = max(len(w) for w in wrapped) * line_h + 26
        b.ensure(row_h)

        if shade:
            b.draw.rectangle(
                [(MARGIN_X, b.y), (PAGE_W - MARGIN_X, b.y + row_h)], fill=shade
            )
        for i, lines in enumerate(wrapped):
            for j, line in enumerate(lines):
                b.draw.text(
                    (xs[i] + 12, b.y + 13 + j * line_h),
                    line,
                    font=head_font if bold else cell_font,
                    fill=cells[i][1],
                )
        b.y += row_h
        b.draw.line([(MARGIN_X, b.y), (PAGE_W - MARGIN_X, b.y)], fill=LINE, width=1)

    draw_row([(name, MUTED) for name, _ in cols], bold=True, shade=(247, 246, 241))

    for metric in result.metrics:
        draw_row(
            [
                (metric.label, INK),
                (metric.value, INK),
                (metric.reference, INK2),
                (metric.status_label, STATUS_COLOR.get(metric.status, INK)),
            ]
        )

    b.gap(46)


def _notes(b: _Builder, result: SelfCheckResult) -> None:
    """逐项解读：每个指标一句说明，指出为什么这个数值值得注意。"""
    for metric in result.metrics:
        b.ensure(100)
        label_font = _font(28, "bold")
        b.draw.text((MARGIN_X, b.y), metric.label, font=label_font, fill=INK)
        label_w = b.draw.textlength(metric.label, font=label_font)
        b.draw.text(
            (MARGIN_X + label_w + 18, b.y + 4),
            metric.status_label,
            font=_font(24),
            fill=STATUS_COLOR.get(metric.status, INK),
        )
        b.y += 42
        b.text(note_for(metric), _font(26), INK2, line_h=44)
        b.gap(20)


def _plan(b: _Builder, result: SelfCheckResult) -> None:
    """复测与就诊：先给科室建议，再按异常项给复查计划。"""
    advice = department_advice(result)
    if advice:
        b.text(advice, _font(28, "bold"), DANGER, line_h=48)
        b.gap(18)

    for title, body in followup_plan(result):
        b.ensure(50)
        b.draw.text((MARGIN_X, b.y), "· " + title, font=_font(27, "bold"), fill=INK2)
        b.y += 40
        b.text(body, _font(27), INK2, line_h=46, indent=26)
        b.gap(16)


def _footer(pages: List[Image.Image]) -> None:
    total = len(pages)
    for i, page in enumerate(pages, start=1):
        draw = ImageDraw.Draw(page)
        y = PAGE_H - 110
        draw.line([(MARGIN_X, y - 26), (PAGE_W - MARGIN_X, y - 26)], fill=LINE, width=1)
        draw.text(
            (MARGIN_X, y),
            "本报告依据用户自填数据与常见参考区间生成，不构成医学诊断，也不能替代执业医师的当面诊疗。",
            font=_font(22),
            fill=MUTED,
        )
        draw.text(
            (MARGIN_X, y + 34),
            "如出现胸痛、呼吸困难、意识改变、大出血等情况，请立即拨打 120。",
            font=_font(22),
            fill=MUTED,
        )
        page_text = f"第 {i} / {total} 页"
        draw.text(
            (PAGE_W - MARGIN_X - draw.textlength(page_text, font=_font(22)), y),
            page_text,
            font=_font(22),
            fill=MUTED,
        )


def render_pages(
    result: SelfCheckResult,
    conclusion: str,
    suggestions: List[str],
    generated_at: datetime = None,
    profile: str = "",
) -> List[Image.Image]:
    """排好版但还没导出，返回页面图像列表（便于预览与测试）。"""
    generated_at = generated_at or datetime.now()
    b = _Builder()

    _header(b, generated_at, profile)
    _banner(b, result)

    _section_title(b, "一", "身体数据")
    _table(b, result)

    _section_title(b, "二", "逐项解读")
    _notes(b, result)
    b.gap(26)

    _section_title(b, "三", "分析结论")
    b.text(conclusion or "（未生成分析结论）", _font(30), INK2, line_h=54)
    b.gap(46)

    _section_title(b, "四", "建议", keep=330)
    if suggestions:
        for i, item in enumerate(suggestions, start=1):
            b.ensure(52)
            b.draw.text((MARGIN_X, b.y), f"{i:02d}", font=_font(30, "bold"), fill=ACCENT)
            for line in _wrap(item, _font(30), CONTENT_W - 70):
                b.ensure(52)
                b.draw.text((MARGIN_X + 70, b.y), line, font=_font(30), fill=INK2)
                b.y += 52
            b.gap(12)
    else:
        b.text("（暂无建议）", _font(30), MUTED, line_h=54)

    b.gap(46)
    _section_title(b, "五", "复测与就诊")
    _plan(b, result)

    _footer(b.pages)
    return b.pages


def render_pdf(
    result: SelfCheckResult,
    conclusion: str,
    suggestions: List[str],
    generated_at: datetime = None,
    profile: str = "",
) -> bytes:
    """生成 PDF 字节流。"""
    pages = render_pages(result, conclusion, suggestions, generated_at, profile)
    buf = io.BytesIO()
    pages[0].save(
        buf, format="PDF", resolution=DPI, save_all=True, append_images=pages[1:]
    )
    return buf.getvalue()


# ===================== 打卡数据趋势报告 =====================
# 与自查报告共用同一套排版器（_Builder）与字体，只是内容换成"一段时间里的趋势"。
# 图是 Pillow 手绘的折线：不引入 matplotlib，服务器上少一个依赖，也不用装中文字体之外的包。


def _trend_header(b: _Builder, generated_at: datetime, report: CheckinReport, profile: str = "") -> None:
    b.draw.text((MARGIN_X, b.y), "木木医生", font=_font(26, "bold"), fill=ACCENT)
    b.draw.text((MARGIN_X + 260, b.y + 4), "打卡数据趋势", font=_font(24), fill=MUTED)
    b.y += 70

    b.draw.text((MARGIN_X, b.y), "健康数据趋势报告", font=_font(64, "bold"), fill=INK)
    b.y += 100

    meta = f"{report.window_label}　·　打卡 {report.records} 次　·　覆盖 {report.days} 天"
    if profile:
        meta = f"{profile}　·　" + meta
    b.draw.text((MARGIN_X, b.y), meta, font=_font(24), fill=MUTED)
    b.y += 34
    b.draw.text(
        (MARGIN_X, b.y),
        f"生成时间 {generated_at.strftime('%Y-%m-%d %H:%M')}　·　指标：{report.groups_label}",
        font=_font(22),
        fill=MUTED,
    )
    b.y += 34
    b.draw.text(
        (MARGIN_X, b.y),
        "本报告依据用户打卡数据与常见参考区间生成，仅供参考，不构成诊断",
        font=_font(22),
        fill=MUTED,
    )
    b.y += 44
    b.rule()
    b.gap(40)


def _trend_banner(b: _Builder, report: CheckinReport) -> None:
    color = DANGER if report.urgent else ACCENT
    box_h = 96
    b.ensure(box_h + 40)
    b.draw.rectangle(
        [(MARGIN_X, b.y), (PAGE_W - MARGIN_X, b.y + box_h)], outline=color, width=2
    )
    b.draw.rectangle([(MARGIN_X, b.y), (MARGIN_X + 8, b.y + box_h)], fill=color)
    b.draw.text((MARGIN_X + 34, b.y + 20), "分析范围", font=_font(26), fill=MUTED)
    b.draw.text(
        (MARGIN_X + 34, b.y + 50),
        f"{report.window_label}　·　{report.groups_label}",
        font=_font(32, "bold"),
        fill=color,
    )
    if report.urgent:
        b.draw.text(
            (MARGIN_X + 34 + 520, b.y + 56),
            f"最近一次已越过警报线：{'、'.join(report.urgent)}",
            font=_font(24),
            fill=color,
        )
    b.y += box_h + 44


def _grid(
    b: _Builder,
    headers: List[str],
    rows: List[List[str]],
    widths: List[int],
    shade_first: bool = True,
) -> None:
    """简易表格：列宽固定，单元格自动换行。"""
    xs = [MARGIN_X]
    for width in widths[:-1]:
        xs.append(xs[-1] + width)

    head_font = _font(23, "bold")
    cell_font = _font(23)
    line_h = 36

    def draw_row(cells: List[str], bold: bool = False, shade=None) -> None:
        font = head_font if bold else cell_font
        wrapped = [
            _wrap(str(cell), font, widths[index] - 20) for index, cell in enumerate(cells)
        ]
        row_h = max(len(lines) for lines in wrapped) * line_h + 22
        b.ensure(row_h)
        if shade:
            b.draw.rectangle(
                [(MARGIN_X, b.y), (PAGE_W - MARGIN_X, b.y + row_h)], fill=shade
            )
        for index, lines in enumerate(wrapped):
            color = MUTED if bold else INK2
            for line_no, line in enumerate(lines):
                b.draw.text(
                    (xs[index] + 8, b.y + 11 + line_no * line_h),
                    line,
                    font=font,
                    fill=color,
                )
        b.y += row_h
        b.draw.line([(MARGIN_X, b.y), (PAGE_W - MARGIN_X, b.y)], fill=LINE, width=1)

    draw_row(headers, bold=True, shade=(247, 246, 241) if shade_first else None)
    for row in rows:
        draw_row(row)
    b.gap(34)


def _trend_table(b: _Builder, report: CheckinReport) -> None:
    widths = [270, 110, 200, 290, 140, CONTENT_W - 270 - 110 - 200 - 290 - 140]

    def unit_of(item: SeriesStats) -> str:
        # BMI 的"单位"就是 BMI 本身，写成 "22.1 BMI" 会重复
        return "" if item.unit == "BMI" else f" {item.unit}"

    rows = [
        [
            item.label,
            f"{item.count} 次",
            f"{item.fmt(item.mean)}{unit_of(item)}",
            f"{item.fmt(item.minimum)} – {item.fmt(item.maximum)}",
            f"{item.change_pct:+.1f}%",
            item.reference,
        ]
        for item in report.series
    ]
    if not rows:
        b.text("所选指标在统计范围内没有记录。", _font(26), MUTED)
        return
    _grid(b, ["指标", "记录", "平均", "最低 – 最高", "变化", "参考范围"], rows, widths)


def _spark(b: _Builder, item: SeriesStats, height: int = 300) -> None:
    """一条指标的折线图：纵轴是数值，横轴是时间（日期 + 时间）。"""
    b.ensure(height + 70)

    left = MARGIN_X + 100
    right = PAGE_W - MARGIN_X - 20
    top = b.y + 16
    bottom = b.y + height

    values = [value for _, value in item.points]
    low, high = min(values), max(values)
    if abs(high - low) < 1e-9:
        pad = max(abs(high) * 0.1, 1.0)
    else:
        pad = (high - low) * 0.18
    low -= pad
    high += pad

    def y_of(value: float) -> float:
        return bottom - (value - low) / (high - low) * (bottom - top)

    # 三条水平参考线 + 上下刻度
    for fraction in (0.0, 0.5, 1.0):
        y = top + (bottom - top) * fraction
        b.draw.line([(left, y), (right, y)], fill=LINE, width=1)
    b.draw.text((MARGIN_X, top - 6), f"{max(values):.{item.decimals}f}", font=_font(20), fill=MUTED)
    b.draw.text(
        (MARGIN_X, bottom - 14), f"{min(values):.{item.decimals}f}", font=_font(20), fill=MUTED
    )

    # 平均值单独画一条虚线，刻度对得准才不会误导
    mean_y = y_of(item.mean)
    for x in range(left, right - 12, 24):
        b.draw.line([(x, mean_y), (x + 12, mean_y)], fill=(185, 199, 191), width=2)
    b.draw.text(
        (MARGIN_X, mean_y - 12),
        f"{item.mean:.{item.decimals}f}",
        font=_font(20),
        fill=MUTED,
    )

    # 轴
    b.draw.line([(left, top), (left, bottom)], fill=INK2, width=2)
    b.draw.line([(left, bottom), (right, bottom)], fill=INK2, width=2)

    moments = [moment for moment, _ in item.points]
    span = (moments[-1] - moments[0]).total_seconds() or 1.0
    usable = right - left - 20
    coords = [
        (
            left + 10 + ((moment - moments[0]).total_seconds() / span) * usable,
            y_of(value),
        )
        for moment, value in item.points
    ]
    if len(coords) >= 2:
        b.draw.line(coords, fill=ACCENT, width=3, joint="curve")
    for x, y in coords:
        b.draw.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill=ACCENT)

    # 横轴只标首尾两个时间，避免挤在一起
    first_label = moments[0].strftime("%m-%d %H:%M")
    last_label = moments[-1].strftime("%m-%d %H:%M")
    b.draw.text((left, bottom + 10), first_label, font=_font(20), fill=MUTED)
    b.draw.text(
        (right - b.draw.textlength(last_label, font=_font(20)), bottom + 10),
        last_label,
        font=_font(20),
        fill=MUTED,
    )
    b.draw.text(
        (left, top - 28), f"纵轴：{item.unit}　横轴：时间", font=_font(20), fill=MUTED
    )

    b.y = bottom + 56


def _series_note(item: SeriesStats) -> str:
    if item.count == 1:
        return (
            f"只有 1 次记录（{item.fmt(item.last)} {item.unit}），还看不出趋势；"
            f"按相同条件多记录几次再对比。参考范围 {item.reference}。"
        )

    parts = [
        f"{item.count} 次记录，平均 {item.fmt(item.mean)} {item.unit}，"
        f"最低 {item.fmt(item.minimum)}、最高 {item.fmt(item.maximum)}"
    ]
    if abs(item.change_pct) < 3:
        parts.append("最近一次与首次相比基本持平")
    else:
        direction = "上升" if item.change_pct > 0 else "下降"
        parts.append(f"最近一次与首次相比{direction} {abs(item.change_pct):.1f}%")
    parts.append(item.trend)
    if item.abnormal:
        parts.append(f"其中 {item.abnormal} 次超出参考范围（{item.reference}）")
    if item.latest_at:
        parts.append(
            f"最近一次 {item.fmt(item.last)} {item.unit}（{item.latest_at.strftime('%Y-%m-%d %H:%M')}）"
        )
    return "；".join(parts) + "。"


def _trend_notes(b: _Builder, report: CheckinReport) -> None:
    if not report.series:
        b.text("所选指标没有可解读的记录。", _font(26), MUTED)
        return
    for item in report.series:
        b.ensure(110)
        b.draw.text((MARGIN_X, b.y), item.label, font=_font(28, "bold"), fill=INK)
        label_w = b.draw.textlength(item.label, font=_font(28, "bold"))
        color = DANGER if item.abnormal or item.trend == "呈上升趋势" else ACCENT
        b.draw.text(
            (MARGIN_X + label_w + 18, b.y + 4),
            f"{item.trend} · {item.count} 次",
            font=_font(22),
            fill=color,
        )
        b.y += 42
        b.text(_series_note(item), _font(25), INK2, line_h=43)
        b.gap(18)


def _trend_plan(b: _Builder, report: CheckinReport) -> None:
    for title, body in trend_followup_plan(report):
        b.ensure(50)
        b.draw.text((MARGIN_X, b.y), "· " + title, font=_font(27, "bold"), fill=INK2)
        b.y += 40
        b.text(body, _font(27), INK2, line_h=46, indent=26)
        b.gap(16)


def render_trend_pages(
    report: CheckinReport,
    generated_at: datetime = None,
    profile: str = "",
) -> List[Image.Image]:
    """排好版但还没导出，返回页面图像列表（便于预览与测试）。"""
    generated_at = generated_at or datetime.now()
    b = _Builder()

    _trend_header(b, generated_at, report, profile)
    _trend_banner(b, report)

    _section_title(b, "一", "数据概览")
    _trend_table(b, report)

    _section_title(b, "二", "趋势图")
    charted = [item for item in report.series if item.count >= 2]
    if not charted:
        b.text("记录少于两次，暂时画不出趋势；再打卡几次即可。", _font(26), MUTED)
    for item in charted:
        # 图表标题和整张图要留在同一页：300 的图 + 上下留白一起预留
        b.ensure(70 + 300 + 70)
        b.draw.text((MARGIN_X, b.y), item.label, font=_font(32, "bold"), fill=INK)
        if item.unit == "BMI":
            summary = f"{item.count} 次记录 · 最近 BMI {item.fmt(item.last)}"
        else:
            summary = f"{item.count} 次记录 · 最近 {item.fmt(item.last)} {item.unit}"
        b.draw.text((MARGIN_X + 240, b.y + 8), summary, font=_font(22), fill=MUTED)
        b.y += 46
        _spark(b, item)
        b.gap(34)

    _section_title(b, "三", "逐项解读")
    _trend_notes(b, report)
    b.gap(24)

    _section_title(b, "四", "分析结论")
    b.text(report.conclusion or "（未生成分析结论）", _font(29), INK2, line_h=52)
    b.gap(44)

    _section_title(b, "五", "建议", keep=300)
    if report.suggestions:
        for index, item in enumerate(report.suggestions, start=1):
            b.ensure(52)
            b.draw.text((MARGIN_X, b.y), f"{index:02d}", font=_font(30, "bold"), fill=ACCENT)
            for line in _wrap(item, _font(30), CONTENT_W - 70):
                b.ensure(52)
                b.draw.text((MARGIN_X + 70, b.y), line, font=_font(30), fill=INK2)
                b.y += 52
            b.gap(12)
    else:
        b.text("（暂无建议）", _font(30), MUTED, line_h=54)

    b.gap(44)
    _section_title(b, "六", "复测与就诊")
    _trend_plan(b, report)

    _footer(b.pages)
    return b.pages


def render_trend_pdf(
    report: CheckinReport,
    generated_at: datetime = None,
    profile: str = "",
) -> bytes:
    """生成打卡趋势报告的 PDF 字节流。"""
    pages = render_trend_pages(report, generated_at, profile)
    buf = io.BytesIO()
    pages[0].save(
        buf, format="PDF", resolution=DPI, save_all=True, append_images=pages[1:]
    )
    return buf.getvalue()
