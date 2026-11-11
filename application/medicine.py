#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
家庭药箱：家庭组共享的药品 / 卫材库存，以及临期、缺量提醒。

三张表 + 一张联想词库（都由首次访问自动创建）：

    medicine_cabinet       药箱里的每一条药品 / 卫材
        group_id           归属的家庭组管理者 id：管理者是自己，成员是 owner，
                           所以同一个家庭组看到的是同一份药箱（见 family.group_owner_id）
        category           drug 药品 / supply 卫材
        name / manufacturer / efficacy
        spec_amount        一个包装里有多少（规格：5 颗/盒 → 5）
        unit               计数单位：颗 / 片 / 袋 / mL …
        package_unit       包装单位：盒 / 瓶 / 包 …
        packages           现有几个包装（5 盒 → 5）
        quantity           自动算出的总数 = spec_amount × packages（5 × 5 = 25 颗）
        production_date / shelf_life_months / expiry_date
                           生产日期 + 保质期（月）→ 自动算出到期日
    medicine_catalog       用户录入过的药名 / 厂商 / 功效 / 规格，按家庭组隔离，
                           下次新增时做联想输入（删掉药箱记录也不会丢）
    medicine_alerts        临期 / 过期 / 缺量提醒。同一条药同一种提醒只有一行，
                           状态变化时激活或失效
    medicine_alert_reads   每个成员各自的已读记录：同一个提醒会发到每个成员，
                           谁点「知道了」只影响他自己

提醒规则（页面与首页横幅共用）：
    临期：到期日只剩 EXPIRY_WARN_DAYS（14）天以内
    过期：已经过了到期日（比临期更急，也提醒清理）
    缺量：剩余数量 ≤ 半个规格单位（5 颗/盒 → 2.5 颗）

还没有家庭组时 group_id 就是用户自己，药箱照样可用；建立家庭组后自动共享。
"""

from __future__ import annotations

import calendar
import io
import logging
from contextlib import contextmanager
from datetime import date, datetime
from typing import Dict, Iterator, List, Optional, Tuple

import auth
import family

logger = logging.getLogger(__name__)

# 保质期只剩两周就算临期；数量剩下不到半个规格单位就算缺量
EXPIRY_WARN_DAYS = 14
LOW_STOCK_RATIO = 0.5

CATEGORY_DRUG = "drug"
CATEGORY_SUPPLY = "supply"
CATEGORIES = {CATEGORY_DRUG: "药品", CATEGORY_SUPPLY: "卫材"}

UNIT_SUGGESTIONS = (
    "颗", "片", "粒", "丸", "袋", "支", "瓶", "贴", "包",
    "mL", "g", "个", "只", "卷", "套", "抽",
)
PACKAGE_UNIT_SUGGESTIONS = ("盒", "瓶", "袋", "包", "板", "罐", "支", "卷", "个")

STATUS_OK = "ok"
STATUS_LOW = "low"
STATUS_EXPIRING = "expiring"
STATUS_EXPIRED = "expired"
STATUS_LABELS = {
    STATUS_OK: "正常",
    STATUS_LOW: "需补药",
    STATUS_EXPIRING: "临近保质期",
    STATUS_EXPIRED: "已过期",
}
ALERT_LOW = "low"
ALERT_EXPIRING = "expiring"
ALERT_EXPIRED = "expired"
ALERT_LABELS = {
    ALERT_LOW: "数量不足",
    ALERT_EXPIRING: "临近保质期",
    ALERT_EXPIRED: "已过期",
}
_ALERT_RANK = {ALERT_EXPIRED: 0, ALERT_EXPIRING: 1, ALERT_LOW: 2}

NAME_MAX = 60
MANUFACTURER_MAX = 60
EFFICACY_MAX = 200
UNIT_MAX = 8
NOTE_MAX = 200
MAX_SPEC = 100000.0
MAX_PACKAGES = 100000.0
MAX_SHELF_LIFE_MONTHS = 600
CATALOG_LIMIT = 300

# 药品照片：入库前统一纠正方向、缩到 1200px、转 JPEG，去掉 EXIF（含定位信息）
IMAGE_MAX_INPUT_BYTES = 8 * 1024 * 1024
IMAGE_MAX_SIDE = 1200
IMAGE_JPEG_QUALITY = 82
IMAGE_MIME = "image/jpeg"

_SCHEMA_ITEMS = """
CREATE TABLE IF NOT EXISTS medicine_cabinet (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    group_id BIGINT UNSIGNED NOT NULL,
    created_by BIGINT UNSIGNED NOT NULL DEFAULT 0,
    category VARCHAR(8) NOT NULL DEFAULT 'drug',
    name VARCHAR(120) NOT NULL,
    manufacturer VARCHAR(120) NOT NULL DEFAULT '',
    efficacy VARCHAR(255) NOT NULL DEFAULT '',
    spec_amount DECIMAL(12,2) NOT NULL DEFAULT 1,
    unit VARCHAR(16) NOT NULL DEFAULT '',
    package_unit VARCHAR(16) NOT NULL DEFAULT '盒',
    packages DECIMAL(12,2) NOT NULL DEFAULT 0,
    quantity DECIMAL(12,2) NOT NULL DEFAULT 0,
    production_date DATE NULL,
    shelf_life_months INT NOT NULL DEFAULT 0,
    expiry_date DATE NULL,
    note VARCHAR(200) NOT NULL DEFAULT '',
    image_mime VARCHAR(24) NOT NULL DEFAULT '',
    image_data MEDIUMBLOB NULL,
    image_updated_at DATETIME NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_medicine_group (group_id),
    KEY idx_medicine_expiry (group_id, expiry_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

_SCHEMA_CATALOG = """
CREATE TABLE IF NOT EXISTS medicine_catalog (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    group_id BIGINT UNSIGNED NOT NULL,
    name VARCHAR(120) NOT NULL,
    manufacturer VARCHAR(120) NOT NULL DEFAULT '',
    efficacy VARCHAR(255) NOT NULL DEFAULT '',
    spec_amount DECIMAL(12,2) NOT NULL DEFAULT 1,
    unit VARCHAR(16) NOT NULL DEFAULT '',
    package_unit VARCHAR(16) NOT NULL DEFAULT '盒',
    uses INT UNSIGNED NOT NULL DEFAULT 1,
    created_by BIGINT UNSIGNED NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_catalog_group_name (group_id, name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

_SCHEMA_ALERTS = """
CREATE TABLE IF NOT EXISTS medicine_alerts (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    group_id BIGINT UNSIGNED NOT NULL,
    medicine_id BIGINT UNSIGNED NOT NULL,
    kind VARCHAR(16) NOT NULL,
    active TINYINT(1) NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_alert_item_kind (medicine_id, kind),
    KEY idx_alert_group_active (group_id, active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

_SCHEMA_READS = """
CREATE TABLE IF NOT EXISTS medicine_alert_reads (
    alert_id BIGINT UNSIGNED NOT NULL,
    user_id BIGINT UNSIGNED NOT NULL,
    read_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (alert_id, user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

_schema_ready = False


class MedicineError(Exception):
    """可以直接展示给用户的家庭药箱错误。"""


class ValidationError(MedicineError):
    """录入内容不合法。"""


class DatabaseError(MedicineError):
    """数据库连不上或读写失败；细节只写进服务端日志。"""


# ===================== 数据库 =====================


def _ensure_schema(cur) -> None:
    global _schema_ready
    if _schema_ready:
        return
    try:
        cur.execute(_SCHEMA_ITEMS)
        cur.execute(_SCHEMA_CATALOG)
        cur.execute(_SCHEMA_ALERTS)
        cur.execute(_SCHEMA_READS)
        # 老库没有照片列（CREATE TABLE IF NOT EXISTS 不会补列），这里显式补一次。
        # 照片存 MEDIUMBLOB：压缩到 1200px 的 JPEG 一般 100–300 KB，够清楚也不撑库。
        cur.execute(
            "SELECT COUNT(*) AS n FROM information_schema.columns"
            " WHERE table_schema = DATABASE() AND table_name = 'medicine_cabinet'"
            " AND column_name = 'image_data'"
        )
        row = cur.fetchone() or {}
        if not int(row.get("n") or 0):
            cur.execute(
                "ALTER TABLE medicine_cabinet"
                " ADD COLUMN image_mime VARCHAR(24) NOT NULL DEFAULT '' AFTER note,"
                " ADD COLUMN image_data MEDIUMBLOB NULL AFTER image_mime,"
                " ADD COLUMN image_updated_at DATETIME NULL AFTER image_data"
            )
        _schema_ready = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure medicine schema failed: %s", exc)
        raise DatabaseError("家庭药箱数据表初始化失败，请检查数据库权限。") from exc


@contextmanager
def _cursor(commit: bool = False) -> Iterator:
    """统一把底层 auth.DatabaseError 翻译成 MedicineError。

    需要多条语句一起成功时传 commit=True，走 auth.db_transaction()。
    """
    try:
        manager = auth.db_transaction() if commit else auth.db_cursor()
        with manager as cur:
            _ensure_schema(cur)
            yield cur
    except auth.DatabaseError as exc:
        raise DatabaseError(str(exc)) from exc


def group_of(user_id: int) -> int:
    """这个账号的家庭药箱 id：家庭组管理者 id，没有家庭组时就是自己。"""
    try:
        return family.group_owner_id(int(user_id))
    except family.FamilyError as exc:
        raise DatabaseError(str(exc)) from exc


# ===================== 纯计算 / 校验 =====================


def add_months(day: date, months: int) -> date:
    """生产日期 + 保质期（月）；月末自动落到目标月的最后一天。"""
    month_index = day.month - 1 + int(months)
    year = day.year + month_index // 12
    month = month_index % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(day.day, last_day))


def expiry_of(
    production_date: Optional[date], shelf_life_months: Optional[int]
) -> Optional[date]:
    """到期日；没有生产日期或保质期时不设（也就不会提醒临期）。"""
    if not production_date or not shelf_life_months or int(shelf_life_months) <= 0:
        return None
    return add_months(production_date, int(shelf_life_months))


def quantity_of(spec_amount: float, packages: float) -> float:
    """总数 = 一个包装的数量 × 包装数（规格 5 颗/盒、已有 5 盒 → 25 颗）。"""
    return round(float(spec_amount or 0) * float(packages or 0), 2)


def low_stock_line(spec_amount: float) -> float:
    """缺量线：一个规格单位的 1/2（5 颗/盒 → 2.5 颗）。"""
    return round(float(spec_amount or 0) * LOW_STOCK_RATIO, 2)


def fmt_number(value) -> str:
    """数字转成界面上好看的样子：25 / 2.5 / 0.05。"""
    try:
        text = f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "0"
    text = text.rstrip("0").rstrip(".")
    return text or "0"


def _as_date(value) -> Optional[date]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValidationError("日期格式不对，请按「2026-11-05」填写。") from exc


def _clean_text(value, limit: int, label: str, required: bool = False) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValidationError(f"请填写{label}。")
    if len(text) > limit:
        raise ValidationError(f"{label}最多 {limit} 个字符。")
    return text


def _clean_amount(value, label: str, maximum: float, allow_zero: bool) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label}要填数字。") from exc
    if number < 0 or (number == 0 and not allow_zero):
        raise ValidationError(f"{label}必须{'大于' if not allow_zero else '不小于'} 0。")
    if number > maximum:
        raise ValidationError(f"{label}不能超过 {fmt_number(maximum)}。")
    return round(number, 2)


def normalize(data: Dict, today: Optional[date] = None) -> Dict:
    """校验表单内容，并算出数量、到期日。页面预览和保存共用这一份规则。"""
    today = today or date.today()

    category = str(data.get("category") or CATEGORY_DRUG).strip()
    if category not in CATEGORIES:
        raise ValidationError("请选择类别（药品或卫材）。")

    name = _clean_text(data.get("name"), NAME_MAX, "名称", required=True)
    manufacturer = _clean_text(
        data.get("manufacturer"), MANUFACTURER_MAX, "厂商"
    )
    efficacy = _clean_text(data.get("efficacy"), EFFICACY_MAX, "功效")
    note = _clean_text(data.get("note"), NOTE_MAX, "备注")

    unit = _clean_text(data.get("unit"), UNIT_MAX, "单位", required=True)
    package_unit = _clean_text(data.get("package_unit"), UNIT_MAX, "包装单位")
    package_unit = package_unit or "盒"

    spec_amount = _clean_amount(
        data.get("spec_amount"), "规格（每个包装的数量）", MAX_SPEC, allow_zero=False
    )
    packages = _clean_amount(
        data.get("packages"), "现有数量（包装数）", MAX_PACKAGES, allow_zero=True
    )

    production_date = _as_date(data.get("production_date"))
    if production_date and production_date > today:
        raise ValidationError("生产日期不能晚于今天。")
    try:
        shelf_life_months = int(data.get("shelf_life_months") or 0)
    except (TypeError, ValueError) as exc:
        raise ValidationError("保质期要填月数（只填数字）。") from exc
    if shelf_life_months < 0 or shelf_life_months > MAX_SHELF_LIFE_MONTHS:
        raise ValidationError(
            f"保质期请填 0–{MAX_SHELF_LIFE_MONTHS} 个月；0 表示不设保质期。"
        )

    return {
        "category": category,
        "name": name,
        "manufacturer": manufacturer,
        "efficacy": efficacy,
        "unit": unit,
        "package_unit": package_unit,
        "spec_amount": spec_amount,
        "packages": packages,
        "quantity": quantity_of(spec_amount, packages),
        "production_date": production_date,
        "shelf_life_months": shelf_life_months,
        "expiry_date": expiry_of(production_date, shelf_life_months),
        "note": note,
    }


def _row_to_item(row: Dict, today: Optional[date] = None) -> Dict:
    """数据库行 → 带状态的前端结构；状态每次读都按今天重新算。"""
    today = today or date.today()
    expiry_date = row.get("expiry_date")
    if isinstance(expiry_date, datetime):
        expiry_date = expiry_date.date()

    spec_amount = float(row.get("spec_amount") or 0)
    packages = float(row.get("packages") or 0)
    quantity = float(row.get("quantity") or 0)

    days_left = (expiry_date - today).days if expiry_date else None
    expired = days_left is not None and days_left < 0
    expiring = days_left is not None and 0 <= days_left <= EXPIRY_WARN_DAYS
    low = quantity <= low_stock_line(spec_amount) + 1e-9

    if expired:
        status = STATUS_EXPIRED
    elif expiring:
        status = STATUS_EXPIRING
    elif low:
        status = STATUS_LOW
    else:
        status = STATUS_OK

    item = {
        "id": int(row["id"]),
        "group_id": int(row.get("group_id") or 0),
        "created_by": int(row.get("created_by") or 0),
        "category": str(row.get("category") or CATEGORY_DRUG),
        "category_label": CATEGORIES.get(
            str(row.get("category") or CATEGORY_DRUG), "药品"
        ),
        "name": str(row.get("name") or ""),
        "manufacturer": str(row.get("manufacturer") or ""),
        "efficacy": str(row.get("efficacy") or ""),
        "unit": str(row.get("unit") or ""),
        "package_unit": str(row.get("package_unit") or ""),
        "spec_amount": spec_amount,
        "packages": packages,
        "quantity": quantity,
        "production_date": (
            row.get("production_date").date()
            if isinstance(row.get("production_date"), datetime)
            else row.get("production_date")
        ),
        "shelf_life_months": int(row.get("shelf_life_months") or 0),
        "expiry_date": expiry_date,
        "note": str(row.get("note") or ""),
        # 列表只带「有没有照片」，不把 BLOB 一起查出来；要看图再单独取
        "has_image": bool(row.get("has_image")),
        "image_updated_at": row.get("image_updated_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "days_left": days_left,
        "low": low,
        "expiring": expiring,
        "expired": expired,
        "status": status,
        "status_label": STATUS_LABELS[status],
        "low_line": low_stock_line(spec_amount),
    }
    item["badges"] = status_badges(item)
    item["spec_label"] = spec_label(item)
    item["quantity_label"] = quantity_label(item)
    item["stock_label"] = stock_label(item)
    item["expiry_label"] = expiry_label(item)
    return item


def status_badges(item: Dict) -> List[Tuple[str, str]]:
    """卡片上的状态标签：临期/过期和缺量可以同时出现。"""
    badges: List[Tuple[str, str]] = []
    if item.get("expired"):
        badges.append(("已过期", "danger"))
    elif item.get("expiring"):
        badges.append(("临近保质期", "warn"))
    if item.get("low"):
        badges.append(("需补药", "warn"))
    return badges


def spec_label(item: Dict) -> str:
    """规格：5 颗/盒；包装单位和计数单位相同时只写一次。"""
    amount = fmt_number(item.get("spec_amount"))
    unit = item.get("unit") or "个"
    package_unit = item.get("package_unit") or ""
    if package_unit and package_unit != unit:
        return f"{amount} {unit}/{package_unit}"
    return f"{amount} {unit}"


def quantity_label(item: Dict) -> str:
    return f"{fmt_number(item.get('quantity'))} {item.get('unit') or '个'}"


def stock_label(item: Dict) -> str:
    """5 盒 × 5 颗/盒 = 25 颗。"""
    packages = item.get("packages")
    package_unit = item.get("package_unit") or "个"
    return (
        f"{fmt_number(packages)} {package_unit} × {spec_label(item)}"
        f" = {quantity_label(item)}"
    )


def expiry_label(item: Dict) -> str:
    expiry_date = item.get("expiry_date")
    if not expiry_date:
        return "未设置保质期"
    days_left = item.get("days_left")
    if days_left is None:
        return str(expiry_date)
    if days_left < 0:
        return f"{expiry_date} · 已过期 {abs(days_left)} 天"
    if days_left == 0:
        return f"{expiry_date} · 今天到期"
    return f"{expiry_date} · 剩 {days_left} 天"


def alert_message(item: Dict, kind: str) -> str:
    """提醒里的一句话，说清楚剩多少 / 还剩几天。"""
    if kind == ALERT_LOW:
        return (
            f"只剩 {quantity_label(item)}，不足半{item.get('package_unit') or '个'}"
            f"（{fmt_number(item.get('low_line'))} {item.get('unit') or '个'}），该补了"
        )
    if kind == ALERT_EXPIRED:
        days = abs(int(item.get("days_left") or 0))
        return f"有效期至 {item.get('expiry_date')}，已过期 {days} 天，请及时清理"
    days = int(item.get("days_left") or 0)
    return f"有效期至 {item.get('expiry_date')}，只剩 {days} 天，请尽快用完或补新批次"


def alert_kinds(item: Dict) -> List[str]:
    """这一条药当前该发哪些提醒。"""
    kinds: List[str] = []
    if item.get("expired"):
        kinds.append(ALERT_EXPIRED)
    elif item.get("expiring"):
        kinds.append(ALERT_EXPIRING)
    if item.get("low"):
        kinds.append(ALERT_LOW)
    return kinds


# ===================== 药箱读写 =====================


_ITEM_COLUMNS = (
    "id, group_id, created_by, category, name, manufacturer, efficacy,"
    " spec_amount, unit, package_unit, packages, quantity, production_date,"
    " shelf_life_months, expiry_date, note,"
    " (image_data IS NOT NULL) AS has_image, image_updated_at,"
    " created_at, updated_at"
)


def list_items(
    group_id: int, keyword: str = "", today: Optional[date] = None
) -> List[Dict]:
    """家庭药箱里的全部条目：到期日近的排前面，没设保质期的排最后。"""
    today = today or date.today()
    try:
        with _cursor() as cur:
            cur.execute(
                f"SELECT {_ITEM_COLUMNS} FROM medicine_cabinet"
                " WHERE group_id = %s"
                " ORDER BY (expiry_date IS NULL), expiry_date ASC, name ASC",
                (int(group_id),),
            )
            rows = cur.fetchall() or []
    except MedicineError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("medicine list failed: %s", exc)
        raise DatabaseError("读取家庭药箱失败，请稍后再试。") from exc

    items = [_row_to_item(row, today) for row in rows]
    text = (keyword or "").strip().casefold()
    if text:
        items = [
            item
            for item in items
            if text in item["name"].casefold()
            or text in item["manufacturer"].casefold()
            or text in item["efficacy"].casefold()
        ]
    return items


def get_item(
    group_id: int, item_id: int, today: Optional[date] = None
) -> Optional[Dict]:
    try:
        with _cursor() as cur:
            cur.execute(
                f"SELECT {_ITEM_COLUMNS} FROM medicine_cabinet"
                " WHERE id = %s AND group_id = %s",
                (int(item_id), int(group_id)),
            )
            row = cur.fetchone()
    except MedicineError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("medicine get failed: %s", exc)
        raise DatabaseError("读取药品信息失败，请稍后再试。") from exc
    return _row_to_item(row, today) if row else None


def prepare_image(data: bytes) -> Tuple[bytes, str]:
    """入库前的药品照片：EXIF 转正、缩到 1200px、转 JPEG，顺带去掉 GPS 等元数据。"""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - 依赖缺失
        raise ValidationError("服务器缺少图片处理组件，照片暂时存不了。") from exc

    if not data:
        raise ValidationError("这张照片是空的，请重新选择。")
    if len(data) > IMAGE_MAX_INPUT_BYTES:
        raise ValidationError(
            f"照片超过 {IMAGE_MAX_INPUT_BYTES // (1024 * 1024)} MB，请压缩后再传。"
        )

    try:
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
            buffer = io.BytesIO()
            image.save(
                buffer, format="JPEG", quality=IMAGE_JPEG_QUALITY, optimize=True
            )
            payload = buffer.getvalue()
            # 少数照片压完还是很大：继续缩，别把库撑大
            side = IMAGE_MAX_SIDE
            while len(payload) > 1024 * 1024 and side > 480:
                side = int(side * 0.8)
                image.thumbnail((side, side))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=78, optimize=True)
                payload = buffer.getvalue()
    except Exception as exc:  # noqa: BLE001
        raise ValidationError("这张照片打不开，请换一张 JPG / PNG 再试。") from exc
    return payload, IMAGE_MIME


def get_image(group_id: int, item_id: int) -> Optional[Tuple[bytes, str]]:
    """取一条药品的照片；没有照片返回 None。列表接口不会带 BLOB，看图走这里。"""
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT image_data, image_mime FROM medicine_cabinet"
                " WHERE id = %s AND group_id = %s AND image_data IS NOT NULL",
                (int(item_id), int(group_id)),
            )
            row = cur.fetchone()
    except MedicineError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("medicine image get failed: %s", exc)
        raise DatabaseError("读取药品照片失败，请稍后再试。") from exc
    if not row or row.get("image_data") is None:
        return None
    return bytes(row["image_data"]), str(row.get("image_mime") or IMAGE_MIME)


def _remember_catalog(cur, group_id: int, user_id: int, clean: Dict) -> None:
    """把这次填过的药写进联想词库；空白字段不覆盖以前记下的内容。"""
    cur.execute(
        "INSERT INTO medicine_catalog"
        " (group_id, name, manufacturer, efficacy, spec_amount, unit, package_unit,"
        "  uses, created_by)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, 1, %s)"
        " ON DUPLICATE KEY UPDATE"
        "  manufacturer = IF(VALUES(manufacturer) <> '', VALUES(manufacturer), manufacturer),"
        "  efficacy = IF(VALUES(efficacy) <> '', VALUES(efficacy), efficacy),"
        "  spec_amount = VALUES(spec_amount),"
        "  unit = IF(VALUES(unit) <> '', VALUES(unit), unit),"
        "  package_unit = IF(VALUES(package_unit) <> '', VALUES(package_unit), package_unit),"
        "  uses = uses + 1",
        (
            int(group_id),
            clean["name"],
            clean["manufacturer"],
            clean["efficacy"],
            clean["spec_amount"],
            clean["unit"],
            clean["package_unit"],
            int(user_id),
        ),
    )


def save(
    group_id: int,
    user_id: int,
    data: Dict,
    item_id: Optional[int] = None,
    today: Optional[date] = None,
    image: Optional[bytes] = None,
    remove_image: bool = False,
) -> Dict:
    """新增或修改一条药品；保存后立刻重算提醒并写进联想词库。

    image 是新照片（会压缩后入库）；remove_image=True 表示删掉旧照片；
    两者同时给时以新照片为准。
    """
    clean = normalize(data, today=today)
    group_id = int(group_id)
    user_id = int(user_id)
    stored = prepare_image(image) if image else None

    try:
        with _cursor(commit=True) as cur:
            if item_id is not None:
                cur.execute(
                    "SELECT id FROM medicine_cabinet WHERE id = %s AND group_id = %s",
                    (int(item_id), group_id),
                )
                if not cur.fetchone():
                    raise ValidationError("这条药品不存在，可能已经被家人删除了。")
                columns = [
                    "category = %s",
                    "name = %s",
                    "manufacturer = %s",
                    "efficacy = %s",
                    "spec_amount = %s",
                    "unit = %s",
                    "package_unit = %s",
                    "packages = %s",
                    "quantity = %s",
                    "production_date = %s",
                    "shelf_life_months = %s",
                    "expiry_date = %s",
                    "note = %s",
                ]
                params = [
                    clean["category"],
                    clean["name"],
                    clean["manufacturer"],
                    clean["efficacy"],
                    clean["spec_amount"],
                    clean["unit"],
                    clean["package_unit"],
                    clean["packages"],
                    clean["quantity"],
                    clean["production_date"],
                    clean["shelf_life_months"],
                    clean["expiry_date"],
                    clean["note"],
                ]
                if stored:
                    columns += [
                        "image_data = %s",
                        "image_mime = %s",
                        "image_updated_at = NOW()",
                    ]
                    params += [stored[0], stored[1]]
                elif remove_image:
                    columns += [
                        "image_data = NULL",
                        "image_mime = ''",
                        "image_updated_at = NOW()",
                    ]
                params += [int(item_id), group_id]
                cur.execute(
                    "UPDATE medicine_cabinet SET "
                    + ", ".join(columns)
                    + " WHERE id = %s AND group_id = %s",
                    tuple(params),
                )
                saved_id = int(item_id)
            else:
                image_mime = stored[1] if stored else ""
                image_data = stored[0] if stored else None
                image_at = datetime.now() if stored else None
                cur.execute(
                    "INSERT INTO medicine_cabinet"
                    " (group_id, created_by, category, name, manufacturer, efficacy,"
                    "  spec_amount, unit, package_unit, packages, quantity,"
                    "  production_date, shelf_life_months, expiry_date, note,"
                    "  image_mime, image_data, image_updated_at)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,"
                    " %s, %s, %s, %s)",
                    (
                        group_id,
                        user_id,
                        clean["category"],
                        clean["name"],
                        clean["manufacturer"],
                        clean["efficacy"],
                        clean["spec_amount"],
                        clean["unit"],
                        clean["package_unit"],
                        clean["packages"],
                        clean["quantity"],
                        clean["production_date"],
                        clean["shelf_life_months"],
                        clean["expiry_date"],
                        clean["note"],
                        image_mime,
                        image_data,
                        image_at,
                    ),
                )
                saved_id = int(cur.lastrowid)
            _remember_catalog(cur, group_id, user_id, clean)
    except MedicineError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("medicine save failed: %s", exc)
        raise DatabaseError("保存药品失败，请稍后再试。") from exc

    # 数量 / 到期日变了，提醒要跟着变（失败不影响这次保存的结果）
    try:
        sync_alerts(group_id, today=today)
    except MedicineError as exc:
        logger.warning("medicine sync alerts after save failed: %s", exc)

    saved = get_item(group_id, saved_id, today=today)
    if not saved:
        raise DatabaseError("药品没有保存成功，请重试。")
    return saved


def adjust_packages(
    group_id: int,
    item_id: int,
    delta: float,
    today: Optional[date] = None,
) -> Dict:
    """补货 / 用掉一个包装：数量按规格自动重算。"""
    item = get_item(group_id, item_id, today=today)
    if not item:
        raise ValidationError("这条药品不存在，可能已经被家人删除了。")
    packages = max(0.0, round(float(item["packages"]) + float(delta), 2))
    quantity = quantity_of(item["spec_amount"], packages)
    try:
        with _cursor(commit=True) as cur:
            cur.execute(
                "UPDATE medicine_cabinet SET packages = %s, quantity = %s"
                " WHERE id = %s AND group_id = %s",
                (packages, quantity, int(item_id), int(group_id)),
            )
    except MedicineError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("medicine adjust failed: %s", exc)
        raise DatabaseError("更新数量失败，请稍后再试。") from exc

    try:
        sync_alerts(group_id, today=today)
    except MedicineError as exc:
        logger.warning("medicine sync alerts after adjust failed: %s", exc)

    updated = get_item(group_id, item_id, today=today)
    if not updated:
        raise DatabaseError("更新数量失败，请重试。")
    return updated


def delete_item(group_id: int, item_id: int) -> None:
    """删掉一条药，连同它的提醒和已读记录；联想词库保留。"""
    try:
        with _cursor(commit=True) as cur:
            cur.execute(
                "DELETE r FROM medicine_alert_reads r"
                " JOIN medicine_alerts a ON a.id = r.alert_id"
                " WHERE a.medicine_id = %s AND a.group_id = %s",
                (int(item_id), int(group_id)),
            )
            cur.execute(
                "DELETE FROM medicine_alerts WHERE medicine_id = %s AND group_id = %s",
                (int(item_id), int(group_id)),
            )
            cur.execute(
                "DELETE FROM medicine_cabinet WHERE id = %s AND group_id = %s",
                (int(item_id), int(group_id)),
            )
            if cur.rowcount == 0:
                raise ValidationError("这条药品不存在，可能已经被家人删除了。")
    except MedicineError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("medicine delete failed: %s", exc)
        raise DatabaseError("删除药品失败，请稍后再试。") from exc


# ===================== 联想词库 =====================


def catalog_names(group_id: int, limit: int = CATALOG_LIMIT) -> List[str]:
    """这里记着这个家庭组录入过的全部药名，按用得多的排前面。"""
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT name FROM medicine_catalog WHERE group_id = %s"
                " ORDER BY uses DESC, updated_at DESC, name ASC LIMIT %s",
                (int(group_id), int(limit)),
            )
            rows = cur.fetchall() or []
    except MedicineError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("medicine catalog failed: %s", exc)
        raise DatabaseError("读取药品联想词库失败，请稍后再试。") from exc
    return [str(row["name"]) for row in rows if row.get("name")]


def catalog_lookup(group_id: int, name: str) -> Optional[Dict]:
    """按名字取回上次填的厂商 / 功效 / 规格，用于表单自动带出。"""
    text = (name or "").strip()
    if not text:
        return None
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT name, manufacturer, efficacy, spec_amount, unit, package_unit"
                " FROM medicine_catalog WHERE group_id = %s AND name = %s",
                (int(group_id), text),
            )
            row = cur.fetchone()
    except MedicineError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("medicine catalog lookup failed: %s", exc)
        raise DatabaseError("读取药品联想词库失败，请稍后再试。") from exc
    if not row:
        return None
    return {
        "name": str(row.get("name") or ""),
        "manufacturer": str(row.get("manufacturer") or ""),
        "efficacy": str(row.get("efficacy") or ""),
        "spec_amount": float(row.get("spec_amount") or 1),
        "unit": str(row.get("unit") or ""),
        "package_unit": str(row.get("package_unit") or ""),
    }


# ===================== 提醒 =====================


def sync_alerts(group_id: int, today: Optional[date] = None) -> int:
    """按当前库存重算提醒：新出现的激活，已经不成立的下线。

    下线时一并删掉已读记录，这样同一条药过一阵又临期 / 又缺量时，
    会重新提醒到每个人，而不是被以前点过的「知道了」吞掉。
    """
    today = today or date.today()
    items = list_items(group_id, today=today)
    desired = {
        (int(item["id"]), kind): item
        for item in items
        for kind in alert_kinds(item)
    }

    active = 0
    try:
        with _cursor(commit=True) as cur:
            cur.execute(
                "SELECT id, medicine_id, kind, active FROM medicine_alerts"
                " WHERE group_id = %s",
                (int(group_id),),
            )
            existing = cur.fetchall() or []
            seen: set = set()
            for row in existing:
                key = (int(row["medicine_id"]), str(row["kind"]))
                seen.add(key)
                alert_id = int(row["id"])
                if key in desired:
                    if not int(row["active"]):
                        cur.execute(
                            "UPDATE medicine_alerts SET active = 1 WHERE id = %s",
                            (alert_id,),
                        )
                        cur.execute(
                            "DELETE FROM medicine_alert_reads WHERE alert_id = %s",
                            (alert_id,),
                        )
                    active += 1
                elif int(row["active"]):
                    cur.execute(
                        "UPDATE medicine_alerts SET active = 0 WHERE id = %s",
                        (alert_id,),
                    )
                    cur.execute(
                        "DELETE FROM medicine_alert_reads WHERE alert_id = %s",
                        (alert_id,),
                    )
            for medicine_id, kind in desired:
                if (medicine_id, kind) in seen:
                    continue
                cur.execute(
                    "INSERT INTO medicine_alerts (group_id, medicine_id, kind, active)"
                    " VALUES (%s, %s, %s, 1)",
                    (int(group_id), medicine_id, kind),
                )
                active += 1
    except MedicineError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("medicine sync alerts failed: %s", exc)
        raise DatabaseError("更新补药提醒失败，请稍后再试。") from exc
    return active


def pending_alerts(
    group_id: int,
    user_id: int,
    today: Optional[date] = None,
    unread_only: bool = True,
) -> List[Dict]:
    """这个家庭组正在生效的提醒，附带「我」读过没有。

    已读记录按 user_id 分开存：同一批提醒会出现在每个成员那里，
    谁点过「知道了」只对谁不再显示。
    """
    today = today or date.today()
    items = {int(item["id"]): item for item in list_items(group_id, today=today)}
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT a.id AS alert_id, a.medicine_id, a.kind, a.created_at,"
                " a.updated_at, r.read_at"
                " FROM medicine_alerts a"
                " LEFT JOIN medicine_alert_reads r"
                "  ON r.alert_id = a.id AND r.user_id = %s"
                " WHERE a.group_id = %s AND a.active = 1",
                (int(user_id), int(group_id)),
            )
            rows = cur.fetchall() or []
    except MedicineError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("medicine alerts failed: %s", exc)
        raise DatabaseError("读取补药提醒失败，请稍后再试。") from exc

    alerts: List[Dict] = []
    for row in rows:
        item = items.get(int(row["medicine_id"]))
        if not item:  # 药被删掉但提醒还没同步时的兜底
            continue
        kind = str(row["kind"])
        if kind not in ALERT_LABELS:
            continue
        read = row.get("read_at") is not None
        if unread_only and read:
            continue
        alerts.append(
            {
                "id": int(row["alert_id"]),
                "medicine_id": int(row["medicine_id"]),
                "kind": kind,
                "label": ALERT_LABELS[kind],
                "message": alert_message(item, kind),
                "read": read,
                "created_at": row.get("created_at"),
                "item": item,
            }
        )
    alerts.sort(
        key=lambda alert: (
            _ALERT_RANK.get(alert["kind"], 9),
            alert["item"]["days_left"] if alert["item"]["days_left"] is not None else 9999,
            alert["item"]["name"],
        )
    )
    return alerts


def mark_read(group_id: int, user_id: int, alert_id: int) -> None:
    """「我知道了」只记在当前账号名下，不影响其他家人。"""
    try:
        with _cursor(commit=True) as cur:
            cur.execute(
                "INSERT IGNORE INTO medicine_alert_reads (alert_id, user_id)"
                " SELECT id, %s FROM medicine_alerts"
                " WHERE id = %s AND group_id = %s AND active = 1",
                (int(user_id), int(alert_id), int(group_id)),
            )
    except MedicineError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("medicine mark read failed: %s", exc)
        raise DatabaseError("更新提醒状态失败，请稍后再试。") from exc


def mark_all_read(group_id: int, user_id: int) -> int:
    try:
        with _cursor(commit=True) as cur:
            cur.execute(
                "INSERT IGNORE INTO medicine_alert_reads (alert_id, user_id)"
                " SELECT id, %s FROM medicine_alerts"
                " WHERE group_id = %s AND active = 1",
                (int(user_id), int(group_id)),
            )
            return int(cur.rowcount or 0)
    except MedicineError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("medicine mark all read failed: %s", exc)
        raise DatabaseError("更新提醒状态失败，请稍后再试。") from exc
