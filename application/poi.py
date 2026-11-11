#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高德 POI 检索：按坐标找附近医院，或按城市名找。

需要环境变量 AMAP_KEY（高德开放平台「Web服务」类型的 key）。
未配置或调用失败时返回空列表，调用方展示降级内容即可。
"""

import os
from typing import Dict, List

import requests

AMAP_KEY = os.environ.get("AMAP_KEY", "").strip()

_AROUND_URL = "https://restapi.amap.com/v3/place/around"
_TEXT_URL = "https://restapi.amap.com/v3/place/text"

# 搜两类：综合医院优先，其次社区卫生服务中心
DEFAULT_KEYWORD = "医院|社区卫生服务中心"


def _parse(pois, limit: int) -> List[Dict]:
    out: List[Dict] = []
    for p in pois or []:
        location = (p.get("location") or "").split(",")
        if len(location) != 2:
            continue
        tel = p.get("tel") or ""
        if isinstance(tel, list):
            tel = "；".join(str(t) for t in tel if t)
        try:
            distance = int(p.get("distance") or 0)
        except (TypeError, ValueError):
            distance = 0
        out.append(
            {
                "name": p.get("name") or "",
                "address": p.get("address") or "",
                "distance": distance,
                "tel": str(tel),
                "lng": location[0],
                "lat": location[1],
            }
        )
        if len(out) >= limit:
            break
    out.sort(key=lambda item: item["distance"] or 10**9)
    return out


def nearby(
    lat: float,
    lng: float,
    keyword: str = DEFAULT_KEYWORD,
    radius: int = 5000,
    limit: int = 5,
    timeout: float = 12.0,
) -> List[Dict]:
    """按坐标检索附近医院。"""
    if not AMAP_KEY:
        return []
    params = {
        "key": AMAP_KEY,
        "location": f"{lng},{lat}",
        "keywords": keyword,
        "radius": str(radius),
        "offset": str(min(limit, 25)),
        "page": "1",
        "extensions": "base",
        "sortrule": "distance",
    }
    try:
        resp = requests.get(_AROUND_URL, params=params, timeout=timeout)
        data = resp.json()
    except Exception:  # noqa: BLE001
        return []
    if str(data.get("status")) != "1":
        return []
    return _parse(data.get("pois"), limit)


def by_city(
    city: str,
    keyword: str = DEFAULT_KEYWORD,
    limit: int = 5,
    timeout: float = 12.0,
) -> List[Dict]:
    """按城市名检索（定位失败时的兜底）。"""
    if not AMAP_KEY or not city.strip():
        return []
    params = {
        "key": AMAP_KEY,
        "keywords": keyword,
        "city": city.strip(),
        "citylimit": "true",
        "offset": str(min(limit, 25)),
        "page": "1",
        "extensions": "base",
    }
    try:
        resp = requests.get(_TEXT_URL, params=params, timeout=timeout)
        data = resp.json()
    except Exception:  # noqa: BLE001
        return []
    if str(data.get("status")) != "1":
        return []
    return _parse(data.get("pois"), limit)


def search_url(keyword: str = "医院") -> str:
    """高德搜索深链：手机上点开会唤起 App，桌面端打开网页版。"""
    from urllib.parse import quote

    return f"https://uri.amap.com/search?keyword={quote(keyword)}&callnative=1"
