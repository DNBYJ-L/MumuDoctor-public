#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境上下文：按坐标取当前气温、体感、湿度、风、降水和空气质量。

数据源
------
天气 / 空气质量：Open-Meteo（免 key，服务端直接请求）
    https://api.open-meteo.com/v1/forecast
    https://air-quality-api.open-meteo.com/v1/air-quality
地名（坐标 → 城市）：高德逆地理编码（配置了 AMAP_KEY 时）→ BigDataCloud 兜底
城市（城市 → 坐标）：Open-Meteo geocoding → 高德地理编码兜底

设计约定
--------
1. 所有函数都不抛异常。取不到就返回 {"ok": False, "error": "..."}，
   对话照常进行，只是这一次没有环境信息。
2. 只取「当前」和「今天」的少量字段，不下载逐小时预报，响应体很小。
3. 返回的是普通 dict，页面直接放进 session_state，坐标本身不写库、
   不进对话历史，模型只看到城市名和天气数值。
4. AQI 用 Open-Meteo 的 us_aqi，等级文案按常见六档划分（优/良/轻度/中度/重度/严重）。
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import requests

AMAP_KEY = os.environ.get("AMAP_KEY", "").strip()

_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
_AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_REVERSE_URL = "https://api.bigdatacloud.net/data/reverse-geocode-client"
_AMAP_REGEO_URL = "https://restapi.amap.com/v3/geocode/regeo"
_AMAP_GEO_URL = "https://restapi.amap.com/v3/geocode/geo"

# 超时都压得比较短：环境信息是锦上添花，不能把回答卡住
_TIMEOUT_WEATHER = 5.0
_TIMEOUT_AIR = 4.0
_TIMEOUT_GEO = 3.0

# 进程内短缓存：同一个位置 10 分钟内复用，多用户/多次 rerun 不重复打接口
_CACHE_TTL = 600.0
_CACHE: Dict[Tuple[float, float, str], Tuple[float, Dict]] = {}

# WMO weather code → 中文
_WEATHER_TEXT = {
    0: "晴",
    1: "大致晴朗",
    2: "多云",
    3: "阴",
    45: "有雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "大毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪",
    80: "阵雨",
    81: "强阵雨",
    82: "暴雨",
    85: "阵雪",
    86: "强阵雪",
    95: "雷阵雨",
    96: "雷暴伴冰雹",
    99: "强雷暴伴冰雹",
}


# ===================== 小工具 =====================


def _get_json(url: str, params: Dict, timeout: float) -> Optional[Dict]:
    """GET 一个 JSON；任何异常都当成没取到。"""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _num(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first(values, index: int = 0):
    """Open-Meteo 的 daily 字段是数组；这里容错成标量。"""
    if isinstance(values, list):
        return values[index] if len(values) > index else None
    return values


def weather_text(code) -> str:
    try:
        return _WEATHER_TEXT.get(int(code), "天气未知")
    except (TypeError, ValueError):
        return "天气未知"


def aqi_level(aqi: Optional[float]) -> str:
    if aqi is None:
        return ""
    if aqi <= 50:
        return "优"
    if aqi <= 100:
        return "良"
    if aqi <= 150:
        return "轻度污染"
    if aqi <= 200:
        return "中度污染"
    if aqi <= 300:
        return "重度污染"
    return "严重污染"


def aqi_tone(aqi: Optional[float]) -> str:
    """给页面用的粗粒度颜色：ok / warn / bad。"""
    if aqi is None:
        return "ok"
    if aqi <= 100:
        return "ok"
    if aqi <= 150:
        return "warn"
    return "bad"


def _temp_text(value: Optional[float]) -> str:
    return f"{value:.0f}°C" if value is not None else "气温未知"


# ===================== 地名 =====================


def _clean_place(name: str) -> str:
    return (name or "").strip()


def _reverse_amap(lat: float, lng: float) -> str:
    if not AMAP_KEY:
        return ""
    data = _get_json(
        _AMAP_REGEO_URL,
        {
            "key": AMAP_KEY,
            "location": f"{lng:.6f},{lat:.6f}",
            "extensions": "base",
            "radius": "1000",
        },
        _TIMEOUT_GEO,
    )
    if not data or str(data.get("status")) != "1":
        return ""
    component = ((data.get("regeocode") or {}).get("addressComponent") or {})
    city = component.get("city")
    if isinstance(city, list):
        city = ""
    return _clean_place(str(city or component.get("province") or ""))


def _reverse_bigdata(lat: float, lng: float) -> str:
    data = _get_json(
        _REVERSE_URL,
        {"latitude": lat, "longitude": lng, "localityLanguage": "zh"},
        _TIMEOUT_GEO,
    )
    if not data:
        return ""
    return _clean_place(str(data.get("city") or data.get("principalSubdivision") or ""))


def place_name(lat: float, lng: float) -> str:
    """坐标 → 城市名；都失败返回空串（页面显示「当前位置」）。"""
    name = _reverse_amap(lat, lng)
    if name:
        return name
    return _reverse_bigdata(lat, lng)


def lookup_city(name: str) -> Optional[Dict]:
    """城市名 → {lat, lng, label}；找不到返回 None。"""
    text = _clean_place(name)
    if not text:
        return None

    data = _get_json(
        _GEOCODE_URL,
        {"name": text, "count": 1, "language": "zh", "format": "json"},
        _TIMEOUT_GEO,
    )
    results = (data or {}).get("results") or []
    if results:
        item = results[0]
        lat = _num(item.get("latitude"))
        lng = _num(item.get("longitude"))
        if lat is not None and lng is not None:
            label = _clean_place(str(item.get("name") or text))
            admin = _clean_place(str(item.get("admin1") or ""))
            if admin and admin != label:
                label = f"{label}（{admin}）"
            return {"lat": lat, "lng": lng, "label": label}

    if AMAP_KEY:
        data = _get_json(
            _AMAP_GEO_URL,
            {"key": AMAP_KEY, "address": text, "output": "JSON"},
            _TIMEOUT_GEO,
        )
        geocodes = (data or {}).get("geocodes") or []
        if geocodes:
            location = str(geocodes[0].get("location") or "").split(",")
            if len(location) == 2:
                lng = _num(location[0])
                lat = _num(location[1])
                if lat is not None and lng is not None:
                    label = _clean_place(
                        str(geocodes[0].get("city") or geocodes[0].get("province") or text)
                    )
                    return {"lat": lat, "lng": lng, "label": label or text}
    return None


# ===================== 天气 / 空气 =====================


def _fetch_weather(lat: float, lng: float) -> Optional[Dict]:
    return _get_json(
        _WEATHER_URL,
        {
            "latitude": lat,
            "longitude": lng,
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,wind_speed_10m"
            ),
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,uv_index_max"
            ),
            "forecast_days": 1,
            "timezone": "auto",
        },
        _TIMEOUT_WEATHER,
    )


def _fetch_air(lat: float, lng: float) -> Optional[Dict]:
    return _get_json(
        _AIR_URL,
        {
            "latitude": lat,
            "longitude": lng,
            "current": "pm2_5,pm10,us_aqi",
            "timezone": "auto",
        },
        _TIMEOUT_AIR,
    )


def _stamp(iso_text: str) -> str:
    """'2026-11-04T17:00' → '11-04 17:00'；解析不了就原样返回。"""
    text = str(iso_text or "")
    if "T" in text and len(text) >= 16:
        return f"{text[5:10]} {text[11:16]}"
    return text


def fetch(lat: float, lng: float, city: str = "") -> Dict:
    """取一个位置当前的环境信息。永不抛异常，失败返回 ok=False。"""
    try:
        lat = float(lat)
        lng = float(lng)
    except (TypeError, ValueError):
        return {"ok": False, "error": "坐标无效"}

    key = (round(lat, 2), round(lng, 2), _clean_place(city))
    now = time.time()
    cached = _CACHE.get(key)
    if cached and now - cached[0] < _CACHE_TTL:
        return dict(cached[1])

    # 天气、空气、地名三个请求并行：最坏情况也只等最慢的那个
    with ThreadPoolExecutor(max_workers=3) as pool:
        weather_future = pool.submit(_fetch_weather, lat, lng)
        air_future = pool.submit(_fetch_air, lat, lng)
        place_future = pool.submit(place_name, lat, lng) if not _clean_place(city) else None
        weather = weather_future.result()
        air = air_future.result()
        place = place_future.result() if place_future else ""

    current = (weather or {}).get("current") or {}
    if not current:
        return {"ok": False, "error": "天气服务暂时不可用"}

    daily = (weather or {}).get("daily") or {}
    air_current = (air or {}).get("current") or {}

    temp = _num(current.get("temperature_2m"))
    tmax = _num(_first(daily.get("temperature_2m_max")))
    tmin = _num(_first(daily.get("temperature_2m_min")))
    aqi = _num(air_current.get("us_aqi"))

    env = {
        "ok": True,
        "city": _clean_place(city) or place,
        "observed_at": _stamp(str(current.get("time") or "")),
        "temp": temp,
        "feels": _num(current.get("apparent_temperature")),
        "humidity": _num(current.get("relative_humidity_2m")),
        "wind": _num(current.get("wind_speed_10m")),
        "precip": _num(current.get("precipitation")),
        "code": current.get("weather_code"),
        "text": weather_text(current.get("weather_code")),
        "tmax": tmax,
        "tmin": tmin,
        "pop": _num(_first(daily.get("precipitation_probability_max"))),
        "uv": _num(_first(daily.get("uv_index_max"))),
        "pm25": _num(air_current.get("pm2_5")),
        "pm10": _num(air_current.get("pm10")),
        "aqi": aqi,
        "aqi_level": aqi_level(aqi),
        "fetched_at": now,
    }
    if env["temp"] is None and env["tmax"] is None:
        return {"ok": False, "error": "天气数据不完整"}

    if len(_CACHE) > 512:
        _CACHE.clear()
    _CACHE[key] = (now, env)
    return dict(env)


# ===================== 文案 =====================


def describe(env: Optional[Dict]) -> str:
    """页面上一行短描述：北京 · 12°C 阴 · 空气良。"""
    if not env or not env.get("ok"):
        return ""
    parts = [env.get("city") or "当前位置"]
    if env.get("temp") is not None:
        parts.append(_temp_text(env["temp"]))
    if env.get("text") and env["text"] != "天气未知":
        parts.append(str(env["text"]))
    if env.get("aqi") is not None:
        parts.append(f"空气{env.get('aqi_level') or ''}".strip())
    return " · ".join(parts)


def _range_text(env: Dict) -> str:
    tmax, tmin = env.get("tmax"), env.get("tmin")
    if tmax is None or tmin is None:
        return ""
    return f"{tmin:.0f}~{tmax:.0f}°C"


def prompt_block(env: Dict, elder: bool = False) -> str:
    """给模型的 system 片段：环境数据 + 怎么用、不要怎么用。"""
    if not env or not env.get("ok"):
        return ""

    if elder:
        bits = [env.get("city") or "你所在的地方"]
        if env.get("temp") is not None:
            bits.append(_temp_text(env["temp"]))
        if env.get("text") and env["text"] != "天气未知":
            bits.append(str(env["text"]))
        if env.get("aqi") is not None:
            bits.append(f"空气{env.get('aqi_level')}")
        return (
            f"【当前环境】{'，'.join(bits)}。\n"
            "能用上就用一句话提醒和天气、空气有关的注意事项（穿衣、通风、少在外面待）；"
            "用不上就不提，别硬扯。"
        )

    lines: List[str] = []
    place = env.get("city") or "用户所在地"
    when = env.get("observed_at") or ""
    temp = _temp_text(env.get("temp"))
    feels = env.get("feels")
    feels_text = f"（体感 {feels:.0f}°C）" if feels is not None else ""
    weather = env.get("text") or ""
    humidity = env.get("humidity")
    wind = env.get("wind")
    precip = env.get("precip")

    main = f"{when + ' 于' if when else ''}{place}：气温 {temp}{feels_text}，{weather}"
    if humidity is not None:
        main += f"，湿度 {humidity:.0f}%"
    if wind is not None:
        main += f"，风速 {wind:.0f} km/h"
    if precip is not None:
        main += f"，当前降水 {precip:.1f} mm"

    day_bits = []
    span = _range_text(env)
    if span:
        day_bits.append(f"今天 {span}")
    if env.get("pop") is not None:
        day_bits.append(f"降水概率 {env['pop']:.0f}%")
    if env.get("uv") is not None:
        day_bits.append(f"紫外线指数 {env['uv']:.1f}")
    if day_bits:
        main += "；" + "，".join(day_bits)

    if env.get("aqi") is not None:
        air = f"空气质量 {env.get('aqi_level')}（AQI {env['aqi']:.0f}"
        if env.get("pm25") is not None:
            air += f"，PM2.5 {env['pm25']:.0f} μg/m³"
        if env.get("pm10") is not None:
            air += f"，PM10 {env['pm10']:.0f} μg/m³"
        air += "）"
        main += "；" + air
    else:
        main += "；空气质量暂时没有数据"

    lines.append(f"【当前环境】{main}。")
    lines.append("")
    lines.append("回答时可以自然结合上面的环境，给出更贴合当下的生活建议：")
    lines.append("- 例如穿衣保暖、室内通风、户外运动的时机与强度、呼吸道和过敏防护、")
    lines.append("  温差对血压和心脑血管的影响、雨雪天的防滑。")
    lines.append("要求：")
    lines.append("1. 只在确实相关时提一句，不要每段都提天气，也不要把上面的数据整段复述。")
    lines.append("2. 环境信息只用来调整生活方式建议，不能当作诊断依据，也不能替代测量值。")
    lines.append("3. 只使用这里给出的数据，不要编造预报、花粉、空气质量或紫外线数值。")
    lines.append("4. 如果和问题无关，就正常回答，不要硬扯环境。")
    return "\n".join(lines)


def notice(env: Optional[Dict]) -> Optional[Dict]:
    """极端天气/空气下补一条确定性提醒，避免模型完全不提。

    返回 {"text": ..., "keys": [...]}；keys 用来判断模型是不是已经说过了。
    只在明显需要提醒时才给，普通天气不打扰。
    """
    if not env or not env.get("ok"):
        return None

    text = str(env.get("text") or "")
    code = env.get("code")
    try:
        code_int = int(code) if code is not None else -1
    except (TypeError, ValueError):
        code_int = -1
    precip = env.get("precip")
    wind = env.get("wind")
    temp = env.get("temp")
    aqi = env.get("aqi")

    # 1. 雷暴 / 短时强降水 / 大风
    if code_int in (95, 96, 99) or (precip is not None and precip >= 5) or (
        wind is not None and wind >= 50
    ):
        if code_int in (95, 96, 99):
            what = "正在打雷下雨"
        elif wind is not None and wind >= 50:
            what = f"风很大（{wind:.0f} km/h）"
        else:
            what = "雨下得不小"
        return {
            "text": (
                f"{what}，尽量别在户外停留，关好门窗；必须出门时避开树木、广告牌和积水路段，"
                "慢走防滑。"
            ),
            "keys": ["雷", "暴雨", "大风", "关窗", "积水"],
        }

    # 2. 空气污染
    if aqi is not None and aqi >= 150:
        level = env.get("aqi_level") or "污染"
        if aqi >= 200:
            text = (
                f"空气质量已达{level}（AQI {aqi:.0f}），尽量留在室内并关窗，"
                "必须外出时戴 N95/KN95 口罩，别做剧烈运动。"
            )
        else:
            text = (
                f"空气{level}（AQI {aqi:.0f}），哮喘、慢阻肺、心血管病人群尽量少在户外活动，"
                "外出戴口罩。"
            )
        return {
            "text": text,
            "keys": ["空气", "AQI", "aqi", "口罩", "PM2.5", "污染"],
        }

    # 3. 极端温度
    if temp is not None and temp >= 35:
        return {
            "text": (
                f"当前 {temp:.0f}°C，注意防暑：避开中午外出，少量多次补水；"
                "出现头晕、恶心、心慌要立刻到阴凉处休息。"
            ),
            "keys": ["高温", "中暑", "防暑", "补水"],
        }
    if temp is not None and temp <= 0:
        return {
            "text": (
                f"当前 {temp:.0f}°C，注意保暖，尤其头颈和手脚；"
                "心脑血管人群起床、出门前先缓一缓，避免冷热骤变。"
            ),
            "keys": ["保暖", "添衣", "防寒", "受凉"],
        }

    # 4. 昼夜温差
    tmax, tmin = env.get("tmax"), env.get("tmin")
    if tmax is not None and tmin is not None and tmax - tmin >= 12:
        return {
            "text": (
                f"今天 {tmin:.0f}~{tmax:.0f}°C，早晚温差大，出门带件外套；"
                "血压偏高的人注意监测。"
            ),
            "keys": ["温差", "早晚", "添衣", "外套"],
        }

    return None
