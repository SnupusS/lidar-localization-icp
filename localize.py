#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Локализация LiDAR по карте CloudCompare .bin (полилинии).

Два режима работы (выбираются при старте):
  1) live  — реальный лидар Hokuyo (SCIP, команда MD) по TCP.
  2) file  — воспроизведение лог-файла сырых данных (например podval.txt)
             с симуляцией поступления кадров в реальном времени (по умолч. 25 Гц).

Перед началом движения пользователь кликает на карте точку, где примерно
находится лидар, — это задаёт стартовую позицию. Курс алгоритм находит сам
(грубым перебором по углу), после чего идёт трекинг.

Алгоритм:
  • Инициализация / восстановление — грубый перебор (XY-сетка × угол) с оценкой
    по KD-дереву карты, затем уточнение ICP. Использует подсказку пользователя.
  • Трекинг — модель постоянной скорости (прогноз) + робастный ICP (усечённый,
    устойчив к динамическим помехам) + защита от телепортации (motion gate):
    позиция/курс не могут «прыгнуть» дальше физически возможного за один кадр.

Декодер сырых данных — из potok_decode_fixed.py: каждое расстояние = 18-битное
значение из 3 SCIP-символов (смещение 0x30), сразу в миллиметрах (без /80).

Визуализация (OpenCV):
  • карта (стены), облако точек скана, ложащееся на стены;
  • лидар в виде треугольника (видно направление) + траектория-след;
  • при инициализации/потере — кандидаты перебора.

Управление окном:
  Q / ESC — выход
  R       — принудительное восстановление (повторная инициализация)
  ПРОБЕЛ  — пауза/продолжение (в режиме file)

Запуск:
  python localize.py                      # спросит режим интерактивно
  python localize.py --file podval.txt    # режим file
  python localize.py --live               # режим live
  python localize.py --file podval.txt --hz 25 --map lines.bin
"""

import sys
import os
import re
import math
import time
import struct
import socket
import argparse

import numpy as np
import cv2
from scipy.spatial import cKDTree as KDTree


#  НАСТРОЙКИ

MAP_FILE       = "lines.bin"        # карта по умолчанию
DATA_FILE      = "podval_1.txt"       # лог сырых данных по умолчанию (режим file)
PLAYBACK_HZ    = 40.0               # частота подачи кадров в режиме file (настраивается)

# Лидар (режим live)
LIDAR_IP       = "192.168.0.10"
LIDAR_PORT     = 10940

# Геометрия скана
STEPS          = 1080               # шагов в одном скане
ANGLE_MIN_DEG  = -135.0             # левый край скана
ANGLE_MAX_DEG  = 135.0             # правый край скана (270° обзора)
WHEEL_BLANK_DEG = 35.0              # крайние ±35° — колёса робота, отбрасываются

# Фильтрация дальности (в метрах, ПОСЛЕ перевода мм→м)
MIN_RANGE_M    = 0.20               # ближе — шум / корпус
MAX_RANGE_M    = 30.0               # дальше — недостоверно (UTM-30LX); сюда же
                                    # отсекается «нет возврата» (60000 мм = 60 м)

# Карта
MAP_STEP_M     = 0.05               # шаг интерполяции полилиний (м)

# ICP (трекинг)
ICP_MAX_CORR   = 1.0                # м — макс. расстояние для соответствия (трекинг)
ICP_ITER       = 25                 # макс. итераций
ICP_TOL        = 1e-4               # порог сходимости (изменение RMSE)
ICP_TRIM       = 0.85               # доля лучших соответствий для SVD (робастность)
ICP_SUBSAMPLE  = 3                  # каждый N-й луч скана для ICP (быстрее)

# Инициализация / восстановление (грубый перебор)
INIT_HEADING_STEP_DEG = 4.0         # шаг по курсу при переборе
INIT_XY_RADIUS_M      = 1.5         # радиус XY-сетки вокруг подсказки (init)
INIT_XY_STEP_M        = 0.5         # шаг XY-сетки
RECOVER_XY_RADIUS_M   = 3.0         # радиус XY-сетки при восстановлении
RECOVER_XY_STEP_M     = 0.6
COARSE_RAYS           = 90          # сколько лучей использовать в переборе

# Защита от телепортации (motion gate) и потеря трекинга
MAX_SPEED_MPS    = 4.0              # макс. линейная скорость робота
MAX_YAWRATE_DPS  = 120.0           # макс. угловая скорость робота (°/с)
GATE_MARGIN      = 1.6             # запас сверх физического предела
GOOD_RMSE_M      = 0.30            # RMSE ниже — уверенный трекинг
BAD_RMSE_M       = 0.70            # RMSE выше — кадр считается плохим
LOST_FRAMES      = 8               # столько плохих кадров подряд → восстановление
VEL_SMOOTH       = 0.6            # сглаживание оценки скорости [0..1]

# Дисплей
IMG_SIZE       = 900
TRAIL_LEN      = 4000              # длина траектории (точек)

#  ДЕКОДЕР СЫРЫХ ДАННЫХ  (из potok_decode_fixed.py — значения сразу в мм)

HEADER_LEN     = 32                 # длина заголовка scan response (эмпирически)
DATA_LEN       = STEPS * 3          # 3240 символов данных расстояний
DECODE_MAX_MM  = 60000             # макс. дальность сенсора (мм) — для санитарной проверки


def decode_3char(s: str) -> int:
    """18-битное расстояние (мм) из 3 SCIP-символов (смещение 0x30)."""
    return ((ord(s[0]) - 0x30) << 12) | ((ord(s[1]) - 0x30) << 6) | (ord(s[2]) - 0x30)


def is_valid_scip_chars(s: str) -> bool:
    """Все символы в допустимом диапазоне SCIP (0x30–0x6F)."""
    return all(0x30 <= ord(c) <= 0x6F for c in s)


def decode_frames_in(raw: str):
    """
    Извлекает ВСЕ scan response из строки raw скользящим окном.
    Возвращает list кадров; каждый кадр — list[int] длиной 1080 (мм).
    """
    frames = []
    if len(raw) < HEADER_LEN + DATA_LEN:
        return frames

    pos = 0
    while pos + HEADER_LEN + DATA_LEN <= len(raw):
        data_start = pos + HEADER_LEN
        chunk = raw[data_start: data_start + DATA_LEN]

        # Все символы блока данных должны быть валидными SCIP
        if not is_valid_scip_chars(chunk):
            pos += 1
            continue

        distances = [decode_3char(chunk[j:j + 3]) for j in range(0, DATA_LEN, 3)]

        # Санитарная проверка: хотя бы половина значений в пределах дальности
        valid_count = sum(1 for d in distances if d <= DECODE_MAX_MM)
        if valid_count >= STEPS // 2:
            frames.append(distances)
            pos = data_start + DATA_LEN          # продолжаем сразу за блоком
        else:
            pos += 1

    return frames


def parse_log_file(filename: str):
    """Читает лог-файл целиком → список кадров (каждый — list[int] из 1080 мм)."""
    frames = []
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            frames.extend(decode_frames_in(line.strip()))
    return frames


#  СКАН → ТОЧКИ (с отбрасыванием колёс и фильтрацией дальности)

# Предрасчёт углов и маски «полезного» сектора (без колёс)
_ANGLES = np.radians(np.linspace(ANGLE_MIN_DEG, ANGLE_MAX_DEG, STEPS).astype(np.float64))
_KEEP_MASK = (_ANGLES >= math.radians(ANGLE_MIN_DEG + WHEEL_BLANK_DEG)) & \
             (_ANGLES <= math.radians(ANGLE_MAX_DEG - WHEEL_BLANK_DEG))
_ANGLES_KEPT = _ANGLES[_KEEP_MASK]


def scan_to_xy(dist_mm) -> np.ndarray:
    """
    1080 расстояний (мм) → массив XY (м) в системе сенсора.
    • отбрасывает крайние ±WHEEL_BLANK_DEG (колёса робота);
    • переводит мм→м (БЕЗ /80 — значение SCIP уже в мм);
    • фильтрует по дальности и «нет возврата».

    Система сенсора: +X — вправо, +Y — вперёд (угол 0° = вперёд).

    """
    raw = np.asarray(dist_mm, dtype=np.float64)
    if raw.shape[0] != STEPS:
        if raw.shape[0] > STEPS:
            raw = raw[:STEPS]
        else:
            raw = np.pad(raw, (0, STEPS - raw.shape[0]))

    d = raw[_KEEP_MASK] / 1000.0                 # мм → м, только полезный сектор
    a = _ANGLES_KEPT

    ok = (d >= MIN_RANGE_M) & (d <= MAX_RANGE_M)
    d = d[ok]
    a = a[ok]

    x = -d * np.sin(a)                           # ← минус: правильная хиральность
    y =  d * np.cos(a)
    return np.column_stack([x, y]).astype(np.float64)


#  ЗАГРУЗКА КАРТЫ  (CloudCompare BIN v3.9)

def load_map(path: str):
    """
    Возвращает (polylines, map_pts, kd):
      polylines — список np.ndarray (n,2) — линии стен (м);
      map_pts   — (M,2) интерполированные точки карты;
      kd        — KD-дерево по map_pts.
    """
    data = open(path, "rb").read()
    if data[:4] != b"CCB2":
        raise ValueError(f"{path}: не CloudCompare BIN (magic={data[:4]!r})")

    marker = "Vertices".encode("utf-16-be")
    positions = [m.start() for m in re.finditer(re.escape(marker), data)]
    if not positions:
        raise ValueError("В файле не найдено блоков Vertices")

    polylines = []
    for v_abs in positions:
        v = v_abs + 16                        # пропуск имени "Vertices" (16 байт UTF-16BE)
        n = struct.unpack_from("<I", data, v + 51)[0]
        pts = np.frombuffer(data, "<f4", n * 3, v + 55).reshape(n, 3).copy()
        if n >= 2:
            polylines.append(pts[:, :2].astype(np.float64))

    if not polylines:
        raise ValueError("Ни одной валидной полилинии")

    pts_list = []
    for poly in polylines:
        for i in range(len(poly) - 1):
            p0, p1 = poly[i], poly[i + 1]
            n_seg = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / MAP_STEP_M)))
            for t in np.linspace(0, 1, n_seg, endpoint=False):
                pts_list.append(p0 + t * (p1 - p0))
        pts_list.append(poly[-1])

    map_pts = np.asarray(pts_list, dtype=np.float64)
    kd = KDTree(map_pts)

    print(f"  Карта: {len(polylines)} полилиний, {len(map_pts)} точек")
    print(f"  X: [{map_pts[:,0].min():.2f}, {map_pts[:,0].max():.2f}] м")
    print(f"  Y: [{map_pts[:,1].min():.2f}, {map_pts[:,1].max():.2f}] м")
    return polylines, map_pts, kd


#  ГЕОМЕТРИЯ

def rot2(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def transform_scan(scan_xy: np.ndarray, pose) -> np.ndarray:
    """Скан из СК сенсора → мировая СК карты по позе (x, y, theta)."""
    x, y, th = pose
    return scan_xy @ rot2(th).T + np.array([x, y])


def wrap_angle(a: float) -> float:
    """Угол → диапазон (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


#  ICP  (точка-точка, усечённый / робастный)

def icp_refine(scan: np.ndarray, pose0, kd: KDTree,
               max_corr=ICP_MAX_CORR, iters=ICP_ITER, trim=ICP_TRIM):
    """
    Уточняет позу совмещением скана с картой.
    Усечение (trim) лучших соответствий делает ICP устойчивым к выбросам
    (динамические объекты, отражения). Возвращает (pose, rmse, n_inliers).
    """
    pose = list(pose0)
    prev_rmse = np.inf
    rmse = np.inf
    n_in = 0

    for _ in range(iters):
        sm = transform_scan(scan, pose)
        dist, idx = kd.query(sm, workers=-1)

        ok = dist < max_corr
        if ok.sum() < 15:
            return tuple(pose), float(prev_rmse if np.isfinite(prev_rmse) else 9.9), int(ok.sum())

        d_ok = dist[ok]
        src = sm[ok]
        dst = kd.data[idx[ok]]

        # Усечение: оставляем trim-долю ближайших соответствий
        if 0.0 < trim < 1.0 and len(d_ok) > 20:
            keep = int(max(15, math.ceil(len(d_ok) * trim)))
            order = np.argsort(d_ok)[:keep]
            src = src[order]
            dst = dst[order]
            d_used = d_ok[order]
        else:
            d_used = d_ok

        n_in = len(d_used)

        sc, dc = src.mean(0), dst.mean(0)
        H = (src - sc).T @ (dst - dc)
        U, _, Vt = np.linalg.svd(H)
        Rd = Vt.T @ U.T
        if np.linalg.det(Rd) < 0:
            Vt[-1] *= -1
            Rd = Vt.T @ U.T
        td = dc - Rd @ sc

        # Композиция поз: new_R = Rd·R, new_t = Rd·t + td
        dth = math.atan2(Rd[1, 0], Rd[0, 0])
        new_xy = Rd @ np.array(pose[:2]) + td
        pose[0], pose[1] = float(new_xy[0]), float(new_xy[1])
        pose[2] = wrap_angle(pose[2] + dth)

        rmse = float(np.sqrt((d_used ** 2).mean()))
        if abs(prev_rmse - rmse) < ICP_TOL:
            break
        prev_rmse = rmse

    return tuple(pose), float(rmse), int(n_in)


#  ГРУБЫЙ ПЕРЕБОР

def coarse_search(scan: np.ndarray, center, kd: KDTree,
                  xy_radius, xy_step, heading_step_deg=INIT_HEADING_STEP_DEG):
    """Перебор поз вокруг center: XY-сетка × курс."""
    cx, cy = center

    # Прорежённый скан для скорости
    if len(scan) > COARSE_RAYS:
        sel = np.linspace(0, len(scan) - 1, COARSE_RAYS).astype(int)
        s = scan[sel]
    else:
        s = scan
    R = len(s)

    offs = np.arange(-xy_radius, xy_radius + 1e-9, xy_step)
    PX, PY = np.meshgrid(offs, offs)
    positions = np.column_stack([PX.ravel() + cx, PY.ravel() + cy])     # (P,2)
    P = len(positions)

    headings = np.radians(np.arange(0.0, 360.0, heading_step_deg))       # (H,)
    H = len(headings)

    c = np.cos(headings); sn = np.sin(headings)
    sx, sy = s[:, 0], s[:, 1]
    rx = c[:, None] * sx[None, :] - sn[:, None] * sy[None, :]            # (H,R)
    ry = sn[:, None] * sx[None, :] + c[:, None] * sy[None, :]            # (H,R)

    wx = rx[:, None, :] + positions[:, 0][None, :, None]                 # (H,P,R)
    wy = ry[:, None, :] + positions[:, 1][None, :, None]
    pts = np.stack([wx, wy], axis=-1).reshape(-1, 2)

    dist, _ = kd.query(pts, workers=-1)
    dist = dist.reshape(H, P, R)
    dist.sort(axis=2)                                                    # усечение по лучам
    k = max(5, int(R * 0.8))
    score = dist[:, :, :k].mean(axis=2)                                  # (H,P)

    hi, pi = np.unravel_index(int(np.argmin(score)), score.shape)
    best = (float(positions[pi, 0]), float(positions[pi, 1]), float(headings[hi]))
    return best, float(score[hi, pi])


#  ЛОКАЛИЗАТОР

class Localizer:
    """Состояние и логика локализации."""

    def __init__(self, kd: KDTree, dt: float):
        self.kd = kd
        self.dt = dt                       # «логический» шаг времени между кадрами

        self.pose = (0.0, 0.0, 0.0)
        self.vel  = (0.0, 0.0, 0.0)        # оценка скорости (м/кадр, м/кадр, рад/кадр)
        self.phase = "INIT"
        self.rmse = None
        self.bad_streak = 0
        self.last_candidates = None        # для визуализации перебора

        # Предельный прирост за кадр
        self.max_lin = MAX_SPEED_MPS * dt * GATE_MARGIN
        self.max_ang = math.radians(MAX_YAWRATE_DPS) * dt * GATE_MARGIN

    def initialize(self, scan: np.ndarray, center_xy):
        pose, score = coarse_search(scan, center_xy, self.kd,
                                     INIT_XY_RADIUS_M, INIT_XY_STEP_M)
        pose, rmse, _ = icp_refine(scan, pose, self.kd,
                                   max_corr=2.0, iters=ICP_ITER, trim=ICP_TRIM)
        self.pose = pose
        self.vel = (0.0, 0.0, 0.0)
        self.rmse = rmse
        self.bad_streak = 0
        self.phase = "TRACK"
        return self.pose, self.rmse, self.phase

    def _recover(self, scan: np.ndarray):
        center = (self.pose[0], self.pose[1])
        pose, score = coarse_search(scan, center, self.kd,
                                    RECOVER_XY_RADIUS_M, RECOVER_XY_STEP_M)
        pose, rmse, _ = icp_refine(scan, pose, self.kd,
                                   max_corr=2.0, iters=ICP_ITER, trim=ICP_TRIM)
        self.pose = pose
        self.vel = (0.0, 0.0, 0.0)
        self.rmse = rmse
        self.bad_streak = 0
        self.phase = "TRACK"
        return self.pose, self.rmse

    def _gate(self, prev, proposed):
        dx = proposed[0] - prev[0]
        dy = proposed[1] - prev[1]
        dth = wrap_angle(proposed[2] - prev[2])

        gated = False
        lin = math.hypot(dx, dy)
        if lin > self.max_lin:
            k = self.max_lin / lin
            dx *= k
            dy *= k
            gated = True
        if abs(dth) > self.max_ang:
            dth = math.copysign(self.max_ang, dth)
            gated = True

        return (prev[0] + dx, prev[1] + dy, wrap_angle(prev[2] + dth)), gated

    def update(self, scan: np.ndarray):
        if self.phase == "INIT":
            # До явной инициализации ничего не делаем
            return self.pose, self.rmse, self.phase

        prev = self.pose

        # 1) Прогноз (модель постоянной скорости)
        pred = (prev[0] + self.vel[0],
                prev[1] + self.vel[1],
                wrap_angle(prev[2] + self.vel[2]))

        # 2) Уточнение ICP от прогноза
        icp_pose, rmse, n_in = icp_refine(scan, pred, self.kd,
                                          max_corr=ICP_MAX_CORR,
                                          iters=ICP_ITER, trim=ICP_TRIM)

        # 3) Защита от телепортации (ограничиваем прирост относительно prev)
        gated_pose, gated = self._gate(prev, icp_pose)

        # 4) Принятие/отклонение по качеству
        if rmse > BAD_RMSE_M or n_in < 15:
            self.bad_streak += 1
            # доверяем прогнозу (плавно), позу всё равно ограничиваем
            accepted, _ = self._gate(prev, pred)
            self.pose = accepted
            self.rmse = rmse
            if self.bad_streak >= LOST_FRAMES:
                self.phase = "RECOVER"
                self._recover(scan)
                return self.pose, self.rmse, self.phase
        else:
            self.bad_streak = 0
            self.pose = gated_pose
            self.rmse = rmse

        # 5) Обновление оценки скорости (сглаживание), с тем же ограничением
        nvx = self.pose[0] - prev[0]
        nvy = self.pose[1] - prev[1]
        nvth = wrap_angle(self.pose[2] - prev[2])
        a = VEL_SMOOTH
        self.vel = (a * self.vel[0] + (1 - a) * nvx,
                    a * self.vel[1] + (1 - a) * nvy,
                    a * self.vel[2] + (1 - a) * nvth)

        self.phase = "TRACK"
        return self.pose, self.rmse, self.phase


#  ИСТОЧНИКИ КАДРОВ

class FilePlaybackSource:
    """Воспроизведение лог-файла с симуляцией реального времени."""

    def __init__(self, path: str, hz: float):
        self.path = path
        self.hz = hz
        self.period = 1.0 / hz
        print(f"  Декодирование лога: {path}")
        self.frames = parse_log_file(path)
        print(f"  Кадров декодировано: {len(self.frames)}")
        if not self.frames:
            raise ValueError("В файле не найдено корректных кадров")
        self.i = 0
        self.paused = False
        self._next_t = None

    @property
    def total(self):
        return len(self.frames)

    def set_pause(self, p: bool):
        self.paused = p
        self._next_t = None

    def next(self):
        """Возвращает (dist_mm:list, index, done:bool). Соблюдает частоту hz."""
        if self.paused:
            return None, self.i, False
        if self.i >= len(self.frames):
            return None, self.i, True

        now = time.perf_counter()
        if self._next_t is None:
            self._next_t = now
        # выдерживаем частоту, но не «спим в долг» при отставании
        if now < self._next_t:
            time.sleep(self._next_t - now)
        self._next_t += self.period
        if self._next_t < time.perf_counter() - self.period:
            self._next_t = time.perf_counter()

        frame = self.frames[self.i]
        idx = self.i
        self.i += 1
        return frame, idx, False


class LiveLidarSource:
    """Реальный лидар Hokuyo (SCIP, команда MD) по TCP."""

    def __init__(self, ip=LIDAR_IP, port=LIDAR_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)
        print(f"  Подключение к {ip}:{port} ...")
        self.sock.connect((ip, port))
        self.sock.sendall(b"BM\n")          # включить лазер
        time.sleep(0.15)
        self._drain()
        self.i = 0
        print("  Лидар подключён.")

    def _drain(self):
        self.sock.setblocking(False)
        try:
            while True:
                if not self.sock.recv(65536):
                    break
        except (BlockingIOError, socket.error):
            pass
        self.sock.setblocking(True)
        self.sock.settimeout(5.0)

    def set_pause(self, p):       # для совместимости интерфейса
        pass

    @property
    def total(self):
        return None

    def next(self):
        """Запрашивает один скан MD, возвращает (dist_mm:list|None, index, done)."""
        self.sock.sendall(b"MD0000108000001\n")
        buf = b""
        deadline = time.time() + 0.5
        while time.time() < deadline:
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk
            if b"\n\n" in buf:
                break

        resp = buf.decode("ascii", errors="replace")
        # как в potok_file.py (это и есть логгер): убираем CC-байт каждой строки
        # и склеиваем — получаем тот же поток, что пишется в лог-файл.
        raw = "".join(line[:-1] for line in resp.split("\n") if len(line) > 3)

        frames = decode_frames_in(raw)
        frame = frames[0] if frames else None
        idx = self.i
        self.i += 1
        return frame, idx, False

    def close(self):
        try:
            self.sock.sendall(b"QT\n")
            time.sleep(0.1)
            self.sock.close()
        except Exception:
            pass


#  ВИЗУАЛИЗАЦИЯ

WIN = "LiDAR Localization"


class Visualizer:
    """Карта, скан, лидар-треугольник, траектория."""

    def __init__(self, polylines, map_pts):
        all_pts = np.vstack(polylines)
        pad = 4.0
        self.xmin = float(all_pts[:, 0].min()) - pad
        self.xmax = float(all_pts[:, 0].max()) + pad
        self.ymin = float(all_pts[:, 1].min()) - pad
        self.ymax = float(all_pts[:, 1].max()) + pad

        self._base = np.full((IMG_SIZE, IMG_SIZE, 3), (15, 15, 15), np.uint8)
        for poly in polylines:
            pts = [self._w2px(p[0], p[1]) for p in poly]
            for i in range(len(pts) - 1):
                cv2.line(self._base, pts[i], pts[i + 1], (70, 70, 70), 1, cv2.LINE_AA)

        self.trail = []

    def _w2px(self, x, y):
        s = IMG_SIZE - 40
        px = int(20 + (x - self.xmin) / (self.xmax - self.xmin) * s)
        py = int(IMG_SIZE - 20 - (y - self.ymin) / (self.ymax - self.ymin) * s)
        return px, py

    def px2w(self, px, py):
        s = IMG_SIZE - 40
        x = self.xmin + (px - 20) / s * (self.xmax - self.xmin)
        y = self.ymin + (IMG_SIZE - 20 - py) / s * (self.ymax - self.ymin)
        return float(x), float(y)

    def _draw_robot(self, img, pose, color=(80, 220, 90)):
        x, y, th = pose
        cx, cy = self._w2px(x, y)
        # вперёд в пикселях: мир (-sin th, cos th) → пиксели (−sin th, −cos th)
        fx, fy = -math.sin(th), -math.cos(th)
        px, py = -fy, fx                       # перпендикуляр
        Lf, Lb, Lw = 18, 9, 9
        tip   = (int(cx + Lf * fx),            int(cy + Lf * fy))
        left  = (int(cx - Lb * fx - Lw * px),  int(cy - Lb * fy - Lw * py))
        right = (int(cx - Lb * fx + Lw * px),  int(cy - Lb * fy + Lw * py))
        tri = np.array([tip, left, right], np.int32)
        cv2.fillConvexPoly(img, tri, color, cv2.LINE_AA)
        cv2.polylines(img, [tri], True, (255, 255, 255), 1, cv2.LINE_AA)

    def reset_trail(self):
        self.trail = []

    def draw(self, scan_xy, pose, phase, rmse, frame_n, total,
             candidates=None, info=""):
        img = self._base.copy()
        x, y, th = pose

        if phase in ("TRACK",):
            self.trail.append((x, y))
            if len(self.trail) > TRAIL_LEN:
                self.trail.pop(0)
        for i in range(1, len(self.trail)):
            cv2.line(img, self._w2px(*self.trail[i - 1]),
                     self._w2px(*self.trail[i]), (0, 140, 200), 2, cv2.LINE_AA)

        if candidates is not None:
            for (cx_, cy_) in candidates:
                pp = self._w2px(cx_, cy_)
                if 0 <= pp[0] < IMG_SIZE and 0 <= pp[1] < IMG_SIZE:
                    cv2.circle(img, pp, 1, (40, 40, 200), -1)

        if scan_xy is not None and len(scan_xy):
            sw = transform_scan(scan_xy, pose)
            for pt in sw:
                pp = self._w2px(float(pt[0]), float(pt[1]))
                if 0 <= pp[0] < IMG_SIZE and 0 <= pp[1] < IMG_SIZE:
                    cv2.circle(img, pp, 2, (230, 230, 0), -1)

        self._draw_robot(img, pose)

        C = (210, 210, 210)
        ph_col = {"TRACK": (0, 220, 120), "INIT": (0, 180, 255),
                  "RECOVER": (0, 120, 255), "WAIT": (180, 180, 0)}.get(phase, C)
        hdg = math.degrees(th) % 360
        tt = f"/{total}" if total else ""
        cv2.putText(img, f"#{frame_n}{tt}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C, 1, cv2.LINE_AA)
        cv2.putText(img, f"[{phase}]", (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ph_col, 1, cv2.LINE_AA)
        cv2.putText(img, f"X: {x:8.3f} m", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C, 1, cv2.LINE_AA)
        cv2.putText(img, f"Y: {y:8.3f} m", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C, 1, cv2.LINE_AA)
        cv2.putText(img, f"Heading: {hdg:6.1f}\u00b0", (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C, 1, cv2.LINE_AA)
        if rmse is not None:
            rc = (0, 210, 0) if rmse < GOOD_RMSE_M else (0, 160, 255) if rmse < BAD_RMSE_M else (0, 0, 255)
            cv2.putText(img, f"RMSE: {rmse:.3f} m", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rc, 1, cv2.LINE_AA)
        if info:
            cv2.putText(img, info, (10, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.putText(img, "Q/ESC=quit   R=recover   SPACE=pause",
                    (10, IMG_SIZE - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

        cv2.imshow(WIN, img)
        return img


#  КЛИК ПОЛЬЗОВАТЕЛЯ

class ClickPicker:
    def __init__(self):
        self.point_px = None

    def cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point_px = (x, y)


def wait_for_click(vis: Visualizer):
    """Показывает карту и ждёт клик пользователя."""
    picker = ClickPicker()
    cv2.setMouseCallback(WIN, picker.cb)

    while True:
        img = vis._base.copy()
        cv2.putText(img, "Click the approximate LiDAR position on the map",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "(Q / ESC to quit)",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
        if picker.point_px is not None:
            cv2.circle(img, picker.point_px, 6, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.imshow(WIN, img)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            return None
        if picker.point_px is not None:
            wx, wy = vis.px2w(*picker.point_px)
            cv2.setMouseCallback(WIN, lambda *a: None)
            return wx, wy


#  ГЛАВНЫЙ ЦИКЛ

def run(source, vis: Visualizer, kd, dt, total):
    loc = Localizer(kd, dt)

    # 1) Получаем первый валидный скан (для инициализации курса)
    print("\n  Ожидание первого скана для инициализации ...")
    first_scan = None
    while first_scan is None:
        frame, idx, done = source.next()
        if done:
            print("  Данные закончились до инициализации.")
            return
        if frame is None:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                return
            continue
        sc = scan_to_xy(frame)
        if len(sc) >= 30:
            first_scan = sc
            first_idx = idx

    # 2) Клик пользователя → стартовая точка
    center = wait_for_click(vis)
    if center is None:
        print("  Отменено пользователем.")
        return
    print(f"  Стартовая подсказка: X={center[0]:.2f} м  Y={center[1]:.2f} м")
    print("  Поиск курса (грубый перебор) ...")
    pose, rmse, phase = loc.initialize(first_scan, center)
    print(f"  Инициализация: X={pose[0]:.2f}  Y={pose[1]:.2f}  "
          f"Heading={math.degrees(pose[2])%360:.1f}°  RMSE={rmse:.3f} м")

    vis.reset_trail()
    vis.draw(first_scan, pose, "TRACK", rmse, first_idx + 1, total)
    cv2.waitKey(1)

    # 3) Основной цикл трекинга
    print("\n  Трекинг. Q/ESC — выход, R — восстановление, SPACE — пауза.\n")
    print(f"{'#':>6}  {'X,m':>9}  {'Y,m':>9}  {'Hdg':>7}  {'phase':<8}  {'RMSE':>7}  {'ms':>5}")
    print("-" * 64)

    paused = False
    last_scan = first_scan
    while True:
        t0 = time.perf_counter()

        frame, idx, done = source.next()
        if done:
            print("\n  Воспроизведение завершено.")
            break

        scan = None
        if frame is not None:
            s = scan_to_xy(frame)
            if len(s) >= 20:
                scan = s
                last_scan = s
                pose, rmse, phase = loc.update(scan)

        dt_ms = (time.perf_counter() - t0) * 1e3
        x, y, th = loc.pose
        spd = math.hypot(loc.vel[0], loc.vel[1]) / dt
        info = f"vel: {spd:.2f} m/s   bad:{loc.bad_streak}"

        vis.draw(scan if scan is not None else last_scan,
                 loc.pose, loc.phase, loc.rmse, idx + 1, total, info=info)

        if frame is not None:
            print(f"\r  {idx+1:5d}  {x:9.3f}  {y:9.3f}  "
                  f"{math.degrees(th)%360:7.1f}  {loc.phase:<8}  "
                  f"{(loc.rmse if loc.rmse is not None else 0):7.3f}  {dt_ms:5.0f}",
                  end="", flush=True)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("\n\n  Выход.")
            break
        elif key in (ord('r'), ord('R')):
            print("\n  Восстановление по запросу ...")
            if last_scan is not None:
                loc._recover(last_scan)
        elif key == ord(' '):
            paused = not paused
            source.set_pause(paused)
            print(f"\n  {'ПАУЗА' if paused else 'ПРОДОЛЖЕНИЕ'}")


def choose_mode_interactive():
    print("\nВыберите режим работы:")
    print("  1) file — воспроизведение лог-файла (симуляция реального времени)")
    print("  2) live — реальный лидар по TCP")
    while True:
        ans = input("Режим [1/2]: ").strip()
        if ans in ("1", "file", "f"):
            return "file"
        if ans in ("2", "live", "l"):
            return "live"
        print("Введите 1 или 2.")


def main():
    ap = argparse.ArgumentParser(description="LiDAR localization on a polyline map")
    ap.add_argument("--map", default=MAP_FILE, help="карта CloudCompare .bin")
    ap.add_argument("--file", dest="datafile", default=None,
                    help="лог сырых данных (включает режим file)")
    ap.add_argument("--live", action="store_true", help="режим реального лидара")
    ap.add_argument("--hz", type=float, default=PLAYBACK_HZ,
                    help="частота подачи кадров в режиме file (Гц)")
    args = ap.parse_args()

    if args.live:
        mode = "live"
    elif args.datafile:
        mode = "file"
    else:
        mode = choose_mode_interactive()

    print(f"\nЗагрузка карты: {args.map}")
    polylines, map_pts, kd = load_map(args.map)
    vis = Visualizer(polylines, map_pts)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, IMG_SIZE, IMG_SIZE)

    source = None
    try:
        if mode == "file":
            path = args.datafile or DATA_FILE
            if not os.path.exists(path):
                print(f"Файл не найден: {path}")
                return
            dt = 1.0 / args.hz
            print(f"\nРежим FILE: {path}  @ {args.hz:.1f} Гц")
            source = FilePlaybackSource(path, args.hz)
            total = source.total
        else:
            dt = 1.0 / PLAYBACK_HZ          # «логический» dt для motion gate
            print("\nРежим LIVE")
            source = LiveLidarSource()
            total = None

        run(source, vis, kd, dt, total)

    except KeyboardInterrupt:
        print("\n\nПрервано (Ctrl+C).")
    except Exception as e:
        import traceback
        print(f"\nОшибка: {e}")
        traceback.print_exc()
    finally:
        if isinstance(source, LiveLidarSource):
            source.close()
        cv2.destroyAllWindows()
        print("Завершено.")


if __name__ == "__main__":
    main()
