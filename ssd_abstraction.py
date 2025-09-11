
# -*- coding: utf-8 -*-
"""
SSD 抽象化プロセス（最小実装・関数API）
- 目的：エピソード（事例）をオンラインに圧縮し、プロトタイプ（抽象概念）へ昇華する。
- 特徴：意味圧p・新奇度・報酬などで重み付けし、更新/忘却/併合/多層要約を提供。
- 応用：統合認知（記憶・忘却・抽象化）、能動NPCの知識形成、LLM補助の構造学習 等。

依存：numpy, math
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import math

# ------------------------------
# パラメータ
# ------------------------------

@dataclass
class AbstractionParams:
    # 割当しきい値（半径）：既存プロトタイプに吸収するか/新規作成するか
    tau_base: float = 1.2
    tau_min: float = 0.2
    beta_kappa: float = 0.6     # κ̄が大きいほど新規作成しにくく（半径小さく）
    beta_E: float = 0.8         # Eが大きいほど新規作成しやすく（半径大きく）

    # 学習・忘却
    alpha: float = 0.4          # 学習率（重みwと組合せ）
    lam_forget: float = 0.01    # 自然減衰（強度s）
    min_strength: float = 0.05  # プルーニング閾値

    # 併合条件
    merge_radius: float = 0.6
    merge_cooldown: int = 5     # 併合後のクールタイムtick

    # 重み付け
    w_p: float = 0.8            # 意味圧pの寄与
    w_reward: float = 0.6       # 報酬/快の寄与
    w_novelty: float = 0.3      # 新奇度の寄与

    # 多層要約
    coarse_levels: int = 2      # 0=そのまま, 1=軽い併合, 2=より粗い併合

# ------------------------------
# 状態表現
# ------------------------------

@dataclass
class Prototype:
    pid: int
    mu: np.ndarray             # 重心
    var: np.ndarray            # 各次元の分散推定（最小値を敷く）
    n: float                   # 有効サンプル数
    s: float                   # 強度（出現/更新で上昇、忘却で減衰）
    last_seen: int
    tags: Dict[str, float] = field(default_factory=dict)
    cooldown: int = 0          # 併合クールタイム

@dataclass
class AbstractionState:
    dim: int
    t: int = 0
    E: float = 0.0            # 未処理圧（熱）: 跳躍/新規構造化の駆動
    kappa_bar: float = 0.0    # 整合慣性の平均
    prototypes: Dict[int, Prototype] = field(default_factory=dict)
    next_pid: int = 0

# ------------------------------
# ユーティリティ
# ------------------------------

def _dist2(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.dot(d, d))

def _softplus(x: float) -> float:
    # 数値安定化したsoftplus
    if x > 20: return x
    return math.log1p(math.exp(x))

def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))

# ------------------------------
# 核となる関数群
# ------------------------------

def ComputeAssignRadius(prm: AbstractionParams, st: AbstractionState) -> float:
    """動的しきい値半径： κ̄が高→半径↓, Eが高→半径↑"""
    r = prm.tau_base * (1.0 - prm.beta_kappa * st.kappa_bar) + prm.beta_E * st.E
    return max(prm.tau_min, r)

def SalienceWeight(prm: AbstractionParams, p: float, reward: float, novelty: float) -> float:
    """サリエンス重み：意味圧・快・新奇度を非線形に集約"""
    x = prm.w_p * p + prm.w_reward * reward + prm.w_novelty * novelty
    return _softplus(x)  # 常に正、p/reward/noveltyが大で急増

def AssignOrCreate(prm: AbstractionParams, st: AbstractionState,
                   x: np.ndarray, tags: Dict[str, float],
                   p: float, reward: float) -> int:
    """既存プロトタイプへ割当 or 新規作成。戻り値は採用pid"""
    # 近傍探索（最小二乗距離）
    r = ComputeAssignRadius(prm, st)
    best_pid, best_d2 = -1, 1e30
    for pid, proto in st.prototypes.items():
        d2 = _dist2(x, proto.mu)
        if d2 < best_d2:
            best_d2, best_pid = d2, pid

    novelty = math.sqrt(best_d2) if best_pid >= 0 else r  # 近いのが無ければ=半径相当
    w = SalienceWeight(prm, p, reward, novelty)

    if best_pid < 0 or best_d2 > r*r:
        # --- 新規作成 ---
        pid = st.next_pid
        st.next_pid += 1
        st.prototypes[pid] = Prototype(
            pid=pid, mu=x.copy(),
            var=np.ones(st.dim)*0.05, n=max(1.0, w), s=0.3 + 0.1*w,
            last_seen=st.t, tags=dict(tags), cooldown=prm.merge_cooldown
        )
        return pid

    # --- 既存更新 ---
    proto = st.prototypes[best_pid]
    # 有効重み（w）でオンライン平均/分散を更新
    n0 = proto.n
    n1 = n0 + w
    delta = x - proto.mu
    proto.mu = proto.mu + (w / n1) * delta
    # 分散はWelford風に更新（各次元）
    proto.var = np.maximum(1e-4, (n0 * proto.var + w * (delta**2)) / n1)
    proto.n = n1
    proto.s = min(1.0, proto.s + 0.05 * w)  # 強度を上げる
    proto.last_seen = st.t
    # タグ強化
    for k, v in tags.items():
        proto.tags[k] = proto.tags.get(k, 0.0) + v * w

    # クールダウンは減らす（併合をまた許可していく）
    if proto.cooldown > 0: proto.cooldown -= 1
    return best_pid

def ForgetAndPrune(prm: AbstractionParams, st: AbstractionState) -> None:
    """忘却（強度sの減衰）と弱小プロトタイプの枝刈り"""
    dead: List[int] = []
    for pid, proto in st.prototypes.items():
        proto.s = max(0.0, proto.s * (1.0 - prm.lam_forget))
        if proto.s < prm.min_strength:
            dead.append(pid)
    for pid in dead:
        del st.prototypes[pid]

def TryMergeNearby(prm: AbstractionParams, st: AbstractionState) -> None:
    """近接プロトタイプを併合（多重構造の粗視化）。大きい方へ吸収"""
    if not st.prototypes: return
    pids = list(st.prototypes.keys())
    merged = set()
    for i in range(len(pids)):
        if pids[i] in merged: continue
        a = st.prototypes.get(pids[i])
        if a is None: continue
        if a.cooldown > 0: continue
        for j in range(i+1, len(pids)):
            if pids[j] in merged: continue
            b = st.prototypes.get(pids[j])
            if b is None: continue
            if b.cooldown > 0: continue
            d = _norm(a.mu - b.mu)
            if d <= prm.merge_radius:
                # 重み比で結合
                wa, wb = a.s, b.s
                wsum = max(1e-8, wa + wb)
                mu = (wa*a.mu + wb*b.mu) / wsum
                var = (wa*a.var + wb*b.var) / wsum
                a.mu, a.var = mu, var
                a.n += b.n
                a.s = min(1.0, a.s + b.s*0.8)
                # タグも合算
                for k, v in b.tags.items():
                    a.tags[k] = a.tags.get(k, 0.0) + v
                a.cooldown = prm.merge_cooldown
                merged.add(b.pid)
        # end for j
    # remove merged
    for pid in merged:
        if pid in st.prototypes:
            del st.prototypes[pid]

def CoarseSummaries(prm: AbstractionParams, st: AbstractionState, level: int = 0) -> Dict[int, Prototype]:
    """粗視化レベルに応じた要約（0=そのまま, 1/2=擬似的に併合して返す）"""
    if level <= 0:
        return st.prototypes
    # 作業コピー
    temp = {pid: Prototype(pid=pid, mu=p.mu.copy(), var=p.var.copy(), n=p.n, s=p.s,
                           last_seen=p.last_seen, tags=dict(p.tags), cooldown=p.cooldown)
            for pid, p in st.prototypes.items()}
    tmp_state = AbstractionState(dim=st.dim, prototypes=temp, next_pid=max(temp.keys(), default=-1)+1,
                                 t=st.t, E=st.E, kappa_bar=st.kappa_bar)
    # level回だけ併合集約
    for _ in range(level):
        TryMergeNearby(prm, tmp_state)
    return tmp_state.prototypes

# ------------------------------
# 高レベル操作（1ステップ）
# ------------------------------

def Step(prm: AbstractionParams, st: AbstractionState,
         x: np.ndarray, tags: Dict[str, float],
         p: float = 1.0, reward: float = 0.0,
         dt: int = 1) -> int:
    """1ステップ：割当/作成→忘却→（条件付）併合→時刻更新"""
    pid = AssignOrCreate(prm, st, x, tags, p, reward)
    ForgetAndPrune(prm, st)
    # κ̄が高くEが低いとき=安定期は併合を優先（省エネ抽象）
    if st.kappa_bar > 0.3 and st.E < 0.4:
        TryMergeNearby(prm, st)
    st.t += dt
    return pid

# ------------------------------
# 要約ダンプ
# ------------------------------

def Summarize(st: AbstractionState, level: int = 0, topk_tags: int = 3) -> List[Dict[str, Any]]:
    protos = CoarseSummaries(AbstractionParams(), st, level=level)
    rows: List[Dict[str, Any]] = []
    for pid, p in protos.items():
        tag_sorted = sorted(p.tags.items(), key=lambda kv: kv[1], reverse=True)[:topk_tags]
        rows.append({
            "pid": pid, "n": round(p.n, 2), "s": round(p.s, 3), "last": p.last_seen,
            "mu": [round(float(v), 3) for v in p.mu.tolist()],
            "var": [round(float(v), 3) for v in p.var.tolist()],
            "tags": [k for k, _ in tag_sorted]
        })
    rows.sort(key=lambda r: r["s"], reverse=True)
    return rows
