"""
animal_crossing_alignment_inertia.py
どうぶつの森NPC用整合慣性システム

基層衝動から生まれる行動パターンが成功体験により強化され、
習慣的な行動として定着していく仕組み
"""

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# ← 実ファイル名に合わせて修正
from ac_basal_system import BasalDrive

@dataclass
class ActionPattern:
    pattern_id: str
    trigger_conditions: Dict
    action_sequence: List[str]
    success_rate: float = 0.5
    usage_count: int = 0
    last_used_tick: int = 0
    strength: float = 0.1
    source_drives: List[BasalDrive] = None
    def __post_init__(self):
        if self.source_drives is None:
            self.source_drives = []

@dataclass
class RoutinePattern:
    routine_id: str
    time_triggers: List[str]
    location_preferences: List[str]
    activity_chain: List[str]
    flexibility: float = 0.3
    establishment_level: float = 0.0

class AnimalCrossingAlignmentInertia:
    """どうぶつの森NPC用整合慣性システム"""
    def __init__(self, npc_name: str, personality_type: str):
        self.npc_name = npc_name
        self.personality_type = personality_type
        
        self.action_patterns: Dict[str, ActionPattern] = {}
        self.routine_patterns: Dict[str, RoutinePattern] = {}
        
        # 学習・忘却
        self.learning_rate = 0.1
        self.decay_rate = 0.02
        self.pattern_threshold = 0.3
        
        # 時系列
        self.current_tick = 0
        self.action_history: List[Dict] = []
        self.success_memory: Dict[str, List[bool]] = defaultdict(list)
        
        # 上限/容量
        self.max_patterns = 15
        self.routine_importance = 0.7
        self.max_total_inertia = 8.0              # ← 追加
        self.inertia_capacity_warning = 6.5       # ← 追加
        self._learning_restricted = False         # ← 追加
        
        self._adjust_parameters_by_personality()

    def _adjust_parameters_by_personality(self):
        adjustments = {
            "peppy":  {"learning_rate": 0.15, "decay_rate": 0.03,  "pattern_threshold": 0.2},
            "lazy":   {"learning_rate": 0.05, "decay_rate": 0.01,  "pattern_threshold": 0.5},
            "cranky": {"learning_rate": 0.08, "decay_rate": 0.015, "pattern_threshold": 0.4},
            "normal": {"learning_rate": 0.1,  "decay_rate": 0.02,  "pattern_threshold": 0.3},
            "jock":   {"learning_rate": 0.12, "decay_rate": 0.025, "pattern_threshold": 0.25},
            "snooty": {"learning_rate": 0.08, "decay_rate": 0.018, "pattern_threshold": 0.35}
        }
        if self.personality_type in adjustments:
            adj = adjustments[self.personality_type]
            self.learning_rate = adj["learning_rate"]
            self.decay_rate = adj["decay_rate"]
            self.pattern_threshold = adj["pattern_threshold"]

    def record_action_attempt(self, action: str, context: Dict, 
                              source_drives: List[BasalDrive], success: bool):
        self.current_tick += 1
        self.action_history.append({
            'tick': self.current_tick, 'action': action,
            'context': context.copy(), 'source_drives': source_drives.copy(),
            'success': success
        })
        self.success_memory[action].append(success)
        if len(self.success_memory[action]) > 10:
            self.success_memory[action].pop(0)
        
        self._update_or_learn_pattern(action, context, source_drives, success)
        self._update_routine_patterns(action, context, success)
        self._decay_patterns()
        self._manage_inertia_capacity()   # ← 容量管理を実働化

    def _update_or_learn_pattern(self, action: str, context: Dict, 
                                 source_drives: List[BasalDrive], success: bool):
        matching = self._find_matching_pattern(action, context, source_drives)
        if matching:
            self._update_existing_pattern(matching, success)
        else:
            if success and len(self.action_patterns) < self.max_patterns and self._check_inertia_capacity():
                self._create_new_pattern(action, context, source_drives)

    def _find_matching_pattern(self, action: str, context: Dict, 
                               source_drives: List[BasalDrive]) -> Optional[ActionPattern]:
        for pattern in self.action_patterns.values():
            if action not in pattern.action_sequence:
                continue
            if not set(source_drives) & set(pattern.source_drives):
                continue
            if self._calculate_context_similarity(context, pattern.trigger_conditions) > 0.6:
                return pattern
        return None

    def _calculate_context_similarity(self, c1: Dict, c2: Dict) -> float:
        if not c1 or not c2: return 0.0
        keys = set(c1.keys()) & set(c2.keys())
        if not keys: return 0.0
        s = 0.0
        for k in keys:
            v1, v2 = c1[k], c2[k]
            if isinstance(v1, str) and isinstance(v2, str):
                s += 1.0 if v1 == v2 else 0.0
            elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                s += max(0.0, 1.0 - abs(v1 - v2))
            else:
                s += 0.5
        return s / len(keys)

    def _update_existing_pattern(self, pattern: ActionPattern, success: bool):
        pattern.usage_count += 1
        pattern.last_used_tick = self.current_tick
        alpha = 0.2
        pattern.success_rate = (1 - alpha) * pattern.success_rate + alpha * (1.0 if success else 0.0)
        if success:
            pattern.strength = min(1.0, pattern.strength + self.learning_rate * (1.0 - pattern.strength))
        else:
            pattern.strength = max(0.1, pattern.strength - self.learning_rate * 0.5)

    def _check_inertia_capacity(self) -> bool:
        return self._calculate_total_inertia() < self.max_total_inertia

    def _calculate_total_inertia(self) -> float:
        pattern_inertia = sum(p.strength for p in self.action_patterns.values())
        routine_inertia = sum(r.establishment_level for r in self.routine_patterns.values())
        return pattern_inertia + routine_inertia

    def _manage_inertia_capacity(self):
        total = self._calculate_total_inertia()
        if total > self.max_total_inertia:
            self._reduce_inertia_by_amount(total - self.max_total_inertia)
        elif total > self.inertia_capacity_warning:
            self._apply_learning_restrictions()

    def _reduce_inertia_by_amount(self, target_reduction: float):
        items = [('action', pid, p.strength) for pid, p in self.action_patterns.items()]
        items += [('routine', rid, r.establishment_level) for rid, r in self.routine_patterns.items()]
        items.sort(key=lambda x: x[2])  # 弱い順
        reduced = 0.0
        for typ, _id, val in items:
            if reduced >= target_reduction: break
            if typ == 'action' and _id in self.action_patterns:
                reduced += self.action_patterns[_id].strength
                del self.action_patterns[_id]
            elif typ == 'routine' and _id in self.routine_patterns:
                reduced += self.routine_patterns[_id].establishment_level
                del self.routine_patterns[_id]

    def _apply_learning_restrictions(self):
        if not self._learning_restricted:
            self.learning_rate *= 0.7
            self._learning_restricted = True

    def _create_new_pattern(self, action: str, context: Dict, source_drives: List[BasalDrive]):
        pid = f"{self.npc_name}_{action}_{len(self.action_patterns)}"
        self.action_patterns[pid] = ActionPattern(
            pattern_id=pid, trigger_conditions=context.copy(),
            action_sequence=[action], success_rate=0.7, usage_count=1,
            last_used_tick=self.current_tick, strength=0.3, source_drives=source_drives.copy()
        )

    def _update_routine_patterns(self, action: str, context: Dict, success: bool):
        time_period = context.get('time_period', 'unknown')
        location = context.get('location', 'unknown')
        if time_period == 'unknown' or not success: return
        key = f"{time_period}_{location}"
        if key in self.routine_patterns:
            r = self.routine_patterns[key]
            if action not in r.activity_chain:
                r.activity_chain.append(action)
            r.establishment_level = min(1.0, r.establishment_level + 0.05)
        else:
            if len(self.routine_patterns) < 8:
                self.routine_patterns[key] = RoutinePattern(
                    routine_id=key, time_triggers=[time_period],
                    location_preferences=[location], activity_chain=[action],
                    establishment_level=0.1
                )

    def _decay_patterns(self):
        remove = []
        for pid, p in self.action_patterns.items():
            ticks = self.current_tick - p.last_used_tick
            decay_factor = 1.0 - (self.decay_rate * ticks / 100.0)
            p.strength *= max(0.1, decay_factor)
            if p.strength < 0.05 and p.usage_count < 3:
                remove.append(pid)
        for pid in remove:
            del self.action_patterns[pid]

    def get_pattern_suggestions(self, current_context: Dict, active_drives: List[BasalDrive]) -> List[Tuple[str, float]]:
        suggestions: List[Tuple[str, float]] = []
        for p in self.action_patterns.values():
            if p.strength < self.pattern_threshold: 
                continue
            ctx = self._calculate_context_similarity(current_context, p.trigger_conditions)
            drv = len(set(active_drives) & set(p.source_drives)) / max(1, len(p.source_drives))
            score = (ctx * 0.4 + drv * 0.3 + p.strength * 0.2 + p.success_rate * 0.1)
            if score > 0.3:
                for a in p.action_sequence:
                    suggestions.append((a, score * p.strength))
        time_period = current_context.get('time_period', '')
        location = current_context.get('location', '')
        for r in self.routine_patterns.values():
            if (time_period in r.time_triggers) or (location in r.location_preferences):
                s = r.establishment_level * self.routine_importance
                for a in r.activity_chain:
                    suggestions.append((a, s))
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:5]

    def get_inertia_strength(self, action: str, context: Dict) -> float:
        m = 0.0
        for p in self.action_patterns.values():
            if action in p.action_sequence:
                m = max(m, p.strength * self._calculate_context_similarity(context, p.trigger_conditions))
        return m

    def should_break_routine(self, current_context: Dict) -> bool:
        boredom = current_context.get('boredom_level', 0.0)
        exploration = current_context.get('exploration_need', 0.0)
        return (boredom > 0.7 - self.routine_importance * 0.3) or (exploration > 0.7 - self.routine_importance * 0.3)

    def get_alignment_summary(self) -> Dict:
        pattern_stats = {
            'total_patterns': len(self.action_patterns),
            'strong_patterns': len([p for p in self.action_patterns.values() if p.strength > 0.5]),
            'routine_patterns': len(self.routine_patterns),
            'avg_success_rate': (sum(p.success_rate for p in self.action_patterns.values()) / len(self.action_patterns))
                                if self.action_patterns else 0.0
        }
        strongest = sorted(self.action_patterns.values(), key=lambda p: p.strength, reverse=True)[:3]
        return {
            'npc_name': self.npc_name,
            'personality_type': self.personality_type,
            'pattern_stats': pattern_stats,
            'strongest_patterns': [
                {'actions': p.action_sequence, 'strength': p.strength,
                 'success_rate': p.success_rate, 'usage_count': p.usage_count}
                for p in strongest
            ],
            'established_routines': [
                {'time_location': r.routine_id, 'activities': r.activity_chain, 'establishment': r.establishment_level}
                for r in self.routine_patterns.values() if r.establishment_level > 0.3
            ],
            'inertia_capacity': {
                'total_inertia': self._calculate_total_inertia(),
                'max_inertia': self.max_total_inertia,
                'warning_level': self.inertia_capacity_warning,
                'learning_restricted': self._learning_restricted,
                'capacity_usage': self._calculate_total_inertia() / self.max_total_inertia
            }
        }
