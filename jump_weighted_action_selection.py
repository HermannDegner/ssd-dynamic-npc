"""
jump_weighted_action_selection.py
跳躍的ランダムと整合慣性を組み合わせた行動選択システム

常にランダム性を保ちながら、整合慣性が強いほど選ばれやすくする
"""

import random
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ← 実ファイル名に合わせて修正
from ac_basal_system import AnimalCrossingBasalSystem, BasalDrive
from ac_alignment_inertia import AnimalCrossingAlignmentInertia
from physical_structure_system import PhysicalStructureSystem  # 追加

@dataclass
class ActionCandidate:
    """行動候補"""
    action_name: str
    base_score: float        # 基層システムからの基本スコア
    inertia_weight: float    # 整合慣性による重み
    random_factor: float     # ランダム要素
    final_score: float       # 最終スコア
    selection_reason: str    # 選択理由

class JumpWeightedActionSelector:
    """跳躍的ランダム + 整合慣性重み付けシステム"""
    
    def __init__(self, personality_type: str):
        self.personality_type = personality_type
        
        # 跳躍パラメータ
        self.base_randomness = 0.3      # 基本ランダム性
        self.inertia_influence = 0.7    # 慣性の影響度
        self.jump_probability = 0.15    # 完全ランダム跳躍の確率
        
        # 利用可能な行動リスト
        self.available_actions = [
            "take_walk", "greet_player", "water_flowers", "read_book",
            "nap", "exercise", "decorate_home", "visit_friend",
            "go_shopping", "fishing", "bug_catching", "sing_song",
            "cook_meal", "clean_house", "sit_and_think", "explore_area"
        ]
        
        self._adjust_parameters_by_personality()
    
    def _adjust_parameters_by_personality(self):
        adjustments = {
            "peppy":  {"base_randomness": 0.4, "jump_probability": 0.25, "inertia_influence": 0.6},
            "lazy":   {"base_randomness": 0.2, "jump_probability": 0.08, "inertia_influence": 0.85},
            "cranky": {"base_randomness": 0.25,"jump_probability": 0.10, "inertia_influence": 0.8},
            "normal": {"base_randomness": 0.3, "jump_probability": 0.15, "inertia_influence": 0.7},
            "jock":   {"base_randomness": 0.35,"jump_probability": 0.20, "inertia_influence": 0.65},
            "snooty": {"base_randomness": 0.25,"jump_probability": 0.12, "inertia_influence": 0.75}
        }
        if self.personality_type in adjustments:
            adj = adjustments[self.personality_type]
            self.base_randomness = adj["base_randomness"]
            self.jump_probability = adj["jump_probability"]
            self.inertia_influence = adj["inertia_influence"]
    
    # ← heat 引数を受け取れるように変更（0..1想定）
    def select_action(self, basal_system: AnimalCrossingBasalSystem,
                      inertia_system: AnimalCrossingAlignmentInertia,
                      current_context: Dict,
                      heat: float = 0.0) -> Tuple[str, 'ActionCandidate']:
        """行動選択のメイン処理"""
        
        # 1. 完全ランダム跳躍の判定（熱で動的化）
        dynamic_jump_p = max(0.0, min(1.0, self.jump_probability * (0.5 + 0.5 * heat)))
        if random.random() < dynamic_jump_p:
            return self._perform_random_jump(current_context)
        
        # 2. 基層システムからの行動提案
        basal_suggestions = basal_system.get_suggested_actions()
        
        # 3. 整合慣性システムからの行動提案
        active_drives = list(basal_system.active_drives.keys())
        inertia_suggestions = inertia_system.get_pattern_suggestions(current_context, active_drives)
        
        # 4. 候補を構築
        candidates = self._build_action_candidates(basal_suggestions, inertia_suggestions, current_context)
        
        # 5. 重み付きランダム選択
        selected_action, selected_candidate = self._weighted_random_selection(candidates)
        return selected_action, selected_candidate
    
    def _perform_random_jump(self, context: Dict) -> Tuple[str, 'ActionCandidate']:
        random_action = random.choice(self.available_actions)
        candidate = ActionCandidate(
            action_name=random_action, base_score=0.1, inertia_weight=0.0,
            random_factor=1.0, final_score=1.0, selection_reason="random_jump"
        )
        return random_action, candidate
    
    def _build_action_candidates(self, basal_suggestions: List[Tuple[str, float]],
                                 inertia_suggestions: List[Tuple[str, float]],
                                 context: Dict) -> List[ActionCandidate]:
        basal_dict = dict(basal_suggestions)
        inertia_dict = dict(inertia_suggestions)
        candidates: List[ActionCandidate] = []
        for action in self.available_actions:
            base_score = basal_dict.get(action, 0.1)
            inertia_weight = inertia_dict.get(action, 0.0)
            random_factor = random.uniform(0.0, self.base_randomness)
            final_score = self._calculate_final_score(base_score, inertia_weight, random_factor)
            candidates.append(ActionCandidate(
                action_name=action, base_score=base_score, inertia_weight=inertia_weight,
                random_factor=random_factor, final_score=final_score, selection_reason="weighted"
            ))
        return candidates
    
    def _calculate_final_score(self, base_score: float, inertia_weight: float, random_factor: float) -> float:
        motivated = base_score + (inertia_weight * self.inertia_influence)
        final_score = motivated + random_factor
        if inertia_weight > 0.7:
            final_score += (inertia_weight - 0.7) * 0.5
        return max(0.0, final_score)
    
    def _weighted_random_selection(self, candidates: List[ActionCandidate]) -> Tuple[str, ActionCandidate]:
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        actions = [c.action_name for c in candidates]
        weights = [max(0.0001, c.final_score) for c in candidates]
        selected_action = random.choices(actions, weights=weights, k=1)[0]
        selected_candidate = next(c for c in candidates if c.action_name == selected_action)
        return selected_action, selected_candidate
    
    def get_selection_probabilities(self, basal_system: AnimalCrossingBasalSystem,
                                    inertia_system: AnimalCrossingAlignmentInertia,
                                    current_context: Dict) -> Dict[str, float]:
        basal_suggestions = basal_system.get_suggested_actions()
        active_drives = list(basal_system.active_drives.keys())
        inertia_suggestions = inertia_system.get_pattern_suggestions(current_context, active_drives)
        action_counts = {a: 0 for a in self.available_actions}
        simulations = 1000
        for _ in range(simulations):
            candidates = self._build_action_candidates(basal_suggestions, inertia_suggestions, current_context)
            a, _ = self._weighted_random_selection(candidates)
            action_counts[a] += 1
        return {a: c / simulations for a, c in action_counts.items()}


class IntegratedNPCSelector:
    """基層システム + 整合慣性 + 跳躍選択 + 物理層の統合"""
    
    def __init__(self, npc_name: str, personality_type: str):
        self.npc_name = npc_name
        self.personality_type = personality_type
        self.basal_system = AnimalCrossingBasalSystem(personality_type)
        self.inertia_system = AnimalCrossingAlignmentInertia(npc_name, personality_type)
        self.action_selector = JumpWeightedActionSelector(personality_type)
        self.physical_system = PhysicalStructureSystem(npc_name, npc_type="animal")  # ← 追加
        
        self.action_history: List[str] = []
        self.selection_reasons = {"weighted": 0, "random_jump": 0, "physical_override": 0}
    
    def update_and_select_action(self, environment: Dict, recent_events: List[Dict], dt_seconds: int = 60) -> str:
        # 1) 基層更新
        self.basal_system.update_basal_state(environment, recent_events)
        # 1.5) 物理層更新（熱=未処理圧）
        phys_override_needed = self.physical_system.update_physical_state(dt_seconds, environment, self.action_history[-3:])
        heat = self.physical_system.get_physical_pressure()  # 0..1
        
        # 2) 物理オーバーライド
        if phys_override_needed:
            forced = self.physical_system.get_forced_actions()
            if forced:
                forced_action = forced[0][0]
                active_drives = list(self.basal_system.active_drives.keys())
                self.inertia_system.record_action_attempt(forced_action, environment, active_drives, success=True)
                self.action_history.append(forced_action)
                self.selection_reasons["physical_override"] += 1
                print(f"{self.npc_name}: {forced_action} (理由: physical_override)")
                return forced_action
        
        # 3) 行動選択（熱で跳躍率を動的化）
        selected_action, candidate = self.action_selector.select_action(
            self.basal_system, self.inertia_system, environment, heat=heat
        )
        # 4) 学習（成功と仮定）
        active_drives = list(self.basal_system.active_drives.keys())
        self.inertia_system.record_action_attempt(selected_action, environment, active_drives, success=True)
        
        # 5) 記録
        self.action_history.append(selected_action)
        self.selection_reasons[candidate.selection_reason] += 1
        
        # 6) ログ
        print(f"{self.npc_name}: {selected_action} (理由: {candidate.selection_reason}, スコア: {candidate.final_score:.2f})")
        return selected_action
    
    def get_behavior_analysis(self) -> Dict:
        if not self.action_history:
            return {}
        uniq = set(self.action_history)
        freq: Dict[str, int] = {}
        for a in self.action_history:
            freq[a] = freq.get(a, 0) + 1
        most = max(freq.items(), key=lambda x: x[1])
        return {
            "total_actions": len(self.action_history),
            "unique_actions": len(uniq),
            "diversity_ratio": len(uniq)/len(self.action_history),
            "most_frequent_action": most[0],
            "most_frequent_count": most[1],
            "selection_reasons": self.selection_reasons.copy(),
            "action_frequency": freq
        }
