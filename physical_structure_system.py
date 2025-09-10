"""
physical_structure_system.py
SSD理論の物理構造層 - 環境的制約と脅威の管理

物理構造 = 逆らえない環境条件（天候、災害、物理法則など）
強い意味圧を生成し、上位層の行動を強制的にオーバーライドする
"""

import random
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class PhysicalThreatLevel(Enum):
    """物理的脅威レベル"""
    NONE = "none"           # 脅威なし
    MINOR = "minor"         # 軽微（雨など）
    MODERATE = "moderate"   # 中程度（嵐など）
    SEVERE = "severe"       # 重大（災害など）
    CRITICAL = "critical"   # 致命的（緊急避難レベル）

@dataclass
class PhysicalCondition:
    """物理的状況"""
    condition_type: str
    intensity: float        # 強度 (0.0-1.0)
    duration: int          # 継続時間（tick）
    threat_level: PhysicalThreatLevel
    forced_actions: List[str]  # 強制される行動
    forbidden_actions: List[str]  # 禁止される行動

class PhysicalStructureSystem:
    """物理構造システム"""
    
    def __init__(self, npc_name: str, npc_type: str = "animal"):
        self.npc_name = npc_name
        self.npc_type = npc_type
        
        # 現在の物理状況
        self.active_conditions: Dict[str, PhysicalCondition] = {}
        self.environmental_pressure = 0.0  # 環境圧力（0-1）
        self.accumulated_heat = 0.0        # 蓄積された"熱"（未処理圧）
        
        # 物理条件の定義
        self.condition_templates = self._initialize_condition_templates()
        
        # 適応度（同じ条件への慣れ）
        self.adaptation_levels: Dict[str, float] = {}
        
        # デバッグ用
        self.condition_history = []

    def _initialize_condition_templates(self) -> Dict[str, PhysicalCondition]:
        """物理条件テンプレートの初期化"""
        return {
            "sunny": PhysicalCondition(
                condition_type="sunny",
                intensity=0.0,
                duration=0,
                threat_level=PhysicalThreatLevel.NONE,
                forced_actions=[],
                forbidden_actions=[]
            ),
            
            "light_rain": PhysicalCondition(
                condition_type="light_rain", 
                intensity=0.3,
                duration=0,
                threat_level=PhysicalThreatLevel.MINOR,
                forced_actions=[],
                forbidden_actions=["take_walk", "exercise", "bug_catching"]
            ),
            
            "heavy_rain": PhysicalCondition(
                condition_type="heavy_rain",
                intensity=0.6,
                duration=0,
                threat_level=PhysicalThreatLevel.MODERATE,
                forced_actions=["seek_shelter", "go_home"],
                forbidden_actions=["take_walk", "exercise", "explore_area", "fishing"]
            ),
            
            "storm": PhysicalCondition(
                condition_type="storm",
                intensity=0.9,
                duration=0,
                threat_level=PhysicalThreatLevel.SEVERE,
                forced_actions=["seek_shelter", "stay_indoors"],
                forbidden_actions=["take_walk", "exercise", "explore_area", "fishing", "visit_friend"]
            ),
            
            "extreme_cold": PhysicalCondition(
                condition_type="extreme_cold",
                intensity=0.7,
                duration=0,
                threat_level=PhysicalThreatLevel.MODERATE,
                forced_actions=["warm_up", "go_indoors"],
                forbidden_actions=["take_walk", "sit_outside", "water_flowers"]
            ),
            
            "extreme_heat": PhysicalCondition(
                condition_type="extreme_heat",
                intensity=0.7,
                duration=0,
                threat_level=PhysicalThreatLevel.MODERATE,
                forced_actions=["seek_shade", "drink_water"],
                forbidden_actions=["exercise", "long_walks"]
            ),
            
            "earthquake": PhysicalCondition(
                condition_type="earthquake",
                intensity=1.0,
                duration=0,
                threat_level=PhysicalThreatLevel.CRITICAL,
                forced_actions=["take_cover", "evacuate"],
                forbidden_actions=["normal_activities"]  # ほぼ全て禁止
            )
        }

    def update_physical_state(self, dt_seconds: int, environment: Dict, 
                            recent_actions: List[str]) -> bool:
        """物理状態の更新"""
        
        # 1. 環境から物理条件を検出
        self._detect_conditions_from_environment(environment)
        
        # 2. 条件の持続時間を更新
        self._update_condition_durations(dt_seconds)
        
        # 3. 環境圧力を計算
        self._calculate_environmental_pressure()
        
        # 4. 熱（未処理圧）を蓄積
        self._accumulate_heat(recent_actions)
        
        # 5. 適応度を更新
        self._update_adaptation_levels()
        
        # 強制行動が必要かどうかを返す
        return self._has_override_conditions()

    def _detect_conditions_from_environment(self, environment: Dict):
        """環境から物理条件を検出"""
        weather = environment.get('weather', 'sunny')
        temperature = environment.get('temperature', 20)  # 摂氏
        special_events = environment.get('special_events', [])
        
        # 既存条件をクリア
        self.active_conditions.clear()
        
        # 天候による条件
        if weather == 'sunny':
            if temperature > 35:  # 極暑
                condition = self.condition_templates["extreme_heat"].copy()
                self.active_conditions["extreme_heat"] = condition
            elif temperature < -5:  # 極寒
                condition = self.condition_templates["extreme_cold"].copy()
                self.active_conditions["extreme_cold"] = condition
            else:
                condition = self.condition_templates["sunny"].copy()
                self.active_conditions["sunny"] = condition
                
        elif weather == 'rainy':
            condition = self.condition_templates["light_rain"].copy()
            self.active_conditions["light_rain"] = condition
            
        elif weather == 'stormy':
            condition = self.condition_templates["storm"].copy()
            self.active_conditions["storm"] = condition
        
        # 特別イベント
        for event in special_events:
            if event == 'earthquake':
                condition = self.condition_templates["earthquake"].copy()
                condition.duration = 5  # 5tick持続
                self.active_conditions["earthquake"] = condition

    def _update_condition_durations(self, dt_seconds: int):
        """条件の持続時間を更新"""
        conditions_to_remove = []
        
        for condition_id, condition in self.active_conditions.items():
            if condition.duration > 0:
                condition.duration -= 1
                if condition.duration <= 0:
                    conditions_to_remove.append(condition_id)
        
        for condition_id in conditions_to_remove:
            del self.active_conditions[condition_id]

    def _calculate_environmental_pressure(self):
        """環境圧力の計算"""
        total_pressure = 0.0
        
        for condition in self.active_conditions.values():
            # 脅威レベルに応じた基本圧力
            base_pressure = {
                PhysicalThreatLevel.NONE: 0.0,
                PhysicalThreatLevel.MINOR: 0.2,
                PhysicalThreatLevel.MODERATE: 0.5,
                PhysicalThreatLevel.SEVERE: 0.8,
                PhysicalThreatLevel.CRITICAL: 1.0
            }.get(condition.threat_level, 0.0)
            
            # 強度による調整
            condition_pressure = base_pressure * condition.intensity
            
            # 適応による軽減
            adaptation = self.adaptation_levels.get(condition.condition_type, 0.0)
            adapted_pressure = condition_pressure * (1.0 - adaptation * 0.5)
            
            total_pressure += adapted_pressure
        
        self.environmental_pressure = min(1.0, total_pressure)

    def _accumulate_heat(self, recent_actions: List[str]):
        """熱（未処理圧）の蓄積"""
        # 環境圧力による熱蓄積
        heat_gain = self.environmental_pressure * 0.1
        
        # 禁止行動を行った場合の追加熱
        forbidden_actions = self.get_forbidden_actions()
        for action in recent_actions:
            if action in forbidden_actions:
                heat_gain += 0.2
        
        # 熱の蓄積（上限1.0）
        self.accumulated_heat = min(1.0, self.accumulated_heat + heat_gain)
        
        # 熱の自然減衰
        self.accumulated_heat *= 0.95

    def _update_adaptation_levels(self):
        """適応度の更新"""
        for condition_id in self.active_conditions.keys():
            if condition_id not in self.adaptation_levels:
                self.adaptation_levels[condition_id] = 0.0
            
            # 継続曝露による適応度向上（ただし上限あり）
            max_adaptation = 0.6  # 最大60%まで慣れる
            adaptation_rate = 0.02
            current_adaptation = self.adaptation_levels[condition_id]
            
            if current_adaptation < max_adaptation:
                self.adaptation_levels[condition_id] = min(
                    max_adaptation,
                    current_adaptation + adaptation_rate
                )

    def _has_override_conditions(self) -> bool:
        """強制行動が必要な条件があるか"""
        for condition in self.active_conditions.values():
            if (condition.threat_level.value in ['severe', 'critical'] or 
                len(condition.forced_actions) > 0):
                return True
        return False

    def get_forced_actions(self) -> List[Tuple[str, float]]:
        """強制される行動リストを優先度順で取得"""
        forced_actions = []
        
        for condition in self.active_conditions.values():
            priority = {
                PhysicalThreatLevel.CRITICAL: 1.0,
                PhysicalThreatLevel.SEVERE: 0.8,
                PhysicalThreatLevel.MODERATE: 0.6,
                PhysicalThreatLevel.MINOR: 0.4,
                PhysicalThreatLevel.NONE: 0.0
            }.get(condition.threat_level, 0.0)
            
            for action in condition.forced_actions:
                forced_actions.append((action, priority * condition.intensity))
        
        # 優先度順にソート
        forced_actions.sort(key=lambda x: x[1], reverse=True)
        return forced_actions

    def get_forbidden_actions(self) -> List[str]:
        """禁止される行動リスト"""
        forbidden = set()
        
        for condition in self.active_conditions.values():
            forbidden.update(condition.forbidden_actions)
        
        return list(forbidden)

    def get_physical_pressure(self) -> float:
        """物理圧力（熱）を取得（0-1）"""
        return min(1.0, self.environmental_pressure + self.accumulated_heat)

    def is_action_possible(self, action: str) -> bool:
        """特定の行動が物理的に可能かチェック"""
        forbidden = self.get_forbidden_actions()
        return action not in forbidden

    def get_environmental_modifier(self, action: str) -> float:
        """行動に対する環境修正値"""
        forbidden = self.get_forbidden_actions()
        
        if action in forbidden:
            return 0.1  # 禁止行動は成功率大幅低下
        
        # 屋内行動は悪天候でボーナス
        indoor_actions = ["read_book", "nap", "decorate_home", "cook_meal", "clean_house"]
        if (action in indoor_actions and 
            any(c.threat_level.value in ['moderate', 'severe'] 
                for c in self.active_conditions.values())):
            return 1.3
        
        return 1.0  # 通常

    def get_physical_summary(self) -> Dict:
        """物理システムの状態要約"""
        return {
            'npc_name': self.npc_name,
            'active_conditions': {
                cid: {
                    'type': condition.condition_type,
                    'intensity': condition.intensity,
                    'threat_level': condition.threat_level.value,
                    'duration': condition.duration
                }
                for cid, condition in self.active_conditions.items()
            },
            'environmental_pressure': self.environmental_pressure,
            'accumulated_heat': self.accumulated_heat,
            'forced_actions': [action for action, _ in self.get_forced_actions()],
            'forbidden_actions': self.get_forbidden_actions(),
            'adaptation_levels': self.adaptation_levels.copy()
        }


# テスト用
def test_physical_structure_system():
    """物理構造システムのテスト"""
    print("=== 物理構造システムテスト ===")
    
    physical_system = PhysicalStructureSystem("テストNPC", "animal")
    
    # シナリオ1: 通常の晴天
    print("\n--- シナリオ1: 晴天 ---")
    environment = {'weather': 'sunny', 'temperature': 25}
    override_needed = physical_system.update_physical_state(60, environment, [])
    
    summary = physical_system.get_physical_summary()
    print(f"環境圧力: {summary['environmental_pressure']:.2f}")
    print(f"強制行動: {summary['forced_actions']}")
    print(f"禁止行動: {summary['forbidden_actions']}")
    
    # シナリオ2: 嵐
    print("\n--- シナリオ2: 嵐 ---")
    environment = {'weather': 'stormy', 'temperature': 15}
    override_needed = physical_system.update_physical_state(60, environment, ['take_walk'])
    
    summary = physical_system.get_physical_summary()
    print(f"環境圧力: {summary['environmental_pressure']:.2f}")
    print(f"強制行動: {summary['forced_actions']}")
    print(f"禁止行動: {summary['forbidden_actions']}")
    print(f"蓄積熱: {summary['accumulated_heat']:.2f}")
    print(f"強制オーバーライド必要: {override_needed}")
    
    # シナリオ3: 地震
    print("\n--- シナリオ3: 地震発生 ---")
    environment = {'weather': 'sunny', 'special_events': ['earthquake']}
    override_needed = physical_system.update_physical_state(60, environment, [])
    
    summary = physical_system.get_physical_summary()
    print(f"環境圧力: {summary['environmental_pressure']:.2f}")
    print(f"強制行動: {summary['forced_actions']}")
    print(f"蓄積熱: {summary['accumulated_heat']:.2f}")
    print(f"強制オーバーライド必要: {override_needed}")
    
    # 適応のテスト
    print("\n--- 適応テスト（雨に慣れる） ---")
    environment = {'weather': 'rainy'}
    for i in range(10):
        physical_system.update_physical_state(60, environment, [])
        if i % 3 == 0:
            adaptation = physical_system.adaptation_levels.get('light_rain', 0.0)
            pressure = physical_system.environmental_pressure
            print(f"  {i+1}回目: 適応度{adaptation:.3f}, 圧力{pressure:.3f}")


if __name__ == "__main__":
    test_physical_structure_system()
