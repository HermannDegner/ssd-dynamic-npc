"""
animal_crossing_basal_system.py
どうぶつの森NPCに特化した基層システム

基層衝動を5つに集約：
1. 安心欲求 (Comfort) - 安全で平和な環境への欲求
2. 社交欲求 (Social) - 他者との交流への欲求  
3. 探索欲求 (Exploration) - 新しい体験への欲求
4. 創造欲求 (Creation) - 何かを作り上げる欲求
5. 承認欲求 (Recognition) - 認められたい欲求
"""

import random
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class BasalDrive(Enum):
    """どうぶつの森NPC用基層衝動"""
    COMFORT = "comfort"           # 安心・安全欲求
    SOCIAL = "social"             # 社交・仲間欲求
    EXPLORATION = "exploration"   # 探索・発見欲求
    CREATION = "creation"         # 創造・表現欲求
    RECOGNITION = "recognition"   # 承認・評価欲求

@dataclass
class BasalState:
    """基層システムの状態"""
    # 各衝動の満足度 (0.0-1.0)
    comfort_level: float = 0.5      # 安心度
    social_fulfillment: float = 0.5  # 社交満足度
    exploration_need: float = 0.3    # 探索欲求
    creation_urge: float = 0.2       # 創造衝動
    recognition_level: float = 0.4   # 承認欲求
    
    # 環境からの刺激
    threat_level: float = 0.0        # 脅威レベル
    social_opportunities: float = 0.5 # 社交機会
    novelty_available: float = 0.3   # 新奇性の利用可能性

class AnimalCrossingBasalSystem:
    """どうぶつの森NPC用基層システム"""
    
    def __init__(self, personality_type: str):
        self.personality_type = personality_type
        self.basal_state = BasalState()
        
        # 性格タイプによる基層衝動の強度調整
        self.drive_weights = self._initialize_drive_weights(personality_type)
        
        # 基層反応のパターン
        self.reaction_patterns = self._initialize_reaction_patterns()
        
        # 現在活性化している衝動
        self.active_drives: Dict[BasalDrive, float] = {}

    def _initialize_drive_weights(self, personality_type: str) -> Dict[BasalDrive, float]:
        """性格タイプに基づく基層衝動の重み設定"""
        
        # どうぶつの森の性格タイプに対応
        weights = {
            # 明るく元気な性格
            "peppy": {
                BasalDrive.SOCIAL: 0.9,
                BasalDrive.EXPLORATION: 0.8, 
                BasalDrive.RECOGNITION: 0.7,
                BasalDrive.CREATION: 0.6,
                BasalDrive.COMFORT: 0.4
            },
            
            # のんびりした性格
            "lazy": {
                BasalDrive.COMFORT: 0.9,
                BasalDrive.SOCIAL: 0.6,
                BasalDrive.CREATION: 0.5,
                BasalDrive.EXPLORATION: 0.3,
                BasalDrive.RECOGNITION: 0.2
            },
            
            # 気難しい性格
            "cranky": {
                BasalDrive.COMFORT: 0.8,
                BasalDrive.RECOGNITION: 0.7,
                BasalDrive.SOCIAL: 0.3,
                BasalDrive.EXPLORATION: 0.4,
                BasalDrive.CREATION: 0.5
            },
            
            # 普通の性格
            "normal": {
                BasalDrive.SOCIAL: 0.7,
                BasalDrive.COMFORT: 0.6,
                BasalDrive.CREATION: 0.6,
                BasalDrive.EXPLORATION: 0.5,
                BasalDrive.RECOGNITION: 0.5
            },
            
            # 元気な性格（男性）
            "jock": {
                BasalDrive.EXPLORATION: 0.9,
                BasalDrive.RECOGNITION: 0.8,
                BasalDrive.SOCIAL: 0.7,
                BasalDrive.CREATION: 0.4,
                BasalDrive.COMFORT: 0.3
            },
            
            # 気取った性格
            "snooty": {
                BasalDrive.RECOGNITION: 0.9,
                BasalDrive.CREATION: 0.8,
                BasalDrive.COMFORT: 0.7,
                BasalDrive.SOCIAL: 0.5,
                BasalDrive.EXPLORATION: 0.4
            }
        }
        
        return weights.get(personality_type, weights["normal"])

    def _initialize_reaction_patterns(self) -> Dict[str, Dict]:
        """基層反応パターンの初期化"""
        return {
            "comfort_seeking": {
                "trigger_conditions": {"threat_level": "> 0.3"},
                "response_actions": ["go_home", "seek_shelter", "avoid_crowds"],
                "emotional_response": "anxious"
            },
            
            "social_approach": {
                "trigger_conditions": {
                    "social_fulfillment": "< 0.4", 
                    "social_opportunities": "> 0.5"
                },
                "response_actions": ["greet_player", "visit_friend", "start_conversation"],
                "emotional_response": "friendly"
            },
            
            "exploration_drive": {
                "trigger_conditions": {
                    "exploration_need": "> 0.6",
                    "novelty_available": "> 0.4"
                },
                "response_actions": ["explore_area", "try_new_activity", "investigate"],
                "emotional_response": "curious"
            },
            
            "creation_impulse": {
                "trigger_conditions": {"creation_urge": "> 0.7"},
                "response_actions": ["decorate_home", "garden", "craft_item"],
                "emotional_response": "inspired"
            },
            
            "recognition_seeking": {
                "trigger_conditions": {"recognition_level": "< 0.3"},
                "response_actions": ["show_off", "share_achievement", "seek_praise"],
                "emotional_response": "eager"
            }
        }

    def update_basal_state(self, environment: Dict, recent_events: List[Dict]):
        """環境と最近の出来事に基づく基層状態の更新"""
        
        # 環境要因による更新
        self._update_from_environment(environment)
        
        # 最近の出来事による更新
        self._update_from_events(recent_events)
        
        # 基層衝動の自然減衰
        self._apply_natural_decay()
        
        # 活性化している衝動を判定
        self._evaluate_active_drives()

    def _update_from_environment(self, environment: Dict):
        """環境要因による基層状態更新"""
        
        # 脅威レベルの評価
        weather = environment.get('weather', 'sunny')
        time_period = environment.get('time_period', 'day')
        crowd_size = environment.get('crowd_size', 0)
        
        # 悪天候や夜間は脅威レベル上昇
        threat_level = 0.0
        if weather == 'stormy':
            threat_level += 0.4
        elif weather == 'rainy':
            threat_level += 0.2
            
        if time_period == 'night':
            threat_level += 0.1
            
        self.basal_state.threat_level = min(1.0, threat_level)
        
        # 社交機会の評価
        self.basal_state.social_opportunities = min(1.0, crowd_size / 5.0)
        
        # 新奇性の評価
        special_events = environment.get('special_events', 0)
        new_items = environment.get('new_items', 0)
        self.basal_state.novelty_available = min(1.0, (special_events + new_items) / 3.0)

    def _update_from_events(self, recent_events: List[Dict]):
        """最近の出来事による基層状態更新"""
        
        for event in recent_events[-5:]:  # 最新5件のみ考慮
            event_type = event.get('type', '')
            intensity = event.get('intensity', 0.5)
            
            if event_type == 'social_interaction':
                # 社交満足度の向上
                success = event.get('success', False)
                change = 0.2 * intensity if success else -0.1 * intensity
                self.basal_state.social_fulfillment = max(0.0, min(1.0, 
                    self.basal_state.social_fulfillment + change))
                    
            elif event_type == 'received_gift':
                # 承認レベルの向上
                self.basal_state.recognition_level = min(1.0,
                    self.basal_state.recognition_level + 0.3 * intensity)
                    
            elif event_type == 'completed_creation':
                # 創造衝動の一時的満足
                self.basal_state.creation_urge = max(0.0,
                    self.basal_state.creation_urge - 0.4 * intensity)
                    
            elif event_type == 'discovered_something':
                # 探索欲求の一時的満足
                self.basal_state.exploration_need = max(0.0,
                    self.basal_state.exploration_need - 0.3 * intensity)

    def _apply_natural_decay(self):
        """基層状態の自然減衰"""
        # 満足度は時間とともに減衰
        self.basal_state.social_fulfillment *= 0.98
        self.basal_state.recognition_level *= 0.99
        
        # 欲求は時間とともに増加
        self.basal_state.exploration_need = min(1.0, 
            self.basal_state.exploration_need + 0.01)
        self.basal_state.creation_urge = min(1.0, 
            self.basal_state.creation_urge + 0.008)
        
        # 安心度は環境に依存
        if self.basal_state.threat_level < 0.1:
            self.basal_state.comfort_level = min(1.0,
                self.basal_state.comfort_level + 0.02)
        else:
            self.basal_state.comfort_level *= 0.95

    def _evaluate_active_drives(self):
        """現在活性化している基層衝動を評価"""
        self.active_drives = {}
        
        for drive, weight in self.drive_weights.items():
            activation_level = self._calculate_drive_activation(drive, weight)
            
            if activation_level > 0.3:  # 閾値以上で活性化
                self.active_drives[drive] = activation_level

    def _calculate_drive_activation(self, drive: BasalDrive, weight: float) -> float:
        """特定の基層衝動の活性化レベルを計算"""
        
        if drive == BasalDrive.COMFORT:
            # 脅威レベルが高いほど安心欲求が活性化
            discomfort = 1.0 - self.basal_state.comfort_level
            threat_amplification = 1.0 + self.basal_state.threat_level
            return discomfort * weight * threat_amplification
            
        elif drive == BasalDrive.SOCIAL:
            # 社交満足度が低いほど、機会があるほど活性化
            social_need = 1.0 - self.basal_state.social_fulfillment
            opportunity_multiplier = 0.5 + self.basal_state.social_opportunities
            return social_need * weight * opportunity_multiplier
            
        elif drive == BasalDrive.EXPLORATION:
            # 探索欲求と新奇性の利用可能性
            exploration_pressure = self.basal_state.exploration_need
            novelty_multiplier = 0.3 + self.basal_state.novelty_available
            return exploration_pressure * weight * novelty_multiplier
            
        elif drive == BasalDrive.CREATION:
            # 創造衝動の蓄積
            return self.basal_state.creation_urge * weight
            
        elif drive == BasalDrive.RECOGNITION:
            # 承認レベルが低いほど活性化
            recognition_need = 1.0 - self.basal_state.recognition_level
            return recognition_need * weight
        
        return 0.0

    def get_suggested_actions(self) -> List[Tuple[str, float]]:
        """基層衝動に基づく行動提案"""
        suggestions = []
        
        for drive, activation in self.active_drives.items():
            pattern_key = self._drive_to_pattern_key(drive)
            
            if pattern_key in self.reaction_patterns:
                pattern = self.reaction_patterns[pattern_key]
                actions = pattern["response_actions"]
                
                for action in actions:
                    suggestions.append((action, activation))
        
        # 活性化レベルでソート
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:3]  # 上位3つ

    def _drive_to_pattern_key(self, drive: BasalDrive) -> str:
        """基層衝動から反応パターンキーへの変換"""
        mapping = {
            BasalDrive.COMFORT: "comfort_seeking",
            BasalDrive.SOCIAL: "social_approach", 
            BasalDrive.EXPLORATION: "exploration_drive",
            BasalDrive.CREATION: "creation_impulse",
            BasalDrive.RECOGNITION: "recognition_seeking"
        }
        return mapping.get(drive, "")

    def get_emotional_state(self) -> str:
        """現在の感情状態を取得"""
        if not self.active_drives:
            return "content"
            
        # 最も強い衝動に基づく感情状態
        dominant_drive = max(self.active_drives.items(), key=lambda x: x[1])
        drive, intensity = dominant_drive
        
        pattern_key = self._drive_to_pattern_key(drive)
        if pattern_key in self.reaction_patterns:
            base_emotion = self.reaction_patterns[pattern_key]["emotional_response"]
            
            # 強度に基づく修飾
            if intensity > 0.8:
                return f"very_{base_emotion}"
            elif intensity > 0.6:
                return base_emotion
            else:
                return f"slightly_{base_emotion}"
                
        return "neutral"

    def get_basal_summary(self) -> Dict:
        """基層システムの状態要約"""
        return {
            "personality_type": self.personality_type,
            "basal_state": {
                "comfort": self.basal_state.comfort_level,
                "social": self.basal_state.social_fulfillment,
                "exploration": self.basal_state.exploration_need,
                "creation": self.basal_state.creation_urge,
                "recognition": self.basal_state.recognition_level
            },
            "active_drives": {drive.value: level for drive, level in self.active_drives.items()},
            "emotional_state": self.get_emotional_state(),
            "suggested_actions": self.get_suggested_actions(),
            "threat_level": self.basal_state.threat_level
        }


# 使用例とテスト
def test_animal_crossing_basal_system():
    """どうぶつの森基層システムのテスト"""
    print("=== どうぶつの森基層システムテスト ===")
    
    # 異なる性格タイプのテスト
    personalities = ["peppy", "lazy", "cranky", "normal"]
    
    for personality in personalities:
        print(f"\n--- {personality.upper()} NPCのテスト ---")
        
        basal_system = AnimalCrossingBasalSystem(personality)
        
        # 環境設定
        environment = {
            'weather': 'sunny',
            'time_period': 'day', 
            'crowd_size': 2,
            'special_events': 1
        }
        
        # イベント履歴
        recent_events = [
            {'type': 'social_interaction', 'success': True, 'intensity': 0.8},
            {'type': 'discovered_something', 'intensity': 0.6}
        ]
        
        # 基層状態更新
        basal_system.update_basal_state(environment, recent_events)
        
        # 結果表示
        summary = basal_system.get_basal_summary()
        print(f"感情状態: {summary['emotional_state']}")
        print(f"活性化衝動: {summary['active_drives']}")
        print(f"推奨行動: {[action for action, _ in summary['suggested_actions']]}")
        
        # ストレス状況のテスト
        print(f"\n{personality} - ストレス状況:")
        stress_env = {
            'weather': 'stormy',
            'time_period': 'night',
            'crowd_size': 0,
            'special_events': 0
        }
        
        basal_system.update_basal_state(stress_env, [])
        stress_summary = basal_system.get_basal_summary()
        print(f"ストレス時感情: {stress_summary['emotional_state']}")
        print(f"ストレス時行動: {[action for action, _ in stress_summary['suggested_actions']]}")


if __name__ == "__main__":
    test_animal_crossing_basal_system()
