"""
integrated_npc_system_with_episodic.py
エピソード記憶を統合したどうぶつの森NPCシステム

エピソード記憶が全ての構造層に作用する横断的システムとして機能
"""

import random
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# 既存システムのインポート
from ac_basal_system import AnimalCrossingBasalSystem, BasalDrive
from animal_crossing_alignment_inertia import AnimalCrossingAlignmentInertia
from physical_structure_system import PhysicalStructureSystem
from jump_weighted_action_selection import JumpWeightedActionSelector
from episodic_memory_system import EpisodicMemorySystem, EpisodicMemory

@dataclass
class ContextualEpisode:
    """文脈を考慮したエピソード記憶"""
    base_memory: EpisodicMemory
    contextual_relevance: float  # 現在状況との関連度
    structural_impact: Dict[str, float]  # 各構造層への影響度
    action_bias: Dict[str, float]  # 行動への偏向

class IntegratedNPCWithEpisodic:
    """エピソード記憶統合型NPCシステム"""
    
    def __init__(self, npc_name: str, personality_type: str):
        self.npc_name = npc_name
        self.personality_type = personality_type
        
        # 基本システム
        self.physical_system = PhysicalStructureSystem(npc_name, "animal")
        self.basal_system = AnimalCrossingBasalSystem(personality_type)
        self.inertia_system = AnimalCrossingAlignmentInertia(npc_name, personality_type)
        self.action_selector = JumpWeightedActionSelector(personality_type)
        
        # エピソード記憶システム
        self.episodic_memory = EpisodicMemorySystem(npc_name, memory_decay_rate=0.98)
        
        # 統合パラメータ
        self.episodic_influence_weights = {
            'physical': 0.1,    # 物理層への影響は限定的
            'basal': 0.6,       # 基層への影響は強い
            'inertia': 0.8,     # 慣性への影響は最大
            'jump': 0.4         # 跳躍選択への影響は中程度
        }
        
        # 履歴
        self.interaction_history = []
        self.decision_log = []
        self.current_tick = 0

    def update_and_select_action(self, environment: Dict, recent_events: List[Dict], 
                                perceived_entities: List[str], dt_seconds: int = 60) -> str:
        """統合システムによる行動更新と選択"""
        
        self.current_tick += 1
        
        # 1. エピソード記憶の更新
        self._process_recent_events_to_memory(recent_events, environment)
        self.episodic_memory.decay_memories()
        
        # 2. エピソード記憶からの意味圧生成
        episodic_pressure = self._generate_episodic_pressure(perceived_entities, environment)
        
        # 3. 物理構造の更新（最優先）
        physical_override = self.physical_system.update_physical_state(
            dt_seconds, environment, self.interaction_history[-3:]
        )
        
        if physical_override:
            return self._handle_physical_override(environment, episodic_pressure)
        
        # 4. エピソード記憶により調整された基層システム更新
        self._update_basal_with_episodic_influence(environment, recent_events, episodic_pressure)
        
        # 5. エピソード記憶により調整された行動選択
        selected_action = self._select_action_with_episodic_context(
            environment, episodic_pressure, perceived_entities
        )
        
        # 6. 学習とログ
        self._record_action_and_learn(selected_action, environment, episodic_pressure)
        
        return selected_action

    def _process_recent_events_to_memory(self, recent_events: List[Dict], environment: Dict):
        """最近の出来事をエピソード記憶として記録"""
        
        for event in recent_events:
            event_type = event.get('type', '')
            involved = event.get('involved_entities', [])
            location = environment.get('location', 'unknown')
            
            # SSD的な意味圧強度の計算
            valence = self._calculate_event_valence(event)
            
            # エピソード記憶として記録
            if valence != 0.0:  # 中性的でない出来事のみ記録
                self.episodic_memory.record_event(
                    event_type=event_type,
                    location=location,
                    involved_entities=involved,
                    valence=valence
                )

    def _calculate_event_valence(self, event: Dict) -> float:
        """SSD理論に基づく出来事の感情価計算"""
        event_type = event.get('type', '')
        success = event.get('success', True)
        intensity = event.get('intensity', 0.5)
        
        # 基層衝動との整合性に基づく感情価
        base_valence = {
            'social_interaction': 0.6,    # 社交欲求充足
            'gift_received': 0.8,         # 承認欲求充足
            'gift_given': 0.5,            # 社交+承認欲求充足
            'completed_creation': 0.7,    # 創造欲求充足
            'discovered_something': 0.6,  # 探索欲求充足
            'conflict': -0.7,             # 安心欲求阻害
            'failure': -0.5,              # 多重欲求阻害
            'ignored': -0.4,              # 社交+承認欲求阻害
            'interrupted': -0.3           # 安心欲求軽度阻害
        }.get(event_type, 0.0)
        
        # 成功/失敗による調整
        if not success:
            base_valence *= -0.5
        
        return base_valence * intensity

    def _generate_episodic_pressure(self, perceived_entities: List[str], 
                                   environment: Dict) -> Dict[str, float]:
        """エピソード記憶からの意味圧生成"""
        
        episodic_proposals = self.episodic_memory.generate_action_proposals(perceived_entities)
        
        # 文脈的関連度の計算
        contextual_episodes = []
        for entity in perceived_entities:
            memories = self.episodic_memory.retrieve_memories_about_entity(entity, max_memories=3)
            for memory in memories:
                relevance = self._calculate_contextual_relevance(memory, environment)
                if relevance > 0.3:
                    contextual_episodes.append(ContextualEpisode(
                        base_memory=memory,
                        contextual_relevance=relevance,
                        structural_impact=self._calculate_structural_impact(memory),
                        action_bias=self._calculate_action_bias(memory)
                    ))
        
        # 統合された意味圧として返す
        return {
            'episodic_proposals': episodic_proposals,
            'contextual_episodes': contextual_episodes,
            'memory_pressure': len(contextual_episodes) * 0.1
        }

    def _calculate_contextual_relevance(self, memory: EpisodicMemory, environment: Dict) -> float:
        """記憶の文脈的関連度計算"""
        relevance = memory.salience  # 基本は顕著性
        
        # 場所の一致
        if memory.location == environment.get('location', ''):
            relevance *= 1.3
        
        # 時間的近さ（最近の記憶ほど関連度高）
        time_diff = time.time() - memory.timestamp
        time_factor = max(0.5, 1.0 - (time_diff / (24 * 3600)))  # 24時間で半減
        relevance *= time_factor
        
        # 感情価の強さ
        emotion_factor = 1.0 + abs(memory.valence) * 0.5
        relevance *= emotion_factor
        
        return min(1.0, relevance)

    def _calculate_structural_impact(self, memory: EpisodicMemory) -> Dict[str, float]:
        """記憶が各構造層に与える影響度"""
        impact = {}
        
        # 基層構造への影響（感情価に比例）
        impact['basal'] = abs(memory.valence) * 0.8
        
        # 慣性構造への影響（成功体験は強化、失敗体験は回避を強化）
        impact['inertia'] = memory.salience * 0.6
        
        # 跳躍選択への影響（強い記憶は行動を保守化または活性化）
        if memory.valence > 0:
            impact['jump'] = memory.valence * 0.3  # ポジティブは適度な冒険を促進
        else:
            impact['jump'] = abs(memory.valence) * -0.5  # ネガティブは保守化
        
        return impact

    def _calculate_action_bias(self, memory: EpisodicMemory) -> Dict[str, float]:
        """記憶による行動への偏向計算"""
        bias = {}
        
        # 出来事タイプに基づく行動偏向
        if memory.valence > 0.5:  # ポジティブな記憶
            bias.update({
                'greet_player': 0.3,
                'visit_friend': 0.2,
                'share_achievement': 0.2
            })
        elif memory.valence < -0.5:  # ネガティブな記憶
            bias.update({
                'avoid_interaction': 0.4,
                'seek_comfort': 0.3,
                'go_home': 0.2
            })
        
        # 記憶の強度で重み付け
        for action in bias:
            bias[action] *= memory.salience
        
        return bias

    def _update_basal_with_episodic_influence(self, environment: Dict, recent_events: List[Dict], 
                                             episodic_pressure: Dict):
        """エピソード記憶の影響を受けた基層システム更新"""
        
        # 通常の基層更新
        self.basal_system.update_basal_state(environment, recent_events)
        
        # エピソード記憶による調整
        contextual_episodes = episodic_pressure.get('contextual_episodes', [])
        
        for episode in contextual_episodes:
            impact = episode.structural_impact.get('basal', 0.0)
            relevance = episode.contextual_relevance
            adjustment_strength = impact * relevance * self.episodic_influence_weights['basal']
            
            # 基層状態の微調整
            if episode.base_memory.valence > 0:
                # ポジティブな記憶は該当する欲求を軽度満足
                self._apply_positive_basal_adjustment(episode.base_memory, adjustment_strength)
            else:
                # ネガティブな記憶は不安や欲求を増加
                self._apply_negative_basal_adjustment(episode.base_memory, adjustment_strength)

    def _apply_positive_basal_adjustment(self, memory: EpisodicMemory, strength: float):
        """ポジティブな記憶による基層調整"""
        event_type = memory.event_type
        
        adjustments = {
            'GIFT_RECEIVED': {'recognition_level': strength * 0.2},
            'social_interaction': {'social_fulfillment': strength * 0.2},
            'SHARED_HOBBY': {'social_fulfillment': strength * 0.15, 'creation_urge': -strength * 0.1},
            'completed_creation': {'creation_urge': -strength * 0.2}
        }
        
        if event_type in adjustments:
            for attr, change in adjustments[event_type].items():
                if hasattr(self.basal_system.basal_state, attr):
                    current_val = getattr(self.basal_system.basal_state, attr)
                    new_val = max(0.0, min(1.0, current_val + change))
                    setattr(self.basal_system.basal_state, attr, new_val)

    def _apply_negative_basal_adjustment(self, memory: EpisodicMemory, strength: float):
        """ネガティブな記憶による基層調整"""
        # ネガティブな記憶は不安と欲求を高める
        self.basal_system.basal_state.comfort_level *= (1.0 - strength * 0.1)
        self.basal_system.basal_state.social_fulfillment *= (1.0 - strength * 0.05)

    def _select_action_with_episodic_context(self, environment: Dict, episodic_pressure: Dict, 
                                           perceived_entities: List[str]) -> str:
        """エピソード記憶の文脈を考慮した行動選択"""
        
        # 基本的な行動選択
        heat = self.physical_system.get_physical_pressure()
        selected_action, candidate = self.action_selector.select_action(
            self.basal_system, self.inertia_system, environment, heat=heat
        )
        
        # エピソード記憶による行動修正
        episodic_proposals = episodic_pressure.get('episodic_proposals', {})
        contextual_episodes = episodic_pressure.get('contextual_episodes', [])
        
        # 強い記憶による行動オーバーライドの判定
        strongest_memory_action = self._check_memory_override(episodic_proposals, contextual_episodes)
        
        if strongest_memory_action and random.random() < 0.3:  # 30%の確率でオーバーライド
            selected_action = strongest_memory_action
            candidate.selection_reason = "episodic_override"
        
        return selected_action

    def _check_memory_override(self, episodic_proposals: Dict[str, float], 
                              contextual_episodes: List[ContextualEpisode]) -> Optional[str]:
        """記憶による行動オーバーライドの判定"""
        
        # エピソード提案の中で最も強いもの
        if episodic_proposals:
            strongest_proposal = max(episodic_proposals.items(), key=lambda x: x[1])
            if strongest_proposal[1] > 0.7:
                action_name = strongest_proposal[0].replace('APPROACH_', 'greet_').replace('AVOID_', 'avoid_')
                return action_name
        
        # 文脈的エピソードからの強い偏向
        for episode in contextual_episodes:
            if episode.contextual_relevance > 0.8 and abs(episode.base_memory.valence) > 0.7:
                action_biases = episode.action_bias
                if action_biases:
                    strongest_bias = max(action_biases.items(), key=lambda x: x[1])
                    if strongest_bias[1] > 0.4:
                        return strongest_bias[0]
        
        return None

    def _handle_physical_override(self, environment: Dict, episodic_pressure: Dict) -> str:
        """物理オーバーライド時の処理"""
        forced_actions = self.physical_system.get_forced_actions()
        if forced_actions:
            forced_action = forced_actions[0][0]
            
            # 物理的強制行動でも記憶は考慮
            active_drives = list(self.basal_system.active_drives.keys())
            self.inertia_system.record_action_attempt(forced_action, environment, active_drives, success=True)
            
            self.interaction_history.append(forced_action)
            self._log_decision(forced_action, "physical_override", episodic_pressure)
            
            return forced_action
        
        return "stay_safe"  # フォールバック

    def _record_action_and_learn(self, selected_action: str, environment: Dict, episodic_pressure: Dict):
        """行動記録と学習"""
        
        # 慣性システムでの学習
        active_drives = list(self.basal_system.active_drives.keys())
        success = self._evaluate_action_success(selected_action, environment, episodic_pressure)
        
        self.inertia_system.record_action_attempt(selected_action, environment, active_drives, success)
        
        # 履歴記録
        self.interaction_history.append(selected_action)
        if len(self.interaction_history) > 10:
            self.interaction_history.pop(0)
        
        # 決定ログ
        self._log_decision(selected_action, "integrated_selection", episodic_pressure)

    def _evaluate_action_success(self, action: str, environment: Dict, episodic_pressure: Dict) -> bool:
        """行動成功の評価（簡易版）"""
        base_success_rate = 0.7
        
        # 物理的制約による調整
        physical_modifier = self.physical_system.get_environmental_modifier(action)
        
        # エピソード記憶による調整
        contextual_episodes = episodic_pressure.get('contextual_episodes', [])
        episodic_modifier = 1.0
        
        for episode in contextual_episodes:
            if action in episode.action_bias:
                bias_strength = episode.action_bias[action]
                if bias_strength > 0:
                    episodic_modifier += bias_strength * 0.5
                else:
                    episodic_modifier += bias_strength * 0.3  # ネガティブバイアスの軽減
        
        final_success_rate = base_success_rate * physical_modifier * episodic_modifier
        return random.random() < min(0.95, max(0.05, final_success_rate))

    def _log_decision(self, action: str, reason: str, episodic_pressure: Dict):
        """決定ログ"""
        self.decision_log.append({
            'tick': self.current_tick,
            'action': action,
            'reason': reason,
            'memory_pressure': episodic_pressure.get('memory_pressure', 0.0),
            'active_episodes': len(episodic_pressure.get('contextual_episodes', [])),
            'timestamp': time.time()
        })
        
        if len(self.decision_log) > 50:
            self.decision_log.pop(0)

    def get_comprehensive_summary(self) -> Dict:
        """統合システムの包括的状態要約"""
        
        # 基本システムの状態
        basal_summary = self.basal_system.get_basal_summary()
        inertia_summary = self.inertia_system.get_alignment_summary()
        physical_summary = self.physical_system.get_physical_summary()
        
        # エピソード記憶の統計
        memory_stats = {
            'total_memories': len(self.episodic_memory.memories),
            'strong_memories': len([m for m in self.episodic_memory.memories if m.salience > 0.7]),
            'positive_memories': len([m for m in self.episodic_memory.memories if m.valence > 0.3]),
            'negative_memories': len([m for m in self.episodic_memory.memories if m.valence < -0.3]),
            'avg_salience': sum(m.salience for m in self.episodic_memory.memories) / len(self.episodic_memory.memories) if self.episodic_memory.memories else 0.0
        }
        
        # 決定理由の統計
        decision_reasons = {}
        for log in self.decision_log[-20:]:  # 最新20件
            reason = log['reason']
            decision_reasons[reason] = decision_reasons.get(reason, 0) + 1
        
        return {
            'npc_name': self.npc_name,
            'personality_type': self.personality_type,
            'current_tick': self.current_tick,
            'basal_state': basal_summary,
            'inertia_state': inertia_summary,
            'physical_state': physical_summary,
            'memory_statistics': memory_stats,
            'recent_decision_patterns': decision_reasons,
            'integration_weights': self.episodic_influence_weights,
            'recent_interactions': self.interaction_history[-5:]
        }


# 使用例とテスト
def test_integrated_system():
    """統合システムのテスト"""
    print("=== エピソード記憶統合NPCシステムテスト ===")
    
    npc = IntegratedNPCWithEpisodic("ミミィ", "peppy")
    
    # シナリオ: プレイヤーとの初回遭遇
    print("\n--- シナリオ1: 初回遭遇 ---")
    environment = {
        'weather': 'sunny',
        'temperature': 22,
        'location': 'plaza',
        'time_period': 'morning'
    }
    
    events = [{
        'type': 'social_interaction',
        'involved_entities': ['PlayerA'],
        'success': True,
        'intensity': 0.8
    }]
    
    perceived = ['PlayerA']
    action1 = npc.update_and_select_action(environment, events, perceived)
    print(f"行動1: {action1}")
    
    # シナリオ: プレゼント受取
    print("\n--- シナリオ2: プレゼント受取 ---")
    events = [{
        'type': 'GIFT_RECEIVED',
        'involved_entities': ['PlayerA'],
        'success': True,
        'intensity': 0.9
    }]
    
    action2 = npc.update_and_select_action(environment, events, perceived)
    print(f"行動2: {action2}")
    
    # シナリオ: 再会（記憶の影響確認）
    print("\n--- シナリオ3: 記憶による再会 ---")
    environment['time_period'] = 'afternoon'
    events = []  # 新しい出来事なし
    
    action3 = npc.update_and_select_action(environment, events, perceived)
    print(f"行動3: {action3}")
    
    # 包括的サマリー
    print("\n--- 統合システム状態 ---")
    summary = npc.get_comprehensive_summary()
    print(f"記憶統計: {summary['memory_statistics']}")
    print(f"決定パターン: {summary['recent_decision_patterns']}")
    print(f"最近の行動: {summary['recent_interactions']}")


if __name__ == "__main__":
    test_integrated_system()
