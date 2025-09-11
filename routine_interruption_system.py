"""
routine_interruption_system.py
習慣行動中の他者遭遇による整合慣性と跳躍の衝突システム

習慣の整合慣性 vs 社交意味圧の力学を実装
"""

import random
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class RoutineState(Enum):
    """習慣実行状態"""
    NOT_IN_ROUTINE = "not_in_routine"
    ROUTINE_ACTIVE = "routine_active"
    ROUTINE_INTERRUPTED = "routine_interrupted"
    ROUTINE_RESUMED = "routine_resumed"
    ROUTINE_ABANDONED = "routine_abandoned"

class InterruptionType(Enum):
    """中断の種類"""
    ENTITY_ENCOUNTER = "entity_encounter"    # 誰かと出会った
    URGENT_NEED = "urgent_need"             # 緊急欲求
    PHYSICAL_FORCE = "physical_force"       # 物理的強制
    CURIOSITY = "curiosity"                 # 好奇心による

@dataclass
class RoutineContext:
    """習慣実行の文脈"""
    routine_id: str
    current_step: int
    total_steps: int
    start_time: float
    expected_duration: int  # 秒
    interruption_resistance: float  # 中断抵抗力 (0-1)
    completion_ratio: float  # 完了率 (0-1)

@dataclass
class InterruptionPressure:
    """中断圧力"""
    interruption_type: InterruptionType
    source_entity: Optional[str]
    pressure_strength: float  # 圧力強度 (0-1)
    urgency: float           # 緊急度 (0-1)
    social_weight: float     # 社交的重み (0-1)
    episodic_influence: float # エピソード記憶からの影響

@dataclass
class InterruptionDecision:
    """中断判定の結果"""
    should_interrupt: bool
    interruption_type: str
    decision_confidence: float
    reasoning: str
    alternative_action: Optional[str]
    resume_probability: float  # 後で習慣に戻る確率

class RoutineInterruptionSystem:
    """習慣中断システム"""
    
    def __init__(self, npc_name: str, personality_type: str):
        self.npc_name = npc_name
        self.personality_type = personality_type
        
        # 現在の習慣状態
        self.current_routine: Optional[RoutineContext] = None
        self.routine_state = RoutineState.NOT_IN_ROUTINE
        
        # 中断に関するパラメータ
        self.interruption_threshold = 0.4  # 中断閾値
        self.social_responsiveness = 0.6   # 社交反応性
        self.routine_commitment = 0.7      # 習慣へのコミット度
        
        # 履歴
        self.interruption_history: List[Dict] = []
        self.routine_completion_stats = {
            "completed": 0,
            "interrupted": 0,
            "abandoned": 0
        }
        
        self._adjust_parameters_by_personality()

    def _adjust_parameters_by_personality(self):
        """性格による中断パラメータ調整"""
        adjustments = {
            "peppy": {
                "interruption_threshold": 0.2,  # 中断されやすい
                "social_responsiveness": 0.9,   # 社交性高
                "routine_commitment": 0.4       # 習慣コミット低
            },
            "lazy": {
                "interruption_threshold": 0.6,  # 中断されにくい
                "social_responsiveness": 0.4,   # 社交性普通
                "routine_commitment": 0.8       # 習慣コミット高（楽だから）
            },
            "cranky": {
                "interruption_threshold": 0.7,  # 中断されにくい
                "social_responsiveness": 0.3,   # 社交性低
                "routine_commitment": 0.9       # 習慣コミット高
            },
            "normal": {
                "interruption_threshold": 0.4,  # 標準
                "social_responsiveness": 0.6,   # 標準
                "routine_commitment": 0.7       # 標準
            },
            "jock": {
                "interruption_threshold": 0.3,  # やや中断されやすい
                "social_responsiveness": 0.7,   # 社交性やや高
                "routine_commitment": 0.8       # 習慣コミット高（訓練重視）
            },
            "snooty": {
                "interruption_threshold": 0.5,  # 選択的
                "social_responsiveness": 0.5,   # 選択的社交
                "routine_commitment": 0.6       # 習慣コミット普通
            }
        }
        
        if self.personality_type in adjustments:
            adj = adjustments[self.personality_type]
            self.interruption_threshold = adj["interruption_threshold"]
            self.social_responsiveness = adj["social_responsiveness"]
            self.routine_commitment = adj["routine_commitment"]

    def start_routine(self, routine_id: str, total_steps: int, expected_duration: int):
        """習慣開始"""
        interruption_resistance = self.routine_commitment * random.uniform(0.8, 1.2)
        
        self.current_routine = RoutineContext(
            routine_id=routine_id,
            current_step=0,
            total_steps=total_steps,
            start_time=time.time(),
            expected_duration=expected_duration,
            interruption_resistance=min(1.0, interruption_resistance),
            completion_ratio=0.0
        )
        
        self.routine_state = RoutineState.ROUTINE_ACTIVE
        print(f"[{self.npc_name}] 習慣開始: {routine_id} (抵抗力: {interruption_resistance:.2f})")

    def process_encounter(self, encountered_entity: str, entity_familiarity: float,
                         episodic_memory_influence: float, current_context: Dict) -> InterruptionDecision:
        """他者遭遇時の処理"""
        
        if self.routine_state != RoutineState.ROUTINE_ACTIVE:
            return self._create_no_interruption_decision("習慣実行中ではない")
        
        # 中断圧力の計算
        interruption_pressure = self._calculate_encounter_pressure(
            encountered_entity, entity_familiarity, episodic_memory_influence, current_context
        )
        
        # 習慣の現在状況を考慮
        routine_resistance = self._calculate_routine_resistance()
        
        # 中断判定
        decision = self._make_interruption_decision(interruption_pressure, routine_resistance)
        
        # 結果に基づく状態更新
        self._apply_interruption_decision(decision, interruption_pressure)
        
        return decision

    def _calculate_encounter_pressure(self, entity: str, familiarity: float,
                                    episodic_influence: float, context: Dict) -> InterruptionPressure:
        """遭遇による中断圧力計算"""
        
        # 基本社交圧力（親しみやすさベース）
        base_social_pressure = self.social_responsiveness * familiarity
        
        # エピソード記憶による調整
        # ポジティブな記憶 → 圧力増加、ネガティブな記憶 → 圧力減少
        episodic_modifier = 1.0 + (episodic_influence * 0.5)
        social_pressure = base_social_pressure * episodic_modifier
        
        # 文脈による調整
        context_urgency = context.get('urgency', 0.3)  # 相手の緊急度
        context_social_weight = context.get('social_importance', 0.5)  # 社交的重要度
        
        # 総合圧力計算
        total_pressure = (social_pressure * 0.6 + 
                         context_urgency * 0.2 + 
                         context_social_weight * 0.2)
        
        return InterruptionPressure(
            interruption_type=InterruptionType.ENTITY_ENCOUNTER,
            source_entity=entity,
            pressure_strength=min(1.0, total_pressure),
            urgency=context_urgency,
            social_weight=context_social_weight,
            episodic_influence=episodic_influence
        )

    def _calculate_routine_resistance(self) -> float:
        """習慣の中断抵抗力計算"""
        if not self.current_routine:
            return 0.0
        
        # 基本抵抗力
        base_resistance = self.current_routine.interruption_resistance
        
        # 完了率による調整（開始直後と終了直前は抵抗が強い）
        progress = self.current_routine.completion_ratio
        progress_factor = 1.0
        
        if progress < 0.2:  # 開始直後
            progress_factor = 1.5  # 「せっかく始めたのに」効果
        elif progress > 0.8:  # 終了直前
            progress_factor = 2.0  # 「もうすぐ終わるのに」効果
        else:  # 中盤
            progress_factor = 0.8  # 中断しやすい
        
        # 時間経過による疲労（長時間の習慣は中断されやすくなる）
        elapsed = time.time() - self.current_routine.start_time
        expected = self.current_routine.expected_duration
        fatigue_factor = max(0.5, 1.0 - (elapsed / expected) * 0.3)
        
        total_resistance = base_resistance * progress_factor * fatigue_factor
        return min(1.0, total_resistance)

    def _make_interruption_decision(self, pressure: InterruptionPressure, 
                                   resistance: float) -> InterruptionDecision:
        """中断判定の実行"""
        
        # 圧力 vs 抵抗の力学
        pressure_vs_resistance = pressure.pressure_strength - resistance
        
        # 閾値による判定
        should_interrupt = pressure_vs_resistance > self.interruption_threshold
        
        # 決定の信頼度
        confidence = abs(pressure_vs_resistance)
        
        # 中断理由の生成
        reasoning = self._generate_interruption_reasoning(pressure, resistance, should_interrupt)
        
        # 代替行動の決定
        alternative_action = self._determine_alternative_action(pressure, should_interrupt)
        
        # 復帰確率の計算
        resume_probability = self._calculate_resume_probability(pressure, resistance, should_interrupt)
        
        return InterruptionDecision(
            should_interrupt=should_interrupt,
            interruption_type=pressure.interruption_type.value,
            decision_confidence=min(1.0, confidence),
            reasoning=reasoning,
            alternative_action=alternative_action,
            resume_probability=resume_probability
        )

    def _generate_interruption_reasoning(self, pressure: InterruptionPressure, 
                                       resistance: float, interrupted: bool) -> str:
        """中断判定の理由生成"""
        
        if not interrupted:
            if resistance > 0.8:
                return f"習慣への強いコミット（抵抗力{resistance:.2f}）により継続"
            elif pressure.pressure_strength < 0.3:
                return f"社交圧力が不足（{pressure.pressure_strength:.2f}）で習慣継続"
            else:
                return f"微妙な判定だが習慣を優先（差分{resistance - pressure.pressure_strength:.2f}）"
        else:
            if pressure.episodic_influence > 0.5:
                return f"エピソード記憶（{pressure.episodic_influence:.2f}）が強く作用し社交を選択"
            elif pressure.urgency > 0.7:
                return f"相手の緊急度（{pressure.urgency:.2f}）が高く中断"
            elif pressure.social_weight > 0.7:
                return f"社交的重要度（{pressure.social_weight:.2f}）により習慣中断"
            else:
                return f"総合的な社交圧力（{pressure.pressure_strength:.2f}）が抵抗力（{resistance:.2f}）を上回り中断"

    def _determine_alternative_action(self, pressure: InterruptionPressure, 
                                     interrupted: bool) -> Optional[str]:
        """代替行動の決定"""
        
        if not interrupted:
            return None
        
        # エンティティとの関係性に基づく行動選択
        if pressure.episodic_influence > 0.5:
            return "approach_with_enthusiasm"  # ポジティブな記憶
        elif pressure.episodic_influence < -0.3:
            return "cautious_greeting"  # ネガティブな記憶
        elif pressure.urgency > 0.7:
            return "immediate_response"  # 緊急対応
        elif pressure.social_weight > 0.6:
            return "polite_interaction"  # 丁寧な対応
        else:
            return "casual_greeting"  # カジュアルな挨拶

    def _calculate_resume_probability(self, pressure: InterruptionPressure, 
                                     resistance: float, interrupted: bool) -> float:
        """習慣復帰確率の計算"""
        
        if not interrupted:
            return 1.0  # 中断されてないので継続
        
        # 習慣へのコミット度が高いほど復帰しやすい
        base_resume = self.routine_commitment
        
        # 完了率が高いほど復帰したい
        if self.current_routine:
            progress_bonus = self.current_routine.completion_ratio * 0.3
        else:
            progress_bonus = 0.0
        
        # 中断理由による調整
        if pressure.urgency > 0.8:
            urgency_penalty = -0.4  # 緊急事態なら復帰困難
        else:
            urgency_penalty = 0.0
        
        # エピソード記憶の影響
        if pressure.episodic_influence > 0.6:
            episodic_penalty = -0.2  # 強いポジティブ記憶なら社交に没頭
        else:
            episodic_penalty = 0.0
        
        resume_prob = base_resume + progress_bonus + urgency_penalty + episodic_penalty
        return max(0.0, min(1.0, resume_prob))

    def _apply_interruption_decision(self, decision: InterruptionDecision, 
                                    pressure: InterruptionPressure):
        """中断判定結果の適用"""
        
        if decision.should_interrupt:
            self.routine_state = RoutineState.ROUTINE_INTERRUPTED
            self.routine_completion_stats["interrupted"] += 1
            
            print(f"[{self.npc_name}] 習慣中断: {decision.reasoning}")
            print(f"[{self.npc_name}] 代替行動: {decision.alternative_action}")
            print(f"[{self.npc_name}] 復帰予測: {decision.resume_probability:.2f}")
        else:
            print(f"[{self.npc_name}] 習慣継続: {decision.reasoning}")
        
        # 履歴記録
        self.interruption_history.append({
            'timestamp': time.time(),
            'routine_id': self.current_routine.routine_id if self.current_routine else None,
            'source_entity': pressure.source_entity,
            'interrupted': decision.should_interrupt,
            'pressure_strength': pressure.pressure_strength,
            'reasoning': decision.reasoning,
            'alternative_action': decision.alternative_action
        })

    def _create_no_interruption_decision(self, reason: str) -> InterruptionDecision:
        """中断なし判定の生成"""
        return InterruptionDecision(
            should_interrupt=False,
            interruption_type="none",
            decision_confidence=1.0,
            reasoning=reason,
            alternative_action=None,
            resume_probability=1.0
        )

    def attempt_routine_resume(self, context: Dict) -> bool:
        """習慣復帰の試み"""
        
        if self.routine_state != RoutineState.ROUTINE_INTERRUPTED:
            return False
        
        if not self.current_routine:
            return False
        
        # 最新の中断記録から復帰確率を取得
        if self.interruption_history:
            last_interruption = self.interruption_history[-1]
            resume_probability = last_interruption.get('resume_probability', 0.5)
        else:
            resume_probability = 0.5
        
        # 文脈による調整
        distraction_level = context.get('distraction_level', 0.3)
        time_elapsed = context.get('time_since_interruption', 0)
        
        # 時間経過により復帰意欲減退
        time_decay = max(0.0, 1.0 - (time_elapsed / 300))  # 5分で半減
        
        adjusted_probability = resume_probability * (1.0 - distraction_level) * time_decay
        
        if random.random() < adjusted_probability:
            self.routine_state = RoutineState.ROUTINE_RESUMED
            print(f"[{self.npc_name}] 習慣復帰: {self.current_routine.routine_id}")
            return True
        else:
            self.routine_state = RoutineState.ROUTINE_ABANDONED
            self.routine_completion_stats["abandoned"] += 1
            print(f"[{self.npc_name}] 習慣放棄: {self.current_routine.routine_id}")
            self.current_routine = None
            return False

    def update_routine_progress(self, steps_completed: int):
        """習慣進行度の更新"""
        if not self.current_routine:
            return
        
        self.current_routine.current_step = steps_completed
        self.current_routine.completion_ratio = steps_completed / self.current_routine.total_steps
        
        # 完了判定
        if self.current_routine.completion_ratio >= 1.0:
            self.routine_state = RoutineState.NOT_IN_ROUTINE
            self.routine_completion_stats["completed"] += 1
            print(f"[{self.npc_name}] 習慣完了: {self.current_routine.routine_id}")
            self.current_routine = None

    def get_interruption_summary(self) -> Dict:
        """中断システムの状態要約"""
        
        # 最近の中断統計
        recent_interruptions = self.interruption_history[-10:]
        interruption_rate = len([h for h in recent_interruptions if h['interrupted']]) / len(recent_interruptions) if recent_interruptions else 0.0
        
        # エンティティ別中断統計
        entity_interruption_stats = {}
        for history in self.interruption_history:
            entity = history['source_entity']
            if entity:
                if entity not in entity_interruption_stats:
                    entity_interruption_stats[entity] = {'total': 0, 'interrupted': 0}
                entity_interruption_stats[entity]['total'] += 1
                if history['interrupted']:
                    entity_interruption_stats[entity]['interrupted'] += 1
        
        return {
            'npc_name': self.npc_name,
            'personality_type': self.personality_type,
            'current_routine_state': self.routine_state.value,
            'current_routine': {
                'routine_id': self.current_routine.routine_id if self.current_routine else None,
                'progress': self.current_routine.completion_ratio if self.current_routine else 0.0,
                'resistance': self.current_routine.interruption_resistance if self.current_routine else 0.0
            } if self.current_routine else None,
            'interruption_parameters': {
                'threshold': self.interruption_threshold,
                'social_responsiveness': self.social_responsiveness,
                'routine_commitment': self.routine_commitment
            },
            'completion_statistics': self.routine_completion_stats.copy(),
            'recent_interruption_rate': interruption_rate,
            'entity_influence': {
                entity: stats['interrupted'] / stats['total'] 
                for entity, stats in entity_interruption_stats.items()
            },
            'recent_interruptions': recent_interruptions[-3:]  # 最新3件
        }


# 使用例とテスト
def test_routine_interruption_system():
    """習慣中断システムのテスト"""
    print("=== 習慣中断システムテスト ===")
    
    # 性格の異なるNPCでテスト
    personalities = ["peppy", "cranky", "lazy"]
    
    for personality in personalities:
        print(f"\n--- {personality.upper()} NPCのテスト ---")
        
        interrupt_system = RoutineInterruptionSystem(f"NPC_{personality}", personality)
        
        # 習慣開始
        interrupt_system.start_routine("morning_exercise", 5, 300)  # 5分間の朝の運動
        
        # 進行度更新
        interrupt_system.update_routine_progress(2)  # 40%完了
        
        # 他者遭遇シミュレーション
        encounters = [
            ("PlayerA", 0.8, 0.6, {"urgency": 0.3, "social_importance": 0.7}),  # 親しいプレイヤー
            ("VillagerB", 0.4, -0.2, {"urgency": 0.8, "social_importance": 0.5}),  # あまり親しくない住民、緊急事態
            ("VillagerC", 0.9, 0.8, {"urgency": 0.2, "social_importance": 0.9})   # 非常に親しい住民、重要な話
        ]
        
        for entity, familiarity, episodic_influence, context in encounters:
            print(f"\n  {entity}と遭遇 (親密度: {familiarity:.1f}, 記憶影響: {episodic_influence:.1f})")
            
            decision = interrupt_system.process_encounter(entity, familiarity, episodic_influence, context)
            
            print(f"    判定: {'中断' if decision.should_interrupt else '継続'}")
            print(f"    信頼度: {decision.decision_confidence:.2f}")
            print(f"    理由: {decision.reasoning}")
            
            if decision.should_interrupt:
                print(f"    代替行動: {decision.alternative_action}")
                
                # 復帰試行
                resume_context = {"distraction_level": 0.2, "time_since_interruption": 120}
                resumed = interrupt_system.attempt_routine_resume(resume_context)
                print(f"    復帰: {'成功' if resumed else '失敗'}")
                
                if not resumed:
                    break  # 習慣放棄なので終了
        
        # 統計表示
        summary = interrupt_system.get_interruption_summary()
        print(f"\n  完了統計: {summary['completion_statistics']}")
        print(f"  中断率: {summary['recent_interruption_rate']:.2f}")


if __name__ == "__main__":
    test_routine_interruption_system()
