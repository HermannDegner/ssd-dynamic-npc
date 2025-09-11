"""
ssd_universal_evaluation_engine.py
構造主観力学（SSD）汎用評価エンジン

あらゆる構造・意味圧・整合・跳躍現象を統一的に評価する汎用DLL化対応エンジン
"""

import ctypes
import json
import math
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum, IntEnum
import numpy as np

# DLL互換性のための型定義
class SSdReturnCode(IntEnum):
    """SSDエンジンリターンコード"""
    SUCCESS = 0
    ERROR_INVALID_INPUT = -1
    ERROR_CALCULATION_FAILED = -2
    ERROR_INSUFFICIENT_DATA = -3
    ERROR_MEMORY_ALLOCATION = -4
    WARNING_LOW_CONFIDENCE = 1

@dataclass
class UniversalStructure:
    """汎用構造定義"""
    structure_id: str
    structure_type: str          # "physical", "biological", "psychological", "social", "digital"
    dimension_count: int         # 次元数
    stability_index: float       # 安定性指標 (0-1)
    complexity_level: float      # 複雑性レベル (0-1)
    dynamic_properties: Dict[str, float]  # 動的特性
    constraint_matrix: List[List[float]]  # 制約行列
    metadata: Dict[str, Any]

@dataclass
class UniversalMeaningPressure:
    """汎用意味圧定義"""
    pressure_id: str
    source_type: str             # "external", "internal", "systemic", "emergent"
    magnitude: float             # 強度 (0-1)
    direction_vector: List[float] # 方向ベクトル
    frequency: float             # 周波数 (Hz)
    duration: float              # 持続時間 (秒)
    propagation_speed: float     # 伝播速度
    decay_function: str          # "exponential", "linear", "logarithmic", "constant"
    interaction_matrix: List[List[float]]  # 相互作用行列

@dataclass
class UniversalAlignment:
    """汎用整合定義"""
    alignment_id: str
    alignment_type: str          # "mechanical", "thermal", "chemical", "biological", "cognitive", "social"
    efficiency: float            # 整合効率 (0-1)
    energy_cost: float          # エネルギーコスト
    time_constant: float        # 時定数
    resonance_frequency: float  # 共鳴周波数
    damping_coefficient: float  # 減衰係数
    nonlinearity_factor: float  # 非線形性

@dataclass
class UniversalJump:
    """汎用跳躍定義"""
    jump_id: str
    trigger_threshold: float     # 発火閾値
    jump_magnitude: float       # 跳躍の大きさ
    direction_change: List[float] # 方向変化ベクトル
    irreversibility: float      # 不可逆性 (0-1)
    cascade_probability: float  # 連鎖確率
    creation_potential: float   # 創造ポテンシャル
    destruction_risk: float     # 破壊リスク

@dataclass
class EvaluationContext:
    """評価文脈"""
    context_id: str
    domain: str                  # "physics", "chemistry", "biology", "psychology", "sociology", "economics", "AI"
    scale_level: str            # "quantum", "atomic", "molecular", "cellular", "organism", "group", "society", "civilization"
    time_scale: float           # 時間スケール (秒)
    space_scale: float          # 空間スケール (メートル)
    observer_position: List[float] # 観測者位置
    measurement_precision: float   # 測定精度
    environmental_factors: Dict[str, float]

@dataclass
class UniversalEvaluationResult:
    """汎用評価結果"""
    evaluation_id: str
    return_code: SSdReturnCode
    
    # 構造分析結果
    structure_stability: float
    structure_complexity: float
    structure_adaptability: float
    
    # 意味圧分析結果
    pressure_magnitude: float
    pressure_coherence: float
    pressure_sustainability: float
    
    # 整合分析結果
    alignment_strength: float
    alignment_efficiency: float
    alignment_durability: float
    
    # 跳躍分析結果
    jump_probability: float
    jump_direction_prediction: List[float]
    jump_impact_estimation: float
    
    # 総合指標
    system_health: float         # システム健全性
    evolution_potential: float   # 進化ポテンシャル
    stability_resilience: float  # 安定性回復力
    
    # メタ情報
    calculation_confidence: float
    computational_cost: float
    prediction_horizon: float   # 予測可能期間
    explanation_tree: Dict      # 説明木構造
    
    # 警告・推奨事項
    warnings: List[str]
    recommendations: List[str]

class UniversalSSDEngine:
    """SSD汎用評価エンジン"""
    
    def __init__(self, engine_config: Dict[str, Any] = None):
        self.engine_id = f"ssd_engine_{int(time.time())}"
        self.version = "1.0.0"
        
        # エンジン設定
        self.config = engine_config or self._default_config()
        
        # 計算プール
        self.calculation_pool = {}
        self.cache_enabled = self.config.get('enable_cache', True)
        self.max_cache_size = self.config.get('max_cache_size', 1000)
        
        # ドメイン特化係数
        self.domain_coefficients = self._initialize_domain_coefficients()
        
        # スケール補正関数
        self.scale_functions = self._initialize_scale_functions()
        
        # 統計情報
        self.evaluation_count = 0
        self.performance_metrics = {
            'total_evaluations': 0,
            'average_computation_time': 0.0,
            'cache_hit_rate': 0.0,
            'accuracy_score': 0.0
        }

    def _default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'precision_level': 'high',      # 'low', 'medium', 'high', 'ultra'
            'calculation_mode': 'balanced', # 'fast', 'balanced', 'accurate'
            'enable_cache': True,
            'enable_prediction': True,
            'enable_explanation': True,
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'time_limit_ms': 5000,
            'parallel_processing': True,
            'memory_limit_mb': 512
        }

    def _initialize_domain_coefficients(self) -> Dict[str, Dict[str, float]]:
        """ドメイン特化係数の初期化"""
        return {
            'physics': {
                'structure_weight': 1.0,
                'pressure_weight': 1.0,
                'alignment_weight': 0.9,
                'jump_weight': 0.8,
                'time_scale_factor': 1.0,
                'space_scale_factor': 1.0
            },
            'chemistry': {
                'structure_weight': 0.9,
                'pressure_weight': 1.0,
                'alignment_weight': 1.0,
                'jump_weight': 0.9,
                'time_scale_factor': 1e3,
                'space_scale_factor': 1e-10
            },
            'biology': {
                'structure_weight': 0.8,
                'pressure_weight': 0.9,
                'alignment_weight': 1.0,
                'jump_weight': 1.0,
                'time_scale_factor': 1e6,
                'space_scale_factor': 1e-6
            },
            'psychology': {
                'structure_weight': 0.7,
                'pressure_weight': 1.0,
                'alignment_weight': 0.8,
                'jump_weight': 1.0,
                'time_scale_factor': 1e0,
                'space_scale_factor': 1e0
            },
            'sociology': {
                'structure_weight': 0.6,
                'pressure_weight': 0.8,
                'alignment_weight': 0.9,
                'jump_weight': 1.0,
                'time_scale_factor': 1e7,
                'space_scale_factor': 1e3
            },
            'economics': {
                'structure_weight': 0.5,
                'pressure_weight': 1.0,
                'alignment_weight': 0.7,
                'jump_weight': 1.0,
                'time_scale_factor': 1e6,
                'space_scale_factor': 1e6
            },
            'AI': {
                'structure_weight': 0.8,
                'pressure_weight': 0.9,
                'alignment_weight': 1.0,
                'jump_weight': 0.9,
                'time_scale_factor': 1e-3,
                'space_scale_factor': 1e0
            }
        }

    def _initialize_scale_functions(self) -> Dict[str, callable]:
        """スケール補正関数の初期化"""
        return {
            'quantum': lambda x: x * math.exp(-x**2 / 1e-20),
            'atomic': lambda x: x * (1 + 0.1 * math.sin(x * 1e12)),
            'molecular': lambda x: x * (1 + 0.05 * math.log(1 + x)),
            'cellular': lambda x: x * (1 - 0.1 * math.exp(-x / 1e-6)),
            'organism': lambda x: x * (1 + 0.2 * math.tanh(x * 1e3)),
            'group': lambda x: x * (1 + 0.3 * math.sigmoid(x - 0.5)),
            'society': lambda x: x * (1 + 0.1 * math.pow(x, 0.7)),
            'civilization': lambda x: x * (1 + 0.05 * math.sqrt(x))
        }

    def evaluate_universal_system(self, 
                                 structures: List[UniversalStructure],
                                 meaning_pressures: List[UniversalMeaningPressure],
                                 context: EvaluationContext) -> UniversalEvaluationResult:
        """汎用システム評価のメイン関数"""
        
        start_time = time.time()
        evaluation_id = f"eval_{self.evaluation_count}_{int(start_time * 1000)}"
        
        try:
            # 1. 入力検証
            validation_result = self._validate_inputs(structures, meaning_pressures, context)
            if validation_result != SSdReturnCode.SUCCESS:
                return self._create_error_result(evaluation_id, validation_result)
            
            # 2. キャッシュ確認
            cache_key = self._generate_cache_key(structures, meaning_pressures, context)
            if self.cache_enabled and cache_key in self.calculation_pool:
                cached_result = self.calculation_pool[cache_key]
                cached_result.evaluation_id = evaluation_id
                return cached_result
            
            # 3. ドメイン係数の適用
            domain_coeff = self.domain_coefficients.get(context.domain, self.domain_coefficients['physics'])
            
            # 4. 構造分析
            structure_analysis = self._analyze_structures(structures, context, domain_coeff)
            
            # 5. 意味圧分析
            pressure_analysis = self._analyze_meaning_pressures(meaning_pressures, context, domain_coeff)
            
            # 6. 整合分析
            alignment_analysis = self._analyze_alignment(structures, meaning_pressures, context, domain_coeff)
            
            # 7. 跳躍分析
            jump_analysis = self._analyze_jump_potential(structures, meaning_pressures, context, domain_coeff)
            
            # 8. 統合評価
            integrated_result = self._integrate_analyses(
                structure_analysis, pressure_analysis, alignment_analysis, jump_analysis, context
            )
            
            # 9. 説明生成
            explanation_tree = self._generate_explanation_tree(
                structure_analysis, pressure_analysis, alignment_analysis, jump_analysis, integrated_result
            )
            
            # 10. 結果構築
            result = UniversalEvaluationResult(
                evaluation_id=evaluation_id,
                return_code=SSdReturnCode.SUCCESS,
                **structure_analysis,
                **pressure_analysis,
                **alignment_analysis,
                **jump_analysis,
                **integrated_result,
                calculation_confidence=self._calculate_confidence(structures, meaning_pressures, context),
                computational_cost=time.time() - start_time,
                prediction_horizon=self._estimate_prediction_horizon(context),
                explanation_tree=explanation_tree,
                warnings=self._generate_warnings(integrated_result),
                recommendations=self._generate_recommendations(integrated_result, context)
            )
            
            # 11. キャッシュ保存
            if self.cache_enabled and len(self.calculation_pool) < self.max_cache_size:
                self.calculation_pool[cache_key] = result
            
            # 12. 統計更新
            self._update_performance_metrics(result)
            
            return result
            
        except Exception as e:
            return self._create_error_result(evaluation_id, SSdReturnCode.ERROR_CALCULATION_FAILED, str(e))

    def _validate_inputs(self, structures: List[UniversalStructure], 
                        meaning_pressures: List[UniversalMeaningPressure],
                        context: EvaluationContext) -> SSdReturnCode:
        """入力検証"""
        
        if not structures:
            return SSdReturnCode.ERROR_INVALID_INPUT
        
        if not meaning_pressures:
            return SSdReturnCode.ERROR_INVALID_INPUT
        
        # 構造の妥当性検証
        for structure in structures:
            if not (0 <= structure.stability_index <= 1):
                return SSdReturnCode.ERROR_INVALID_INPUT
            if not (0 <= structure.complexity_level <= 1):
                return SSdReturnCode.ERROR_INVALID_INPUT
        
        # 意味圧の妥当性検証
        for pressure in meaning_pressures:
            if not (0 <= pressure.magnitude <= 1):
                return SSdReturnCode.ERROR_INVALID_INPUT
            if pressure.frequency < 0:
                return SSdReturnCode.ERROR_INVALID_INPUT
        
        return SSdReturnCode.SUCCESS

    def _analyze_structures(self, structures: List[UniversalStructure], 
                           context: EvaluationContext, 
                           domain_coeff: Dict[str, float]) -> Dict[str, float]:
        """構造分析"""
        
        if not structures:
            return {'structure_stability': 0.0, 'structure_complexity': 0.0, 'structure_adaptability': 0.0}
        
        # 安定性分析
        stability_scores = []
        for structure in structures:
            base_stability = structure.stability_index
            
            # 制約行列による補正
            if structure.constraint_matrix:
                constraint_effect = self._analyze_constraint_matrix(structure.constraint_matrix)
                stability = base_stability * (1 + constraint_effect * 0.2)
            else:
                stability = base_stability
            
            # ドメイン補正
            stability *= domain_coeff['structure_weight']
            
            # スケール補正
            scale_func = self.scale_functions.get(context.scale_level, lambda x: x)
            stability = scale_func(stability)
            
            stability_scores.append(min(1.0, max(0.0, stability)))
        
        # 複雑性分析
        complexity_scores = []
        for structure in structures:
            base_complexity = structure.complexity_level
            
            # 次元数による補正
            dimension_factor = 1 + math.log(structure.dimension_count) * 0.1
            complexity = base_complexity * dimension_factor
            
            # 動的特性による補正
            if structure.dynamic_properties:
                dynamics_factor = sum(structure.dynamic_properties.values()) / len(structure.dynamic_properties)
                complexity *= (1 + dynamics_factor * 0.3)
            
            complexity_scores.append(min(1.0, max(0.0, complexity)))
        
        # 適応性分析（安定性と複雑性のバランス）
        adaptability_scores = []
        for i, structure in enumerate(structures):
            stability = stability_scores[i]
            complexity = complexity_scores[i]
            
            # 適応性は中程度の安定性と適度な複雑性のバランス
            optimal_stability = 0.6
            optimal_complexity = 0.7
            
            stability_deviation = abs(stability - optimal_stability)
            complexity_deviation = abs(complexity - optimal_complexity)
            
            adaptability = 1.0 - (stability_deviation + complexity_deviation) / 2.0
            adaptability_scores.append(max(0.0, adaptability))
        
        return {
            'structure_stability': sum(stability_scores) / len(stability_scores),
            'structure_complexity': sum(complexity_scores) / len(complexity_scores),
            'structure_adaptability': sum(adaptability_scores) / len(adaptability_scores)
        }

    def _analyze_meaning_pressures(self, meaning_pressures: List[UniversalMeaningPressure],
                                  context: EvaluationContext,
                                  domain_coeff: Dict[str, float]) -> Dict[str, float]:
        """意味圧分析"""
        
        if not meaning_pressures:
            return {'pressure_magnitude': 0.0, 'pressure_coherence': 0.0, 'pressure_sustainability': 0.0}
        
        # 強度分析
        magnitude_scores = []
        for pressure in meaning_pressures:
            base_magnitude = pressure.magnitude
            
            # 周波数による補正
            frequency_factor = 1 + math.log(1 + pressure.frequency) * 0.1
            magnitude = base_magnitude * frequency_factor
            
            # 持続時間による補正
            duration_factor = min(2.0, 1 + pressure.duration / 3600)  # 1時間でmax
            magnitude *= duration_factor
            
            # 伝播速度による補正
            propagation_factor = 1 + pressure.propagation_speed * 0.1
            magnitude *= propagation_factor
            
            # ドメイン補正
            magnitude *= domain_coeff['pressure_weight']
            
            magnitude_scores.append(min(1.0, max(0.0, magnitude)))
        
        # 一貫性分析（方向ベクトルの整合性）
        coherence_score = self._calculate_vector_coherence([p.direction_vector for p in meaning_pressures])
        
        # 持続可能性分析
        sustainability_scores = []
        for pressure in meaning_pressures:
            # 減衰関数による持続性評価
            if pressure.decay_function == 'constant':
                sustainability = 1.0
            elif pressure.decay_function == 'exponential':
                sustainability = 0.3
            elif pressure.decay_function == 'linear':
                sustainability = 0.6
            elif pressure.decay_function == 'logarithmic':
                sustainability = 0.8
            else:
                sustainability = 0.5
            
            # 相互作用行列による補正
            if pressure.interaction_matrix:
                interaction_strength = self._analyze_interaction_matrix(pressure.interaction_matrix)
                sustainability *= (1 + interaction_strength * 0.3)
            
            sustainability_scores.append(min(1.0, max(0.0, sustainability)))
        
        return {
            'pressure_magnitude': sum(magnitude_scores) / len(magnitude_scores),
            'pressure_coherence': coherence_score,
            'pressure_sustainability': sum(sustainability_scores) / len(sustainability_scores)
        }

    def _analyze_alignment(self, structures: List[UniversalStructure],
                          meaning_pressures: List[UniversalMeaningPressure],
                          context: EvaluationContext,
                          domain_coeff: Dict[str, float]) -> Dict[str, float]:
        """整合分析"""
        
        # 構造と意味圧の整合性を分析
        alignment_strength_scores = []
        alignment_efficiency_scores = []
        alignment_durability_scores = []
        
        for structure in structures:
            for pressure in meaning_pressures:
                # 強度：構造の安定性と意味圧の大きさのマッチング
                strength = self._calculate_alignment_strength(structure, pressure)
                alignment_strength_scores.append(strength)
                
                # 効率：エネルギーコストと効果のバランス
                efficiency = self._calculate_alignment_efficiency(structure, pressure, context)
                alignment_efficiency_scores.append(efficiency)
                
                # 持久性：時間的な持続性
                durability = self._calculate_alignment_durability(structure, pressure, context)
                alignment_durability_scores.append(durability)
        
        return {
            'alignment_strength': sum(alignment_strength_scores) / len(alignment_strength_scores) if alignment_strength_scores else 0.0,
            'alignment_efficiency': sum(alignment_efficiency_scores) / len(alignment_efficiency_scores) if alignment_efficiency_scores else 0.0,
            'alignment_durability': sum(alignment_durability_scores) / len(alignment_durability_scores) if alignment_durability_scores else 0.0
        }

    def _analyze_jump_potential(self, structures: List[UniversalStructure],
                               meaning_pressures: List[UniversalMeaningPressure],
                               context: EvaluationContext,
                               domain_coeff: Dict[str, float]) -> Dict[str, float]:
        """跳躍分析"""
        
        # 跳躍確率の計算
        jump_probabilities = []
        jump_directions = []
        jump_impacts = []
        
        for structure in structures:
            structure_limit = self._calculate_structure_limit(structure)
            
            for pressure in meaning_pressures:
                pressure_intensity = pressure.magnitude
                
                # 跳躍確率：意味圧が構造限界を超える確率
                if pressure_intensity > structure_limit:
                    excess = pressure_intensity - structure_limit
                    probability = min(1.0, excess * 2.0)
                else:
                    probability = 0.0
                
                jump_probabilities.append(probability)
                
                # 跳躍方向：意味圧の方向ベクトル
                jump_directions.append(pressure.direction_vector)
                
                # 跳躍インパクト：構造変化の大きさ
                impact = self._calculate_jump_impact(structure, pressure)
                jump_impacts.append(impact)
        
        # 平均跳躍方向の計算
        if jump_directions:
            avg_direction = self._calculate_average_direction(jump_directions, jump_probabilities)
        else:
            avg_direction = [0.0, 0.0, 0.0]
        
        return {
            'jump_probability': sum(jump_probabilities) / len(jump_probabilities) if jump_probabilities else 0.0,
            'jump_direction_prediction': avg_direction,
            'jump_impact_estimation': sum(jump_impacts) / len(jump_impacts) if jump_impacts else 0.0
        }

    def _integrate_analyses(self, structure_analysis: Dict, pressure_analysis: Dict,
                           alignment_analysis: Dict, jump_analysis: Dict,
                           context: EvaluationContext) -> Dict[str, float]:
        """分析結果の統合"""
        
        # システム健全性：全体的な安定性と機能性
        system_health = (
            structure_analysis['structure_stability'] * 0.3 +
            alignment_analysis['alignment_strength'] * 0.3 +
            alignment_analysis['alignment_efficiency'] * 0.2 +
            (1.0 - jump_analysis['jump_probability']) * 0.2
        )
        
        # 進化ポテンシャル：成長と変化の可能性
        evolution_potential = (
            structure_analysis['structure_adaptability'] * 0.4 +
            pressure_analysis['pressure_sustainability'] * 0.3 +
            jump_analysis['jump_probability'] * 0.3
        )
        
        # 安定性回復力：外乱からの回復能力
        stability_resilience = (
            structure_analysis['structure_stability'] * 0.4 +
            alignment_analysis['alignment_durability'] * 0.3 +
            pressure_analysis['pressure_coherence'] * 0.3
        )
        
        return {
            'system_health': min(1.0, max(0.0, system_health)),
            'evolution_potential': min(1.0, max(0.0, evolution_potential)),
            'stability_resilience': min(1.0, max(0.0, stability_resilience))
        }

    # ヘルパーメソッド（簡略化版）
    def _analyze_constraint_matrix(self, matrix: List[List[float]]) -> float:
        """制約行列分析"""
        if not matrix or not matrix[0]:
            return 0.0
        return sum(sum(row) for row in matrix) / (len(matrix) * len(matrix[0]))

    def _analyze_interaction_matrix(self, matrix: List[List[float]]) -> float:
        """相互作用行列分析"""
        return self._analyze_constraint_matrix(matrix)

    def _calculate_vector_coherence(self, vectors: List[List[float]]) -> float:
        """ベクトル一貫性計算"""
        if not vectors or len(vectors) < 2:
            return 1.0
        
        # 簡易版：ベクトル間の平均コサイン類似度
        coherence_sum = 0.0
        pairs = 0
        
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                similarity = self._cosine_similarity(vectors[i], vectors[j])
                coherence_sum += similarity
                pairs += 1
        
        return coherence_sum / pairs if pairs > 0 else 1.0

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """コサイン類似度計算"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def _calculate_alignment_strength(self, structure: UniversalStructure, 
                                     pressure: UniversalMeaningPressure) -> float:
        """整合強度計算"""
        # 構造の安定性と意味圧のマッチング
        stability_match = 1.0 - abs(structure.stability_index - pressure.magnitude)
        complexity_factor = 1.0 - structure.complexity_level * 0.3
        return stability_match * complexity_factor

    def _calculate_alignment_efficiency(self, structure: UniversalStructure,
                                       pressure: UniversalMeaningPressure,
                                       context: EvaluationContext) -> float:
        """整合効率計算"""
        # 簡易版：複雑性が低いほど効率的
        base_efficiency = 1.0 - structure.complexity_level * 0.5
        pressure_factor = 1.0 - pressure.magnitude * 0.2
        return base_efficiency * pressure_factor

    def _calculate_alignment_durability(self, structure: UniversalStructure,
                                       pressure: UniversalMeaningPressure,
                                       context: EvaluationContext) -> float:
        """整合持久性計算"""
        # 構造の安定性と意味圧の持続性
        structure_durability = structure.stability_index
        pressure_persistence = 1.0 if pressure.decay_function == 'constant' else 0.5
        return structure_durability * pressure_persistence

    def _calculate_structure_limit(self, structure: UniversalStructure) -> float:
        """構造限界計算"""
        # 安定性の逆数（不安定なほど限界が低い）
        return structure.stability_index * 0.8

    def _calculate_jump_impact(self, structure: UniversalStructure,
                              pressure: UniversalMeaningPressure) -> float:
        """跳躍インパクト計算"""
        # 構造の複雑性と意味圧の強度の積
        return structure.complexity_level * pressure.magnitude

    def _calculate_average_direction(self, directions: List[List[float]], 
                                    weights: List[float]) -> List[float]:
        """重み付き平均方向計算"""
        if not directions or not weights:
            return [0.0, 0.0, 0.0]
        
        dimension = len(directions[0])
        weighted_sum = [0.0] * dimension
        weight_sum = sum(weights)
        
        for direction, weight in zip(directions, weights):
            for i in range(min(dimension, len(direction))):
                weighted_sum[i] += direction[i] * weight
        
        if weight_sum > 0:
            return [ws / weight_sum for ws in weighted_sum]
        else:
            return [0.0] * dimension

    def _generate_cache_key(self, structures: List[UniversalStructure],
                           meaning_pressures: List[UniversalMeaningPressure],
                           context: EvaluationContext) -> str:
        """キャッシュキー生成"""
        structure_hash = hash(tuple(s.structure_id for s in structures))
        pressure_hash = hash(tuple(p.pressure_id for p in meaning_pressures))
        context_hash = hash((context.domain, context.scale_level, context.time_scale))
        return f"{structure_hash}_{pressure_hash}_{context_hash}"

    def _calculate_confidence(self, structures: List[UniversalStructure],
                             meaning_pressures: List[UniversalMeaningPressure],
                             context: EvaluationContext) -> float:
        """計算信頼度評価"""
        data_quality = min(len(structures) / 5.0, 1.0)  # 5構造で最高品質
        pressure_quality = min(len(meaning_pressures) / 3.0, 1.0)  # 3意味圧で最高品質
        precision_factor = {'low': 0.5, 'medium': 0.7, 'high': 0.9, 'ultra': 1.0}[self.config['precision_level']]
        
        return (data_quality + pressure_quality + precision_factor) / 3.0

    def _estimate_prediction_horizon(self, context: EvaluationContext) -> float:
        """予測可能期間推定"""
        scale_factors = {
            'quantum': 1e-15,
            'atomic': 1e-12,
            'molecular': 1e-9,
            'cellular': 1e-3,
            'organism': 1e3,
            'group': 1e6,
            'society': 1e9,
            'civilization': 1e12
        }
        return scale_factors.get(context.scale_level, 1e0) * context.time_scale

    def _generate_explanation_tree(self, structure_analysis: Dict, pressure_analysis: Dict,
                                  alignment_analysis: Dict, jump_analysis: Dict,
                                  integrated_result: Dict) -> Dict:
        """説明木構造生成"""
        return {
            'structure_factors': {
                'stability_contribution': structure_analysis['structure_stability'] * 0.3,
                'complexity_effect': structure_analysis['structure_complexity'] * 0.2,
                'adaptability_bonus': structure_analysis['structure_adaptability'] * 0.1
            },
            'pressure_factors': {
                'magnitude_impact': pressure_analysis['pressure_magnitude'] * 0.25,
                'coherence_effect': pressure_analysis['pressure_coherence'] * 0.15,
                'sustainability_factor': pressure_analysis['pressure_sustainability'] * 0.1
            },
            'alignment_factors': {
                'strength_contribution': alignment_analysis['alignment_strength'] * 0.3,
                'efficiency_bonus': alignment_analysis['alignment_efficiency'] * 0.2,
                'durability_effect': alignment_analysis['alignment_durability'] * 0.15
            },
            'jump_factors': {
                'probability_risk': jump_analysis['jump_probability'] * -0.2,
                'direction_uncertainty': len(jump_analysis['jump_direction_prediction']) * 0.05,
                'impact_consideration': jump_analysis['jump_impact_estimation'] * 0.1
            },
            'integration_logic': {
                'system_health_formula': '0.3*stability + 0.3*alignment_strength + 0.2*efficiency + 0.2*(1-jump_prob)',
                'evolution_potential_formula': '0.4*adaptability + 0.3*sustainability + 0.3*jump_prob',
                'resilience_formula': '0.4*stability + 0.3*durability + 0.3*coherence'
            }
        }

    def _generate_warnings(self, integrated_result: Dict[str, float]) -> List[str]:
        """警告生成"""
        warnings = []
        
        if integrated_result['system_health'] < 0.3:
            warnings.append("CRITICAL: システム健全性が危険レベルです")
        elif integrated_result['system_health'] < 0.5:
            warnings.append("WARNING: システム健全性が低下しています")
        
        if integrated_result['stability_resilience'] < 0.4:
            warnings.append("WARNING: 外乱に対する回復力が不足しています")
        
        if integrated_result['evolution_potential'] > 0.8:
            warnings.append("NOTICE: 高い変化ポテンシャルが検出されました")
        
        return warnings

    def _generate_recommendations(self, integrated_result: Dict[str, float],
                                 context: EvaluationContext) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        if integrated_result['system_health'] < 0.6:
            recommendations.append("構造の安定化または意味圧の軽減を検討してください")
        
        if integrated_result['evolution_potential'] < 0.3:
            recommendations.append("システムの変化・成長を促進する要素の導入を検討してください")
        
        if integrated_result['stability_resilience'] < 0.5:
            recommendations.append("冗長性やバックアップシステムの導入を推奨します")
        
        # ドメイン特化推奨
        if context.domain == 'AI':
            recommendations.append("学習率やハイパーパラメータの調整を検討してください")
        elif context.domain == 'biology':
            recommendations.append("環境適応機構の強化を検討してください")
        elif context.domain == 'sociology':
            recommendations.append("社会制度やコミュニケーション経路の見直しを検討してください")
        
        return recommendations

    def _create_error_result(self, evaluation_id: str, return_code: SSdReturnCode, 
                            error_message: str = "") -> UniversalEvaluationResult:
        """エラー結果生成"""
        return UniversalEvaluationResult(
            evaluation_id=evaluation_id,
            return_code=return_code,
            structure_stability=0.0,
            structure_complexity=0.0,
            structure_adaptability=0.0,
            pressure_magnitude=0.0,
            pressure_coherence=0.0,
            pressure_sustainability=0.0,
            alignment_strength=0.0,
            alignment_efficiency=0.0,
            alignment_durability=0.0,
            jump_probability=0.0,
            jump_direction_prediction=[0.0, 0.0, 0.0],
            jump_impact_estimation=0.0,
            system_health=0.0,
            evolution_potential=0.0,
            stability_resilience=0.0,
            calculation_confidence=0.0,
            computational_cost=0.0,
            prediction_horizon=0.0,
            explanation_tree={},
            warnings=[f"ERROR: {error_message}"] if error_message else [],
            recommendations=[]
        )

    def _update_performance_metrics(self, result: UniversalEvaluationResult):
        """パフォーマンス指標更新"""
        self.evaluation_count += 1
        self.performance_metrics['total_evaluations'] = self.evaluation_count
        
        # 平均計算時間更新
        prev_avg = self.performance_metrics['average_computation_time']
        new_avg = (prev_avg * (self.evaluation_count - 1) + result.computational_cost) / self.evaluation_count
        self.performance_metrics['average_computation_time'] = new_avg
        
        # キャッシュヒット率更新（簡易版）
        cache_hits = len(self.calculation_pool)
        self.performance_metrics['cache_hit_rate'] = cache_hits / self.evaluation_count

    # DLL互換関数（C互換インターフェース）
    def evaluate_system_c_interface(self, input_json: str) -> str:
        """C言語DLL互換インターフェース"""
        try:
            input_data = json.loads(input_json)
            
            # JSONから構造体への変換
            structures = [UniversalStructure(**s) for s in input_data.get('structures', [])]
            meaning_pressures = [UniversalMeaningPressure(**p) for p in input_data.get('meaning_pressures', [])]
            context = EvaluationContext(**input_data.get('context', {}))
            
            # 評価実行
            result = self.evaluate_universal_system(structures, meaning_pressures, context)
            
            # 結果をJSONで返す
            return json.dumps(asdict(result), indent=2)
            
        except Exception as e:
            error_result = self._create_error_result("error", SSdReturnCode.ERROR_CALCULATION_FAILED, str(e))
            return json.dumps(asdict(error_result), indent=2)

    # バッチ処理機能
    def evaluate_batch_systems(self, batch_inputs: List[Tuple[List[UniversalStructure], 
                                                           List[UniversalMeaningPressure], 
                                                           EvaluationContext]]) -> List[UniversalEvaluationResult]:
        """バッチ評価処理"""
        results = []
        
        for structures, pressures, context in batch_inputs:
            result = self.evaluate_universal_system(structures, pressures, context)
            results.append(result)
        
        return results

    # リアルタイムストリーミング評価
    def start_streaming_evaluation(self, callback_function):
        """ストリーミング評価開始"""
        self.streaming_callback = callback_function
        self.streaming_active = True

    def stream_evaluate(self, structures: List[UniversalStructure],
                       meaning_pressures: List[UniversalMeaningPressure],
                       context: EvaluationContext):
        """ストリーミング評価"""
        if hasattr(self, 'streaming_active') and self.streaming_active:
            result = self.evaluate_universal_system(structures, pressures, context)
            if hasattr(self, 'streaming_callback'):
                self.streaming_callback(result)

    # 統計・モニタリング機能
    def get_engine_statistics(self) -> Dict[str, Any]:
        """エンジン統計情報取得"""
        return {
            'engine_info': {
                'engine_id': self.engine_id,
                'version': self.version,
                'uptime_seconds': time.time() - (self.evaluation_count * self.performance_metrics.get('average_computation_time', 0))
            },
            'performance_metrics': self.performance_metrics.copy(),
            'cache_info': {
                'cache_size': len(self.calculation_pool),
                'max_cache_size': self.max_cache_size,
                'cache_enabled': self.cache_enabled
            },
            'configuration': self.config.copy()
        }

    def reset_engine(self):
        """エンジンリセット"""
        self.calculation_pool.clear()
        self.evaluation_count = 0
        self.performance_metrics = {
            'total_evaluations': 0,
            'average_computation_time': 0.0,
            'cache_hit_rate': 0.0,
            'accuracy_score': 0.0
        }


# DLL化対応ラッパー関数
def create_ssd_engine(config_json: str = "{}") -> UniversalSSDEngine:
    """SSDエンジン作成（DLL互換）"""
    config = json.loads(config_json) if config_json != "{}" else {}
    return UniversalSSDEngine(config)

def evaluate_system_dll(engine: UniversalSSDEngine, input_json: str) -> str:
    """システム評価（DLL互換）"""
    return engine.evaluate_system_c_interface(input_json)

def get_engine_stats_dll(engine: UniversalSSDEngine) -> str:
    """エンジン統計取得（DLL互換）"""
    return json.dumps(engine.get_engine_statistics(), indent=2)


# 使用例とテスト
def test_universal_ssd_engine():
    """汎用SSDエンジンのテスト"""
    print("=== SSD汎用評価エンジンテスト ===")
    
    # エンジン初期化
    config = {
        'precision_level': 'high',
        'calculation_mode': 'balanced',
        'enable_cache': True
    }
    engine = UniversalSSDEngine(config)
    
    # テストケース1: 生物学的システム
    print("\n--- テストケース1: 細胞分裂システム ---")
    
    structures = [
        UniversalStructure(
            structure_id="cell_membrane",
            structure_type="biological",
            dimension_count=3,
            stability_index=0.7,
            complexity_level=0.6,
            dynamic_properties={'permeability': 0.5, 'elasticity': 0.8},
            constraint_matrix=[[0.8, 0.2], [0.3, 0.9]],
            metadata={'cell_type': 'eukaryotic'}
        ),
        UniversalStructure(
            structure_id="dna_structure",
            structure_type="biological",
            dimension_count=4,
            stability_index=0.9,
            complexity_level=0.95,
            dynamic_properties={'replication_rate': 0.7},
            constraint_matrix=[[1.0, 0.0], [0.0, 1.0]],
            metadata={'chromosome_count': 46}
        )
    ]
    
    meaning_pressures = [
        UniversalMeaningPressure(
            pressure_id="growth_signal",
            source_type="external",
            magnitude=0.6,
            direction_vector=[1.0, 0.5, 0.0],
            frequency=0.1,
            duration=3600,
            propagation_speed=1.0,
            decay_function="exponential",
            interaction_matrix=[[0.8, 0.2], [0.4, 0.6]]
        ),
        UniversalMeaningPressure(
            pressure_id="nutrient_availability",
            source_type="environmental",
            magnitude=0.8,
            direction_vector=[0.8, 0.8, 0.2],
            frequency=0.05,
            duration=7200,
            propagation_speed=0.5,
            decay_function="linear",
            interaction_matrix=[[0.9, 0.1], [0.2, 0.8]]
        )
    ]
    
    context = EvaluationContext(
        context_id="cell_division_context",
        domain="biology",
        scale_level="cellular",
        time_scale=3600,
        space_scale=1e-6,
        observer_position=[0.0, 0.0, 0.0],
        measurement_precision=0.95,
        environmental_factors={'temperature': 0.7, 'ph_level': 0.6}
    )
    
    result = engine.evaluate_universal_system(structures, meaning_pressures, context)
    
    print(f"リターンコード: {result.return_code}")
    print(f"システム健全性: {result.system_health:.3f}")
    print(f"進化ポテンシャル: {result.evolution_potential:.3f}")
    print(f"安定性回復力: {result.stability_resilience:.3f}")
    print(f"跳躍確率: {result.jump_probability:.3f}")
    print(f"計算信頼度: {result.calculation_confidence:.3f}")
    print(f"計算コスト: {result.computational_cost:.4f}秒")
    print(f"警告: {result.warnings}")
    print(f"推奨: {result.recommendations}")
    
    # テストケース2: 社会システム
    print("\n--- テストケース2: 組織変革システム ---")
    
    social_structures = [
        UniversalStructure(
            structure_id="organizational_hierarchy",
            structure_type="social",
            dimension_count=2,
            stability_index=0.8,
            complexity_level=0.7,
            dynamic_properties={'bureaucracy_level': 0.6, 'flexibility': 0.4},
            constraint_matrix=[[0.9, 0.1], [0.2, 0.8]],
            metadata={'employee_count': 500}
        )
    ]
    
    social_pressures = [
        UniversalMeaningPressure(
            pressure_id="market_disruption",
            source_type="external",
            magnitude=0.9,
            direction_vector=[1.0, -0.5, 0.3],
            frequency=0.01,
            duration=86400,
            propagation_speed=2.0,
            decay_function="logarithmic",
            interaction_matrix=[[0.7, 0.3], [0.5, 0.5]]
        )
    ]
    
    social_context = EvaluationContext(
        context_id="organizational_change",
        domain="sociology",
        scale_level="group",
        time_scale=86400,
        space_scale=1000,
        observer_position=[0.0, 0.0, 0.0],
        measurement_precision=0.8,
        environmental_factors={'competition_level': 0.8, 'regulatory_pressure': 0.4}
    )
    
    social_result = engine.evaluate_universal_system(social_structures, social_pressures, social_context)
    
    print(f"組織システム健全性: {social_result.system_health:.3f}")
    print(f"変革ポテンシャル: {social_result.evolution_potential:.3f}")
    print(f"組織レジリエンス: {social_result.stability_resilience:.3f}")
    
    # DLL互換インターフェースのテスト
    print("\n--- DLL互換インターフェーステスト ---")
    
    dll_input = {
        'structures': [asdict(social_structures[0])],
        'meaning_pressures': [asdict(social_pressures[0])],
        'context': asdict(social_context)
    }
    
    dll_result_json = engine.evaluate_system_c_interface(json.dumps(dll_input))
    dll_result = json.loads(dll_result_json)
    
    print(f"DLLインターフェース結果: システム健全性 = {dll_result['system_health']:.3f}")
    
    # エンジン統計
    print("\n--- エンジン統計 ---")
    stats = engine.get_engine_statistics()
    print(f"総評価回数: {stats['performance_metrics']['total_evaluations']}")
    print(f"平均計算時間: {stats['performance_metrics']['average_computation_time']:.4f}秒")
    print(f"キャッシュサイズ: {stats['cache_info']['cache_size']}")


if __name__ == "__main__":
    test_universal_ssd_engine()