"""
SSDレンズ学習システム - どうぶつの森NPC用
構造主観力学理論に基づく言語モデル学習

意味圧 → 構造変化 → 言語表現 の学習パラダイム
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from transformers import AutoTokenizer, AutoModel

@dataclass
class SSDState:
    """SSD状態の表現"""
    # 物理層状態
    weather: str = "sunny"
    temperature: float = 20.0
    time_period: str = "day"
    threat_level: float = 0.0
    
    # 基層状態
    comfort_level: float = 0.5
    social_fulfillment: float = 0.5
    exploration_need: float = 0.3
    creation_urge: float = 0.2
    recognition_level: float = 0.4
    
    # 慣性状態
    habit_strength: Dict[str, float] = None
    memory_activation: Dict[str, float] = None
    
    # コンテキスト
    location: str = "village"
    player_present: bool = False
    other_npcs_present: List[str] = None
    
    def __post_init__(self):
        if self.habit_strength is None:
            self.habit_strength = {}
        if self.memory_activation is None:
            self.memory_activation = {}
        if self.other_npcs_present is None:
            self.other_npcs_present = []

@dataclass
class SSDResponse:
    """SSD応答の表現"""
    speech: str
    action: str
    emotion: str
    internal_state_change: Dict[str, float]
    memory_update: str

class SSDStateEncoder:
    """SSD状態をベクトルに変換"""
    
    def __init__(self):
        # カテゴリカル変数のマッピング
        self.weather_map = {"sunny": 0, "cloudy": 1, "rainy": 2, "stormy": 3}
        self.time_map = {"morning": 0, "day": 1, "evening": 2, "night": 3}
        self.location_map = {"village": 0, "home": 1, "beach": 2, "forest": 3, "shop": 4}
        self.action_map = {
            "greet": 0, "chat": 1, "explore": 2, "rest": 3, "work": 4,
            "exercise": 5, "create": 6, "seek_shelter": 7, "celebrate": 8
        }
        self.emotion_map = {
            "happy": 0, "sad": 1, "angry": 2, "calm": 3, "excited": 4,
            "anxious": 5, "friendly": 6, "lonely": 7, "confident": 8
        }
        
        self.vector_dim = 32  # 状態ベクトルの次元数
    
    def encode(self, state: SSDState) -> torch.Tensor:
        """SSD状態をベクトルに変換"""
        vector = []
        
        # 物理層 (6次元)
        vector.append(self.weather_map.get(state.weather, 0) / 3.0)  # 正規化
        vector.append(state.temperature / 40.0)  # -20~40度を0-1に
        vector.append(self.time_map.get(state.time_period, 1) / 3.0)
        vector.append(state.threat_level)
        vector.append(1.0 if state.player_present else 0.0)
        vector.append(len(state.other_npcs_present) / 5.0)  # 最大5人想定
        
        # 基層状態 (5次元)
        vector.extend([
            state.comfort_level,
            state.social_fulfillment,
            state.exploration_need,
            state.creation_urge,
            state.recognition_level
        ])
        
        # 慣性状態 (10次元) - 主要な習慣パターン
        common_habits = ["morning_routine", "social_chat", "creative_work", 
                        "exploration", "rest", "exercise", "cooking", "cleaning",
                        "shopping", "entertainment"]
        for habit in common_habits:
            vector.append(state.habit_strength.get(habit, 0.0))
        
        # 記憶活性化 (10次元) - 主要な記憶カテゴリ
        memory_categories = ["friendship", "achievement", "failure", "discovery",
                           "celebration", "conflict", "learning", "creation",
                           "seasonal", "special_event"]
        for category in memory_categories:
            vector.append(state.memory_activation.get(category, 0.0))
        
        # 位置情報 (1次元)
        vector.append(self.location_map.get(state.location, 0) / 4.0)
        
        return torch.tensor(vector, dtype=torch.float32)
    
    def decode_action(self, action_idx: int) -> str:
        """行動インデックスを文字列に変換"""
        action_list = list(self.action_map.keys())
        if 0 <= action_idx < len(action_list):
            return action_list[action_idx]
        return "unknown"
    
    def decode_emotion(self, emotion_idx: int) -> str:
        """感情インデックスを文字列に変換"""
        emotion_list = list(self.emotion_map.keys())
        if 0 <= emotion_idx < len(emotion_list):
            return emotion_list[emotion_idx]
        return "neutral"

class MeaningPressureCalculator(nn.Module):
    """意味圧計算層"""
    
    def __init__(self, state_dim: int = 32):
        super().__init__()
        self.state_dim = state_dim
        
        # 各層の意味圧計算
        self.physical_pressure = nn.Linear(6, 16)
        self.basal_pressure = nn.Linear(5, 16)  
        self.inertia_pressure = nn.Linear(21, 16)  # 習慣10 + 記憶10 + 位置1
        
        # 意味圧統合
        self.pressure_integration = nn.Linear(48, 32)  # 16*3 = 48
        self.pressure_norm = nn.LayerNorm(32)
        
        self.activation = nn.GELU()
    
    def forward(self, state_vector: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """意味圧を計算"""
        # 状態ベクトルを各層に分割
        physical = state_vector[:, :6]      # 物理層
        basal = state_vector[:, 6:11]       # 基層
        inertia = state_vector[:, 11:]      # 慣性層
        
        # 各層の意味圧計算
        phys_pressure = self.activation(self.physical_pressure(physical))
        basal_pressure = self.activation(self.basal_pressure(basal))
        inertia_pressure = self.activation(self.inertia_pressure(inertia))
        
        # 意味圧統合
        combined_pressure = torch.cat([phys_pressure, basal_pressure, inertia_pressure], dim=-1)
        integrated_pressure = self.pressure_norm(self.pressure_integration(combined_pressure))
        
        pressure_components = {
            "physical": phys_pressure,
            "basal": basal_pressure, 
            "inertia": inertia_pressure,
            "integrated": integrated_pressure
        }
        
        return integrated_pressure, pressure_components

class StructuralChangePredictor(nn.Module):
    """構造変化予測層"""
    
    def __init__(self, pressure_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.pressure_dim = pressure_dim
        self.hidden_dim = hidden_dim
        
        # 構造変化予測ネットワーク
        self.change_predictor = nn.Sequential(
            nn.Linear(pressure_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, pressure_dim)
        )
        
        # 行動選択予測
        self.action_predictor = nn.Linear(pressure_dim, 9)  # 9種類の行動
        
        # 感情状態予測
        self.emotion_predictor = nn.Linear(pressure_dim, 9)  # 9種類の感情
        
        # 状態変化の大きさ予測
        self.change_magnitude = nn.Linear(pressure_dim, 1)
        
    def forward(self, meaning_pressure: torch.Tensor) -> Dict[str, torch.Tensor]:
        """構造変化を予測"""
        # 次の状態への変化を予測
        state_change = self.change_predictor(meaning_pressure)
        
        # 行動・感情予測
        action_logits = self.action_predictor(meaning_pressure)
        emotion_logits = self.emotion_predictor(meaning_pressure)
        
        # 変化の大きさ
        change_mag = torch.sigmoid(self.change_magnitude(meaning_pressure))
        
        return {
            "state_change": state_change,
            "action_logits": action_logits,
            "emotion_logits": emotion_logits,
            "change_magnitude": change_mag
        }

class SSDLanguageHead(nn.Module):
    """SSD言語生成ヘッド"""
    
    def __init__(self, ssd_dim: int = 32, vocab_size: int = 32000, max_length: int = 64):
        super().__init__()
        self.ssd_dim = ssd_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # SSD状態を言語空間にマッピング
        self.ssd_to_lang = nn.Linear(ssd_dim, 512)
        
        # 言語生成用のTransformerデコーダー
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # 出力層
        self.output_projection = nn.Linear(512, vocab_size)
        
        # 位置埋め込み
        self.pos_embedding = nn.Embedding(max_length, 512)
        
    def forward(self, ssd_features: torch.Tensor, target_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """SSD特徴から言語を生成"""
        batch_size = ssd_features.size(0)
        
        # SSD特徴を言語空間にマッピング
        lang_context = self.ssd_to_lang(ssd_features).unsqueeze(1)  # [batch, 1, 512]
        
        if target_tokens is not None:
            # 訓練時: 教師ありシーケンス生成
            seq_len = target_tokens.size(1)
            pos_ids = torch.arange(seq_len, device=target_tokens.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embedding(pos_ids)
            
            # トークン埋め込み（簡易版）
            token_emb = torch.randn(batch_size, seq_len, 512, device=target_tokens.device)
            token_emb = token_emb + pos_emb
            
            # デコーダー処理
            output = self.transformer_decoder(token_emb, lang_context)
            logits = self.output_projection(output)
            
            return logits
        else:
            # 推論時: 自回帰生成（簡易版）
            generated = torch.zeros(batch_size, 1, 512, device=ssd_features.device)
            output = self.transformer_decoder(generated, lang_context)
            logits = self.output_projection(output)
            
            return logits

class SSDLensModel(nn.Module):
    """SSDレンズ学習の統合モデル"""
    
    def __init__(self, vocab_size: int = 32000):
        super().__init__()
        
        # SSD状態エンコーダー
        self.state_encoder = SSDStateEncoder()
        
        # 意味圧計算
        self.meaning_pressure = MeaningPressureCalculator()
        
        # 構造変化予測
        self.structure_predictor = StructuralChangePredictor()
        
        # 言語生成ヘッド
        self.language_head = SSDLanguageHead(vocab_size=vocab_size)
        
        # 損失関数の重み
        self.loss_weights = {
            "language": 1.0,
            "action": 0.5,
            "emotion": 0.5,
            "change": 0.3
        }
    
    def forward(self, ssd_state_batch: List[SSDState], 
                target_speech: Optional[torch.Tensor] = None,
                target_actions: Optional[torch.Tensor] = None,
                target_emotions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """順伝播"""
        
        # SSD状態をベクトル化
        state_vectors = torch.stack([self.state_encoder.encode(state) for state in ssd_state_batch])
        
        # 意味圧計算
        meaning_pressure, pressure_components = self.meaning_pressure(state_vectors)
        
        # 構造変化予測
        structural_changes = self.structure_predictor(meaning_pressure)
        
        # 言語生成
        if target_speech is not None:
            language_logits = self.language_head(meaning_pressure, target_speech)
        else:
            language_logits = self.language_head(meaning_pressure)
        
        outputs = {
            "language_logits": language_logits,
            "action_logits": structural_changes["action_logits"],
            "emotion_logits": structural_changes["emotion_logits"],
            "state_change": structural_changes["state_change"],
            "change_magnitude": structural_changes["change_magnitude"],
            "meaning_pressure": meaning_pressure,
            "pressure_components": pressure_components
        }
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """SSDレンズ学習の損失計算"""
        
        losses = {}
        
        # 言語生成損失
        if "speech" in targets and outputs["language_logits"] is not None:
            language_loss = nn.CrossEntropyLoss()(
                outputs["language_logits"].reshape(-1, outputs["language_logits"].size(-1)),
                targets["speech"].reshape(-1)
            )
            losses["language"] = language_loss * self.loss_weights["language"]
        
        # 行動予測損失
        if "action" in targets:
            action_loss = nn.CrossEntropyLoss()(outputs["action_logits"], targets["action"])
            losses["action"] = action_loss * self.loss_weights["action"]
        
        # 感情予測損失
        if "emotion" in targets:
            emotion_loss = nn.CrossEntropyLoss()(outputs["emotion_logits"], targets["emotion"])
            losses["emotion"] = emotion_loss * self.loss_weights["emotion"]
        
        # 構造変化損失（次状態予測）
        if "next_state" in targets:
            change_loss = nn.MSELoss()(outputs["state_change"], targets["next_state"])
            losses["change"] = change_loss * self.loss_weights["change"]
        
        # 総損失
        total_loss = sum(losses.values())
        losses["total"] = total_loss
        
        return losses

class AnimalCrossingDataset(Dataset):
    """どうぶつの森風の学習データセット"""
    
    def __init__(self, data_size: int = 1000):
        self.data_size = data_size
        self.state_encoder = SSDStateEncoder()
        self.data = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> List[Dict]:
        """合成学習データの生成"""
        data = []
        
        personalities = ["peppy", "lazy", "cranky", "normal", "jock", "snooty"]
        
        for i in range(self.data_size):
            personality = random.choice(personalities)
            
            # ランダムなSSD状態生成
            state = self._generate_random_state(personality)
            
            # 対応する応答生成
            response = self._generate_response_for_state(state, personality)
            
            data.append({
                "state": state,
                "response": response,
                "personality": personality
            })
        
        return data
    
    def _generate_random_state(self, personality: str) -> SSDState:
        """性格に基づくランダム状態生成"""
        
        # 性格別の傾向
        personality_traits = {
            "peppy": {"social_fulfillment": 0.8, "exploration_need": 0.7},
            "lazy": {"comfort_level": 0.9, "creation_urge": 0.3},
            "cranky": {"comfort_level": 0.6, "recognition_level": 0.7},
            "normal": {"social_fulfillment": 0.6, "comfort_level": 0.6},
            "jock": {"exploration_need": 0.9, "recognition_level": 0.8},
            "snooty": {"recognition_level": 0.9, "creation_urge": 0.8}
        }
        
        base_traits = personality_traits.get(personality, {})
        
        state = SSDState(
            weather=random.choice(["sunny", "cloudy", "rainy", "stormy"]),
            temperature=random.uniform(-5, 35),
            time_period=random.choice(["morning", "day", "evening", "night"]),
            threat_level=random.uniform(0, 1) if random.random() < 0.2 else 0,
            
            comfort_level=base_traits.get("comfort_level", random.uniform(0.3, 0.8)),
            social_fulfillment=base_traits.get("social_fulfillment", random.uniform(0.2, 0.7)),
            exploration_need=base_traits.get("exploration_need", random.uniform(0.1, 0.6)),
            creation_urge=base_traits.get("creation_urge", random.uniform(0.1, 0.5)),
            recognition_level=base_traits.get("recognition_level", random.uniform(0.2, 0.6)),
            
            location=random.choice(["village", "home", "beach", "forest", "shop"]),
            player_present=random.choice([True, False]),
            other_npcs_present=random.choices(["alice", "bob", "charlie"], k=random.randint(0, 3))
        )
        
        return state
    
    def _generate_response_for_state(self, state: SSDState, personality: str) -> SSDResponse:
        """状態に基づく応答生成（ルールベース）"""
        
        # 簡易的な応答生成ロジック
        if state.threat_level > 0.5:
            speech = "うわあ！危険そう！避難しましょう！"
            action = "seek_shelter"
            emotion = "anxious"
        elif state.player_present and state.social_fulfillment < 0.4:
            speech = "こんにちは！お話ししませんか？"
            action = "greet"
            emotion = "friendly"
        elif state.exploration_need > 0.7:
            speech = "何か新しいことがしたいな！"
            action = "explore"
            emotion = "excited"
        elif state.comfort_level < 0.3:
            speech = "ちょっと疲れちゃった..."
            action = "rest"
            emotion = "calm"
        else:
            speech = "今日もいい天気ですね！"
            action = "chat"
            emotion = "happy"
        
        return SSDResponse(
            speech=speech,
            action=action,
            emotion=emotion,
            internal_state_change={"comfort": 0.1},
            memory_update="daily_interaction"
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # ダミーのトークン化（実際にはtokenizerを使用）
        speech_tokens = torch.randint(0, 1000, (20,))  # 20トークンの固定長
        
        return {
            "state": item["state"],
            "speech_tokens": speech_tokens,
            "action": self.state_encoder.action_map.get(item["response"].action, 0),
            "emotion": self.state_encoder.emotion_map.get(item["response"].emotion, 0),
            "personality": item["personality"]
        }

def train_ssd_lens_model():
    """SSDレンズモデルの訓練"""
    
    print("SSDレンズ学習システムを初期化中...")
    
    # モデル初期化
    model = SSDLensModel(vocab_size=1000)  # 簡易的な語彙サイズ
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # データセット・データローダー
    dataset = AnimalCrossingDataset(data_size=1000)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"データセットサイズ: {len(dataset)}")
    print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 訓練ループ
    model.train()
    for epoch in range(10):  # 簡易的に10エポック
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # 前向き計算
            outputs = model(
                ssd_state_batch=batch["state"],
                target_speech=batch["speech_tokens"]
            )
            
            # 損失計算
            targets = {
                "speech": batch["speech_tokens"],
                "action": batch["action"],
                "emotion": batch["emotion"]
            }
            
            losses = model.compute_loss(outputs, targets)
            
            # 後向き計算
            losses["total"].backward()
            optimizer.step()
            
            total_loss += losses["total"].item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"エポック {epoch+1}/10, 平均損失: {avg_loss:.4f}")
    
    print("訓練完了！")
    return model

def demo_ssd_lens_inference(model: SSDLensModel):
    """SSDレンズ推論のデモ"""
    
    print("\n=== SSDレンズ推論デモ ===")
    
    # テストケース1: 嵐の中のNPC
    test_state_1 = SSDState(
        weather="stormy",
        threat_level=0.8,
        comfort_level=0.2,
        player_present=True,
        location="village"
    )
    
    # テストケース2: 社交的なNPC
    test_state_2 = SSDState(
        weather="sunny",
        social_fulfillment=0.3,
        exploration_need=0.7,
        player_present=True,
        location="beach"
    )
    
    test_cases = [
        ("嵐の状況", test_state_1),
        ("社交状況", test_state_2)
    ]
    
    model.eval()
    with torch.no_grad():
        for name, state in test_cases:
            print(f"\n【{name}】")
            print(f"入力状態: 天候={state.weather}, 脅威={state.threat_level}, 快適={state.comfort_level}")
            
            outputs = model([state])
            
            # 行動・感情予測
            action_pred = torch.argmax(outputs["action_logits"], dim=-1)
            emotion_pred = torch.argmax(outputs["emotion_logits"], dim=-1)
            
            action_str = model.state_encoder.decode_action(action_pred.item())
            emotion_str = model.state_encoder.decode_emotion(emotion_pred.item())
            
            print(f"予測行動: {action_str}")
            print(f"予測感情: {emotion_str}")
            print(f"意味圧の強さ: {outputs['meaning_pressure'].norm().item():.3f}")

if __name__ == "__main__":
    print("🌟 SSDレンズ学習システム - どうぶつの森NPC用 🌟")
    print("構造主観力学に基づく革新的言語モデル学習")
    print("-" * 50)
    
    # モデル訓練
    trained_model = train_ssd_lens_model()
    
    # 推論デモ
    demo_ssd_lens_inference(trained_model)
    
    print("\n✨ SSDレンズ学習プロトタイプ完成！")
    print("このシステムは意味圧 → 構造変化 → 言語表現の")
    print("新しい学習パラダイムを実装しています。")
