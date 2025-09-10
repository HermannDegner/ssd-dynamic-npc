"""
LLMを使ったSSD状態分類システム
既存の言語モデルでテキストからSSD状態を抽出し、学習データを自動生成
"""

import json
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import asyncio
from datetime import datetime

# 実際の実装では以下のいずれかを使用
# from openai import OpenAI  # OpenAI API
# from anthropic import Anthropic  # Claude API
# import ollama  # ローカルLLM

@dataclass
class SSDStateFromText:
    """テキストから抽出されたSSD状態"""
    # 物理層
    weather_condition: str = "unknown"
    threat_level: float = 0.0
    time_of_day: str = "unknown"
    location_type: str = "unknown"
    
    # 基層（0-1の値）
    comfort_level: float = 0.5
    social_need: float = 0.5
    exploration_desire: float = 0.5
    creation_urge: float = 0.5
    recognition_need: float = 0.5
    
    # 慣性・記憶
    mentioned_habits: List[str] = None
    emotional_memories: List[str] = None
    
    # メタ情報
    confidence_score: float = 0.0
    personality_indicators: List[str] = None
    
    def __post_init__(self):
        if self.mentioned_habits is None:
            self.mentioned_habits = []
        if self.emotional_memories is None:
            self.emotional_memories = []
        if self.personality_indicators is None:
            self.personality_indicators = []

class LLMSSDClassifier:
    """LLMを使ったSSD状態分類器"""
    
    def __init__(self, model_type: str = "mock"):
        self.model_type = model_type
        self.classification_prompt = self._create_classification_prompt()
        
        # 実際の実装では適切なクライアントを初期化
        # self.client = OpenAI() or Anthropic() or ollama
        
    def _create_classification_prompt(self) -> str:
        """SSD状態分類用のプロンプト作成"""
        return """
あなたは、テキストからキャラクターの内部状態を分析する専門家です。
以下のテキストを読んで、キャラクターのSSD（構造主観力学）状態を分析してください。

## 分析項目

### 物理層状態
- weather_condition: 天候状況 (sunny/cloudy/rainy/stormy/unknown)
- threat_level: 脅威レベル (0.0-1.0)
- time_of_day: 時間帯 (morning/day/evening/night/unknown)
- location_type: 場所タイプ (home/village/nature/shop/unknown)

### 基層状態（各項目0.0-1.0で評価）
- comfort_level: 安心・快適さのレベル
- social_need: 社交への欲求
- exploration_desire: 探索・新体験への欲求
- creation_urge: 創造・表現への衝動
- recognition_need: 承認・評価への欲求

### 慣性・記憶
- mentioned_habits: 言及された習慣・行動パターン
- emotional_memories: 感情的な記憶・過去の経験
- personality_indicators: 性格を示す特徴

### メタ情報
- confidence_score: 分析の確信度 (0.0-1.0)

## 出力形式
JSON形式で出力してください：

```json
{
    "weather_condition": "sunny",
    "threat_level": 0.0,
    "time_of_day": "morning",
    "location_type": "village",
    "comfort_level": 0.8,
    "social_need": 0.6,
    "exploration_desire": 0.3,
    "creation_urge": 0.2,
    "recognition_need": 0.4,
    "mentioned_habits": ["morning_walk", "coffee_routine"],
    "emotional_memories": ["happy_childhood", "recent_success"],
    "personality_indicators": ["optimistic", "friendly"],
    "confidence_score": 0.9
}
```

分析対象テキスト:
"""

    async def classify_text(self, text: str) -> SSDStateFromText:
        """テキストからSSD状態を分類"""
        
        # 実際の実装例（モック）
        if self.model_type == "mock":
            return self._mock_classification(text)
        
        # 実際のLLM呼び出し例
        # response = await self._call_llm(text)
        # return self._parse_llm_response(response)
    
    def _mock_classification(self, text: str) -> SSDStateFromText:
        """モック分類（デモ用）"""
        
        # 簡単なキーワードベース分析（実際はLLMが担当）
        text_lower = text.lower()
        
        # 天候検出
        weather_keywords = {
            "sunny": ["晴れ", "太陽", "明るい"],
            "rainy": ["雨", "濡れる", "傘"],
            "stormy": ["嵐", "雷", "強風"],
            "cloudy": ["曇り", "雲"]
        }
        
        weather = "unknown"
        for condition, keywords in weather_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                weather = condition
                break
        
        # 感情・欲求の推定
        comfort_indicators = ["安心", "快適", "リラックス", "平和"]
        social_indicators = ["話す", "友達", "一緒", "会話", "みんな"]
        exploration_indicators = ["探索", "冒険", "新しい", "発見", "試す"]
        creation_indicators = ["作る", "描く", "書く", "創作", "表現"]
        recognition_indicators = ["褒められ", "認められ", "評価", "すごい"]
        
        comfort_level = min(1.0, sum(0.2 for word in comfort_indicators if word in text_lower))
        social_need = min(1.0, sum(0.2 for word in social_indicators if word in text_lower))
        exploration_desire = min(1.0, sum(0.2 for word in exploration_indicators if word in text_lower))
        creation_urge = min(1.0, sum(0.2 for word in creation_indicators if word in text_lower))
        recognition_need = min(1.0, sum(0.2 for word in recognition_indicators if word in text_lower))
        
        # 脅威レベル
        threat_keywords = ["危険", "怖い", "不安", "心配", "嵐"]
        threat_level = min(1.0, sum(0.3 for word in threat_keywords if word in text_lower))
        
        # 時間帯検出
        time_keywords = {
            "morning": ["朝", "午前"],
            "day": ["昼", "午後", "日中"],
            "evening": ["夕方", "夕暮れ"],
            "night": ["夜", "夜中"]
        }
        
        time_of_day = "unknown"
        for period, keywords in time_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                time_of_day = period
                break
        
        return SSDStateFromText(
            weather_condition=weather,
            threat_level=threat_level,
            time_of_day=time_of_day,
            location_type="village",  # デフォルト
            comfort_level=comfort_level,
            social_need=social_need,
            exploration_desire=exploration_desire,
            creation_urge=creation_urge,
            recognition_need=recognition_need,
            mentioned_habits=[],
            emotional_memories=[],
            personality_indicators=[],
            confidence_score=0.7  # モックなので低め
        )
    
    async def _call_llm(self, text: str) -> str:
        """実際のLLM呼び出し（実装例）"""
        
        # OpenAI GPT-4の場合
        """
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.classification_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
        """
        
        # Claude 3.5 Sonnetの場合
        """
        response = await self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": f"{self.classification_prompt}\n\n{text}"}
            ]
        )
        return response.content[0].text
        """
        
        # Ollamaローカルの場合
        """
        response = ollama.chat(
            model='llama3.1',
            messages=[
                {'role': 'system', 'content': self.classification_prompt},
                {'role': 'user', 'content': text}
            ]
        )
        return response['message']['content']
        """
        
        return ""  # モック用
    
    def _parse_llm_response(self, response: str) -> SSDStateFromText:
        """LLM応答をパース"""
        try:
            # JSONレスポンスをパース
            response_clean = response.strip()
            if "```json" in response_clean:
                start = response_clean.find("```json") + 7
                end = response_clean.find("```", start)
                response_clean = response_clean[start:end].strip()
            
            data = json.loads(response_clean)
            return SSDStateFromText(**data)
            
        except Exception as e:
            print(f"パースエラー: {e}")
            # デフォルト値を返す
            return SSDStateFromText()

class SSDTrainingDataGenerator:
    """SSD学習データ自動生成器"""
    
    def __init__(self):
        self.classifier = LLMSSDClassifier()
        
        # どうぶつの森風サンプルテキスト
        self.sample_texts = [
            "今日は朝から雨が降っていて、ちょっと憂鬱な気分です。家でゆっくり読書でもしようかな。",
            "お天気が良くて、みんなでピクニックに行きました！とても楽しかったです。",
            "新しい場所を探検してみたくなりました。何か面白い発見があるかもしれません。",
            "今日は創作活動に集中したい気分です。新しい作品を作ってみようと思います。",
            "友達に褒められて、とても嬉しい気持ちになりました。頑張った甲斐がありました。",
            "夜になって少し不安になってきました。早く家に帰ろうと思います。",
            "毎朝のコーヒーが日課になっています。この時間がとても落ち着きます。",
            "村のお祭りでみんなと踊りました。こういう時間がとても大切だと感じます。"
        ]
    
    async def generate_training_data(self, num_samples: int = 100) -> List[Dict]:
        """学習データの自動生成"""
        
        training_data = []
        
        print(f"SSD学習データを{num_samples}サンプル生成中...")
        
        # 既存サンプルテキストの処理
        for text in self.sample_texts:
            ssd_state = await self.classifier.classify_text(text)
            
            training_data.append({
                "input_text": text,
                "ssd_state": asdict(ssd_state),
                "timestamp": datetime.now().isoformat(),
                "source": "sample"
            })
        
        # 追加のバリエーション生成
        additional_samples = num_samples - len(self.sample_texts)
        if additional_samples > 0:
            for i in range(additional_samples):
                # ランダムなバリエーションを生成
                variation_text = self._generate_text_variation()
                ssd_state = await self.classifier.classify_text(variation_text)
                
                training_data.append({
                    "input_text": variation_text,
                    "ssd_state": asdict(ssd_state),
                    "timestamp": datetime.now().isoformat(),
                    "source": "generated"
                })
        
        print(f"データ生成完了: {len(training_data)}サンプル")
        return training_data
    
    def _generate_text_variation(self) -> str:
        """テキストのバリエーション生成"""
        
        # 簡単なテンプレートベース生成（実際はLLMで行う）
        weather_options = ["晴れて", "雨が降って", "曇りで", "嵐で"]
        time_options = ["朝", "昼", "夕方", "夜"]
        activity_options = ["散歩", "読書", "料理", "掃除", "絵を描く", "友達と話す"]
        emotion_options = ["楽しい", "悲しい", "興奮した", "落ち着いた", "不安な"]
        
        weather = random.choice(weather_options)
        time = random.choice(time_options)
        activity = random.choice(activity_options)
        emotion = random.choice(emotion_options)
        
        templates = [
            f"{time}に{weather}いるので、{activity}をしたい気分です。{emotion}気持ちになります。",
            f"今日は{weather}います。{activity}をして{emotion}時間を過ごしました。",
            f"{time}の時間帯に{activity}をするのが好きです。{weather}いる日は特に{emotion}気分になります。"
        ]
        
        return random.choice(templates)
    
    def save_training_data(self, data: List[Dict], filename: str = "ssd_training_data.json"):
        """学習データをファイルに保存"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"学習データを{filename}に保存しました")
    
    def load_training_data(self, filename: str = "ssd_training_data.json") -> List[Dict]:
        """学習データをファイルから読み込み"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"学習データを{filename}から読み込みました: {len(data)}サンプル")
            return data
        except FileNotFoundError:
            print(f"ファイル{filename}が見つかりません")
            return []

class SSDDataAnalyzer:
    """SSD学習データの分析器"""
    
    def analyze_data_distribution(self, training_data: List[Dict]) -> Dict:
        """データ分布の分析"""
        
        if not training_data:
            return {}
        
        # 基層状態の分布
        comfort_levels = [item["ssd_state"]["comfort_level"] for item in training_data]
        social_needs = [item["ssd_state"]["social_need"] for item in training_data]
        exploration_desires = [item["ssd_state"]["exploration_desire"] for item in training_data]
        creation_urges = [item["ssd_state"]["creation_urge"] for item in training_data]
        recognition_needs = [item["ssd_state"]["recognition_need"] for item in training_data]
        
        # 天候分布
        weather_dist = {}
        for item in training_data:
            weather = item["ssd_state"]["weather_condition"]
            weather_dist[weather] = weather_dist.get(weather, 0) + 1
        
        # 時間帯分布
        time_dist = {}
        for item in training_data:
            time_period = item["ssd_state"]["time_of_day"]
            time_dist[time_period] = time_dist.get(time_period, 0) + 1
        
        analysis = {
            "total_samples": len(training_data),
            "basal_states": {
                "comfort_avg": sum(comfort_levels) / len(comfort_levels),
                "social_avg": sum(social_needs) / len(social_needs),
                "exploration_avg": sum(exploration_desires) / len(exploration_desires),
                "creation_avg": sum(creation_urges) / len(creation_urges),
                "recognition_avg": sum(recognition_needs) / len(recognition_needs)
            },
            "weather_distribution": weather_dist,
            "time_distribution": time_dist,
            "confidence_scores": [item["ssd_state"]["confidence_score"] for item in training_data]
        }
        
        return analysis
    
    def print_analysis(self, analysis: Dict):
        """分析結果の表示"""
        print("\n=== SSD学習データ分析結果 ===")
        print(f"総サンプル数: {analysis['total_samples']}")
        
        print("\n【基層状態平均値】")
        basal = analysis['basal_states']
        print(f"  安心レベル: {basal['comfort_avg']:.3f}")
        print(f"  社交欲求: {basal['social_avg']:.3f}")
        print(f"  探索欲求: {basal['exploration_avg']:.3f}")
        print(f"  創造衝動: {basal['creation_avg']:.3f}")
        print(f"  承認欲求: {basal['recognition_avg']:.3f}")
        
        print("\n【天候分布】")
        for weather, count in analysis['weather_distribution'].items():
            percentage = (count / analysis['total_samples']) * 100
            print(f"  {weather}: {count}件 ({percentage:.1f}%)")
        
        print("\n【時間帯分布】")
        for time_period, count in analysis['time_distribution'].items():
            percentage = (count / analysis['total_samples']) * 100
            print(f"  {time_period}: {count}件 ({percentage:.1f}%)")
        
        avg_confidence = sum(analysis['confidence_scores']) / len(analysis['confidence_scores'])
        print(f"\n平均確信度: {avg_confidence:.3f}")

async def main():
    """メイン実行関数"""
    print("🤖 LLMベースSSD状態分類システム")
    print("=" * 50)
    
    # データ生成器の初期化
    generator = SSDTrainingDataGenerator()
    analyzer = SSDDataAnalyzer()
    
    # 学習データの生成
    training_data = await generator.generate_training_data(num_samples=20)
    
    # データ分析
    analysis = analyzer.analyze_data_distribution(training_data)
    analyzer.print_analysis(analysis)
    
    # データ保存
    generator.save_training_data(training_data)
    
    # 個別サンプルの表示
    print("\n=== サンプル例 ===")
    for i, sample in enumerate(training_data[:3]):
        print(f"\n【サンプル{i+1}】")
        print(f"入力: {sample['input_text']}")
        print(f"天候: {sample['ssd_state']['weather_condition']}")
        print(f"脅威: {sample['ssd_state']['threat_level']:.2f}")
        print(f"安心: {sample['ssd_state']['comfort_level']:.2f}")
        print(f"社交: {sample['ssd_state']['social_need']:.2f}")
        print(f"確信度: {sample['ssd_state']['confidence_score']:.2f}")
    
    print("\n✨ 学習データ生成完了！")
    print("このデータを使って本格的なSSDレンズ学習が可能になります。")

if __name__ == "__main__":
    asyncio.run(main())
