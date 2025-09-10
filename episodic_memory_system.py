import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# --- データ構造定義 ---

@dataclass
class EpisodicMemory:
    """
    個々のエピソード記憶を表すデータ構造。
    SSDにおける「構造に刻まれた変化の痕跡」に相当します。
    """
    timestamp: float
    event_type: str  # 例: "GIFT_RECEIVED", "ARGUMENT", "SHARED_HOBBY"
    location: str    # 例: "plaza", "beach", "shop"
    involved_entities: List[str] # 関わったNPCやプレイヤーのID
    valence: float   # 感情価: ポジティブ(+)かネガティブ(-)か。-1.0 ~ 1.0
    salience: float = 1.0 # 記憶の顕著性・鮮明さ。時間と共に減衰する。

# --- 本体 ---

class EpisodicMemorySystem:
    """
    エピソード記憶の記録、想起、そして「エピソード慣性」による行動提案を管理するシステム。
    """

    def __init__(self, owner_id: str, memory_decay_rate: float = 0.98):
        """
        Args:
            owner_id (str): この記憶システムの所有者であるNPCのID。
            memory_decay_rate (float): 1回の更新で記憶の顕著性がどれだけ減衰するか。
        """
        self.owner_id = owner_id
        self.memories: List[EpisodicMemory] = []
        self.memory_decay_rate = memory_decay_rate

    def record_event(self, event_type: str, location: str, involved_entities: List[str], valence: float):
        """
        新しい出来事をエピソード記憶として記録（構造に痕跡を刻む）。
        """
        # 自分自身のIDは関与エンティティから除外する
        relevant_entities = [entity for entity in involved_entities if entity != self.owner_id]
        if not relevant_entities:
            return # 自分自身だけのイベントは記録しない（今回は）

        new_memory = EpisodicMemory(
            timestamp=time.time(),
            event_type=event_type,
            location=location,
            involved_entities=relevant_entities,
            valence=valence
        )
        self.memories.append(new_memory)
        print(f"[{self.owner_id}] New Memory Recorded: {new_memory}")

    def retrieve_memories_about_entity(self, entity_id: str, max_memories: int = 5) -> List[EpisodicMemory]:
        """
        特定のエンティティに関する記憶を、顕著性が高い順に想起する。
        """
        related_memories = [
            mem for mem in self.memories if entity_id in mem.involved_entities
        ]
        # 顕著性が高い順にソートして返す
        return sorted(related_memories, key=lambda m: m.salience, reverse=True)[:max_memories]

    def generate_action_proposals(self, perceived_entities: List[str]) -> Dict[str, float]:
        """
        知覚したエンティティに基づき、「エピソード慣性」による行動提案を生成する。
        これが、既存の行動選択システムへの「意味圧」となる。

        Returns:
            Dict[str, float]: 行動提案とそのスコアの辞書。
                               例: {"APPROACH_PlayerA": 0.8, "AVOID_VillagerB": 0.9}
        """
        proposals = {}
        for entity_id in perceived_entities:
            # そのエンティティに関する記憶を想起する
            recalled_memories = self.retrieve_memories_about_entity(entity_id)
            if not recalled_memories:
                continue

            # 想起した記憶の感情価と顕著性を基に、総合的な感情を計算
            # 顕著性が高い（鮮明な）記憶ほど影響が大きくなる
            total_valence = sum(mem.valence * mem.salience for mem in recalled_memories)
            total_salience = sum(mem.salience for mem in recalled_memories)
            
            if total_salience == 0: continue

            average_weighted_valence = total_valence / total_salience

            # 総合的な感情価に基づいて行動を提案する
            # これが「エピソード慣性」の発現
            if average_weighted_valence > 0.3:  # ポジティブな記憶が優位な場合
                # スコアは感情価の強さと記憶の鮮明さに依存する
                score = min(1.0, average_weighted_valence * total_salience)
                proposals[f"APPROACH_{entity_id}"] = score
                proposals[f"GREET_{entity_id}"] = score * 0.8 # 挨拶は少し低めのスコアで
            elif average_weighted_valence < -0.3: # ネガティブな記憶が優位な場合
                score = min(1.0, abs(average_weighted_valence) * total_salience)
                proposals[f"AVOID_{entity_id}"] = score

        return proposals

    def decay_memories(self):
        """
        全ての記憶の顕著性を時間と共に減衰させる（忘却のプロセス）。
        優位性の遷移を表現する。
        """
        for memory in self.memories:
            memory.salience *= self.memory_decay_rate
        
        # 顕著性が非常に低くなった古い記憶はリストから削除して、処理を軽量化
        self.memories = [mem for mem in self.memories if mem.salience > 0.05]


# --- 実行デモ ---
if __name__ == '__main__':
    # NPC "VillagerA" のエピソード記憶システムを作成
    npc_memory = EpisodicMemorySystem(owner_id="VillagerA")

    # 1. プレイヤーからプレゼントをもらう（ポジティブな出来事）
    npc_memory.record_event(
        event_type="GIFT_RECEIVED",
        location="plaza",
        involved_entities=["PlayerA", "VillagerA"],
        valence=0.9
    )

    # 2. VillagerB と口論になる（ネガティブな出来事）
    npc_memory.record_event(
        event_type="ARGUMENT",
        location="beach",
        involved_entities=["VillagerB", "VillagerA"],
        valence=-0.8
    )

    print("\n--- 1. Initial State ---")
    # この時点で、VillagerAがプレイヤーとVillagerBを発見したらどうなるか？
    perceived = ["PlayerA", "VillagerB"]
    proposals = npc_memory.generate_action_proposals(perceived)
    print(f"Perceived: {perceived}")
    print(f"Action Proposals: {proposals}")
    # 予想結果: PlayerAに接近(APPROACH)し、VillagerBを避ける(AVOID)提案が生成される

    # 3. 時間が経過し、記憶が少し風化する
    print("\n--- 2. After time passes ---")
    for _ in range(5): # 5回分の更新をシミュレート
        npc_memory.decay_memories()

    proposals_after_decay = npc_memory.generate_action_proposals(perceived)
    print(f"Memories decayed...")
    print(f"Action Proposals after decay: {proposals_after_decay}")
    # 予想結果: 提案スコアが全体的に少し低下している

    # 4. プレイヤーと再び楽しい時間を過ごす（記憶の強化）
    print("\n--- 3. Reinforcing a positive memory ---")
    npc_memory.record_event(
        event_type="SHARED_HOBBY",
        location="plaza",
        involved_entities=["PlayerA", "VillagerA"],
        valence=0.7
    )
    proposals_reinforced = npc_memory.generate_action_proposals(perceived)
    print(f"Perceived: {perceived}")
    print(f"Action Proposals (reinforced): {proposals_reinforced}")
    # 予想結果: PlayerAへの接近スコアが、前よりもさらに高くなる