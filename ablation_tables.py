import json
import pandas as pd

def get_ablation_data(file_name):
    print(f"\n=== ABLASYON VERİLERİ: {file_name} ===")
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    records = data.get("records", data)
    df = pd.DataFrame(records)
    
    print("\n--- TABLO 7 İÇİN: Alfa (Alpha) Dağılımı (Sadece Variant 2) ---")
    df_v2 = df[df["variant"] == 2]
    alpha_group = df_v2.groupby("alpha").agg(
        Target_Score=("steered_target_score", "mean"),
        PPL=("ppl_steered", "mean"),
        JSD=("jsd", "mean")
    ).reset_index()
    print(alpha_group.to_string(index=False))
    
    print("\n--- TABLO 8 İÇİN: Katman (Layer) Dağılımı (Genel Ortalama) ---")
    layer_group = df.groupby("layer_config").agg(
        Target_Score=("steered_target_score", "mean"),
        PPL=("ppl_steered", "mean"),
        JSD=("jsd", "mean")
    ).reset_index()
    print(layer_group.to_string(index=False))

# Çalıştır
get_ablation_data("eval_mechanistic_phi4.json")
get_ablation_data("eval_mechanistic_qwen2.5.json")