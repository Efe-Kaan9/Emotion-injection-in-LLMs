import json
import os
import pandas as pd
import warnings
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

warnings.filterwarnings("ignore")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

def tokenize(text):
    if not isinstance(text, str): return []
    return nltk.word_tokenize(text.lower())

def calc_distinct_n(texts, n):
    ngrams = set()
    total = 0
    for t in texts:
        toks = tokenize(t)
        for i in range(len(toks) - n + 1):
            ngrams.add(tuple(toks[i:i+n]))
            total += 1
    return len(ngrams) / total if total > 0 else 0.0

def calc_self_bleu(texts):
    if len(texts) < 2: return 0.0
    smooth = SmoothingFunction().method1
    tokenized = [tokenize(t) for t in texts]
    scores = []
    for i, hyp in enumerate(tokenized):
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        if not refs or not hyp: continue
        scores.append(sentence_bleu(refs, hyp, smoothing_function=smooth))
    return sum(scores) / len(scores) if scores else 0.0

def load_json(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(f"[UYARI] {file_name} bulunamadı!")
    return None

def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# --- HAYAT KURTARAN DİNAMİK TEMİZLEYİCİ ---
def clean_generation(raw_text, prompt_text, emotion_text):
    """
    Hem Vanilla/Var2 içindeki "nötr promptu" 
    Hem de SysPrompt/LoRA içindeki "Rewrite..." şablonunu dinamik olarak siler.
    Sadece modelin ürettiği saf 'response' döner.
    """
    if not isinstance(raw_text, str): return ""
    t = raw_text.strip()
    p = str(prompt_text).strip()
    e = str(emotion_text).strip()
    
    # 1. System Prompt / LoRA Şablonlarını Kes
    t1 = f"Rewrite the following sentence conveying the emotion of {e.upper()}: {p}"
    t2 = f"Rewrite the following sentence conveying the emotion of {e}: {p}"
    
    if t.startswith(t1):
        t = t[len(t1):].strip()
    elif t.startswith(t2):
        t = t[len(t2):].strip()
        
    # 2. Vanilla / Var1 / Var2 Nötr Promptlarını Kes
    if t.startswith(p):
        t = t[len(p):].strip()
        
    return t.strip()

print("="*90)
print(" 🚀 FAIR EVALUATION SCRIPT (TABLE 4 & 5) - STRICT N=50 INTRA-GROUP POOLING ")
print(" 🛡️  [GÜVENLİK AKTİF] Tablo 5 için prompt şablonları dinamik olarak siliniyor...")
print("="*90)

mech_phi = load_json("eval_mechanistic_phi4.json")
mech_qwen = load_json("eval_mechanistic_qwen2.5.json")
lora_phi = load_json("eval_mechanistic_lora_phi4_v2.json")
lora_qwen = load_json("eval_mechanistic_lora_qwen_v2.json")
sys_prompt = load_json("system_prompt_results.json")

def process_model_for_all_tables(model_name, model_key, mech_data, lora_data, sys_data, opt_alpha, opt_layer):
    print(f"\n>>> Analiz Ediliyor: {model_name} (Şampiyon Parametreler: Alpha={opt_alpha}, Layer={opt_layer})")

    df_mech = pd.DataFrame(mech_data.get("records", []))
    df_lora = pd.DataFrame(lora_data.get("records", []))
    df_sys = pd.DataFrame(sys_data.get(model_key, {}).get("records", []))

    # --- Şampiyon (Core) 50 Seti Belirle ---
    df_var2_optimal = df_mech[(df_mech['variant'] == 2) & (df_mech['alpha'] == opt_alpha) & (df_mech['layer_config'] == opt_layer)]
    emo_col_ref = get_col(df_var2_optimal, ['target_emotion', 'emotion'])
    core_keys = set(zip(df_var2_optimal['prompt'], df_var2_optimal[emo_col_ref]))

    # --- Bütün Yöntemleri Filtrele ---
    def filter_core_50(df, variant=None, alpha=None, layer=None):
        if df.empty: return pd.DataFrame()
        temp = df.copy()
        if variant is not None and 'variant' in temp.columns: temp = temp[temp['variant'] == variant]
        if alpha is not None and 'alpha' in temp.columns: temp = temp[temp['alpha'] == alpha]
        if layer is not None and 'layer_config' in temp.columns: temp = temp[temp['layer_config'] == layer]
        
        e_col = get_col(temp, ['target_emotion', 'emotion'])
        p_col = get_col(temp, ['prompt', 'text'])
        
        if e_col and p_col:
            temp = temp[temp.apply(lambda r: (r[p_col], r[e_col]) in core_keys, axis=1)]
        return temp

    Dataset_Var2 = filter_core_50(df_mech, variant=2, alpha=opt_alpha, layer=opt_layer)
    Dataset_Var1 = filter_core_50(df_mech, variant=1, alpha=opt_alpha, layer=opt_layer)
    Dataset_LoRA = filter_core_50(df_lora)
    Dataset_Sys  = filter_core_50(df_sys)
    
    # -----------------------------------------------------------------------------------
    # TABLO 4: MECHANISTIC (Prompt Dahil - Syntactic Burden Ölçümü)
    # -----------------------------------------------------------------------------------




    # -----------------------------------------------------------------------------------
    # TABLO 5: LINGUISTIC (SADECE SAF RESPONSE - PROMPTLAR SİLİNMİŞTİR)
    # -----------------------------------------------------------------------------------
    def calc_table5_metrics_grouped(df, is_vanilla=False):
        if df.empty: return 0.0, 0.0, 0.0
        
        text_col = get_col(df, ['vanilla_text']) if is_vanilla else get_col(df, ['generated_text', 'system_text', 'text'])
        emo_col = get_col(df, ['target_emotion', 'emotion'])
        p_col = get_col(df, ['prompt', 'text'])
        
        if not text_col or not emo_col or not p_col: return 0.0, 0.0, 0.0
        
        d1_scores, d2_scores, sb_scores = [], [], []
        
        for _, group in df.groupby(emo_col):
            texts = []
            for _, row in group.iterrows():
                raw_text = str(row[text_col])
                # HAYAT KURTARAN DİNAMİK SİLME İŞLEMİ
                clean_text = clean_generation(raw_text, row[p_col], row[emo_col])
                if clean_text:
                    texts.append(clean_text)
            
            if len(texts) == 0: continue
            
            d1_scores.append(calc_distinct_n(texts, 1))
            d2_scores.append(calc_distinct_n(texts, 2))
            sb_scores.append(calc_self_bleu(texts))
            
        if not d1_scores: return 0.0, 0.0, 0.0
        return sum(d1_scores)/len(d1_scores), sum(d2_scores)/len(d2_scores), sum(sb_scores)/len(sb_scores)

    Table5_Var2 = calc_table5_metrics_grouped(Dataset_Var2)
    Table5_Var1 = calc_table5_metrics_grouped(Dataset_Var1)
    Table5_LoRA = calc_table5_metrics_grouped(Dataset_LoRA)
    Table5_Sys  = calc_table5_metrics_grouped(Dataset_Sys)
    Table5_Vanilla = calc_table5_metrics_grouped(Dataset_Var2, is_vanilla=True)

    print(f"\n    [TABLE 5 - LINGUISTIC (Tüm promptlar silindi, SADECE modelin ürettiği saf İngilizce üzerinden hesaplandı)]")
    print(f"    {'Method':<20} | {'Dist-1':<8} | {'Dist-2':<8} | {'Self-BLEU':<8}")
    print(f"    {'-'*60}")
    print(f"    {'Vanilla':<20} | {Table5_Vanilla[0]:<8.4f} | {Table5_Vanilla[1]:<8.4f} | {Table5_Vanilla[2]:<8.4f}")
    print(f"    {'Sys. Prompt':<20} | {Table5_Sys[0]:<8.4f} | {Table5_Sys[1]:<8.4f} | {Table5_Sys[2]:<8.4f}")
    print(f"    {'LoRA':<20} | {Table5_LoRA[0]:<8.4f} | {Table5_LoRA[1]:<8.4f} | {Table5_LoRA[2]:<8.4f}")
    print(f"    {'Steered (Var 1)':<20} | {Table5_Var1[0]:<8.4f} | {Table5_Var1[1]:<8.4f} | {Table5_Var1[2]:<8.4f}")
    print(f"    {'Steered (Var 2)':<20} | {Table5_Var2[0]:<8.4f} | {Table5_Var2[1]:<8.4f} | {Table5_Var2[2]:<8.4f}\n")

if mech_phi and lora_phi and sys_prompt:
    process_model_for_all_tables("Phi-4-mini", "phi4", mech_phi, lora_phi, sys_prompt, opt_alpha=1.0, opt_layer="all")
if mech_qwen and lora_qwen and sys_prompt:
    process_model_for_all_tables("Qwen-2.5", "qwen2.5", mech_qwen, lora_qwen, sys_prompt, opt_alpha=1.5, opt_layer="second_half")