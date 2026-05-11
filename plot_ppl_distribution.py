import json
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('Agg')

""""
[PHI4] - ppl_steered Perplexity Distribution:
  Total Valid Samples : 746
  50th Percentile (Median) : 8.37
  75th Percentile          : 11.97
  90th Percentile          : 16.16
  95th Percentile          : 19.88
  99th Percentile          : 36.48
  Max Value (Outlier)      : 27019075704847652.00

[QWEN2.5] - ppl_steered Perplexity Distribution:
  Total Valid Samples : 750
  50th Percentile (Median) : 14.71
  75th Percentile          : 23.32
  90th Percentile          : 36.84
  95th Percentile          : 48.91
  99th Percentile          : 78.68
  Max Value (Outlier)      : 182593.37
"""

# Kendi JSON dosyanın adını buraya yaz
JSON_FILE = "real_world_raw.json" 

def analyze_ppl(data, model_name, method_key):
    # Sadece geçerli (inf veya NaN olmayan, 0'dan büyük) PPL değerlerini al
    ppls = [item[method_key] for item in data if item["model"] == model_name and item[method_key] < float('inf')]
    ppls = np.array(ppls)
    
    if len(ppls) == 0:
        return
        
    print(f"\n[{model_name.upper()}] - {method_key} Perplexity Distribution:")
    print(f"  Total Valid Samples : {len(ppls)}")
    print(f"  50th Percentile (Median) : {np.percentile(ppls, 50):.2f}")
    print(f"  75th Percentile          : {np.percentile(ppls, 75):.2f}")
    print(f"  90th Percentile          : {np.percentile(ppls, 90):.2f}")
    print(f"  95th Percentile          : {np.percentile(ppls, 95):.2f}")
    print(f"  99th Percentile          : {np.percentile(ppls, 99):.2f}")
    print(f"  Max Value (Outlier)      : {np.max(ppls):.2f}")
    
    return ppls

# Veriyi yükle
with open(JSON_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Analizi yazdır
phi4_steered = analyze_ppl(raw_data, "phi4", "ppl_steered")
qwen_steered = analyze_ppl(raw_data, "qwen2.5", "ppl_steered")

# ---------------- CDF GRAFİĞİ ÇİZİMİ (Makale İçin) ----------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 5), dpi=300)

# Uç değerleri keserek grafiği okunabilir yapalım (Örn: PPL 100'e kadar olanları göster)
PPL_CUTOFF = 100

if phi4_steered is not None:
    sorted_phi = np.sort(phi4_steered)
    y_phi = np.arange(1, len(sorted_phi) + 1) / len(sorted_phi)
    plt.plot(sorted_phi[sorted_phi < PPL_CUTOFF], y_phi[sorted_phi < PPL_CUTOFF], 
             label='Phi-4-mini (Steered V2)', linewidth=2.5, color='#1a5276')

if qwen_steered is not None:
    sorted_qwen = np.sort(qwen_steered)
    y_qwen = np.arange(1, len(sorted_qwen) + 1) / len(sorted_qwen)
    plt.plot(sorted_qwen[sorted_qwen < PPL_CUTOFF], y_qwen[sorted_qwen < PPL_CUTOFF], 
             label='Qwen-2.5 (Steered V2)', linewidth=2.5, color='#c0392b')

plt.axhline(y=0.90, color='gray', linestyle='--', alpha=0.7)
plt.text(PPL_CUTOFF * 0.8, 0.92, '90th Percentile', color='gray', fontsize=10)

plt.title('Cumulative Distribution of Perplexity (OOD Evaluation)')
plt.xlabel('Perplexity (PPL)')
plt.ylabel('Proportion of Samples')
plt.xlim(0, PPL_CUTOFF)
plt.ylim(0, 1.05)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('ppl_cdf_distribution.pdf', bbox_inches='tight')
print("\n[SAVED] Cumulative Distribution plot saved as 'ppl_cdf_distribution.pdf'")