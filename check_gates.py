import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, AutoConfig

from projector_agnostic import ModelConfig, create_projector

class EmotionExtractor(torch.nn.Module):
    def __init__(self, encoder_name="distilbert-base-uncased", num_emotions=28):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name, cache_dir="./.hf_cache")
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_emotions)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        out = self.encoder(input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

def main():
    print("Makale için ZARİF YATAY 'Head-Wise' Kapı Grafiği Oluşturuluyor...")
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir="./.hf_cache")
    extractor = EmotionExtractor().to(device)
    
    if os.path.exists("checkpoint.pt"):
        sd = torch.load("checkpoint.pt", map_location=device)
        state = sd.get("model_state_dict", sd)
        extractor.load_state_dict(state)
    extractor.eval()

    cfg_phi4 = AutoConfig.from_pretrained("microsoft/Phi-4-mini-instruct", cache_dir="./.hf_cache")
    cfg_qwen = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", cache_dir="./.hf_cache")

    mc_phi4 = ModelConfig.from_hf_config(cfg_phi4, "phi4")
    mc_qwen = ModelConfig.from_hf_config(cfg_qwen, "qwen2.5")

    p2_phi4 = create_projector(mc_phi4, variant=2, device="cpu")
    if os.path.exists("checkpoints/variant2_projector_phi4.pt"):
        ckpt_phi4 = torch.load("checkpoints/variant2_projector_phi4.pt", map_location="cpu")
        p2_phi4.load_state_dict(ckpt_phi4.get("model_state_dict", ckpt_phi4))
    p2_phi4.eval()

    p2_qwen = create_projector(mc_qwen, variant=2, device="cpu")
    if os.path.exists("checkpoints/variant2_projector_qwen2.5.pt"):
        ckpt_qw = torch.load("checkpoints/variant2_projector_qwen2.5.pt", map_location="cpu")
        p2_qwen.load_state_dict(ckpt_qw.get("model_state_dict", ckpt_qw))
    p2_qwen.eval()

    prompts = {
        "Joy": "I am so incredibly happy and excited! This is a dream come true, I can't stop smiling!",
        "Anger": "This is completely unacceptable and infuriating! I demand an explanation for this terrible mistake.",
        "Sadness": "I feel so heartbroken and alone. Nothing seems to make sense anymore since this tragic loss."
    }

    results_phi4 = []
    results_qwen = []
    labels = list(prompts.keys())

    with torch.no_grad():
        for name, text in prompts.items():
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            logits = extractor(**inputs)
            emotion_vec = torch.sigmoid(logits)

            # PHI-4 MİNİ 
            _, _, gates_phi = p2_phi4(emotion_vec)
            results_phi4.append(gates_phi.squeeze().detach().cpu().numpy())

            # QWEN 2.5 
            _, _, gates_qw = p2_qwen(emotion_vec)
            results_qwen.append(gates_qw.squeeze().detach().cpu().numpy())

    results_phi4 = np.array(results_phi4)
    results_qwen = np.array(results_qwen)

    # ====================================================================
    # 5. GÖRSELLEŞTİRME (YATAY / LANDSCAPE FORMAT)
    # ====================================================================
    colors = ['#2A9D8F', '#E63946', '#457B9D']
    markers = ['o', 's', 'D'] 

    # DEĞİŞİKLİK: 1 satır, 2 sütun. Genişlik 16, yükseklik 6 inç (Yatay yerleşim)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300, facecolor='white')
    
    # DEĞİŞİKLİK: 'hspace' (dikey boşluk) yerine 'wspace' (yatay boşluk) kullandık
    plt.subplots_adjust(top=0.88, bottom=0.15, wspace=0.15, left=0.05, right=0.95)

    datasets = [
        (results_phi4, "Modulated Gate Activations\nAcross Attention Heads (Phi-4-mini)"),
        (results_qwen, "Modulated Gate Activations\nAcross Attention Heads (Qwen-2.5-1.5B)")
    ]

    for idx, (ax, (results, title)) in enumerate(zip(axes, datasets)):
        ax.set_facecolor('#FAFAFA')
        ax.grid(axis='y', linestyle='-', linewidth=0.5, color='#E0E0E0', zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')

        for i, (label, color, marker) in enumerate(zip(labels, colors, markers)):
            ax.plot(results[i], label=label, color=color, marker=marker, 
                    linewidth=2.5, markersize=8, 
                    markeredgecolor='white', markeredgewidth=1.2, 
                    zorder=3, alpha=0.9)
        
        ax.set_title(title, fontsize=14, fontweight='bold', color='#222222', pad=15)
        ax.set_xlabel("Attention Head Index", fontsize=12, color='#444444', labelpad=10)
        
        # Sadece sol grafikte Y ekseni ismini gösterelim, sağdaki temiz dursun
        if idx == 0:
            ax.set_ylabel("Gate Opening Ratio (\u03B1)", fontsize=12, color='#444444', labelpad=10)
        
        # X eksenindeki numaraları kalabalık yapmaması için ayarlıyoruz
        step = 1 if results.shape[1] <= 12 else 2
        ax.set_xticks(range(0, results.shape[1], step)) 
        ax.tick_params(axis='both', colors='#555555', labelsize=11, width=1.5)
        
        # Lejantı sadece sol grafiğe koyalım (iki kere basmaya gerek yok)
        if idx == 0:
            ax.legend(fontsize=11, loc='upper right', frameon=True, facecolor='white', edgecolor='#DDDDDD', framealpha=0.9, borderpad=0.8)

    plt.savefig("paper_gate_activations_elegant_horizontal.png", bbox_inches="tight")
    print("\n[BAŞARILI] Yatay makale görseli 'paper_gate_activations_elegant_horizontal.png' olarak kaydedildi!")

if __name__ == "__main__":
    main()