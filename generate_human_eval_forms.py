import re
import random

def parse_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    samples = []
    # Her bir Sample bloğunu ayır (Sample #000 ile başlayan bloklar)
    blocks = re.split(r'------------------------------------------------------------------------\nSample', content)
    
    for block in blocks[1:]: # İlk kısım başlık olduğu için atlıyoruz
        sample_data = {}
        
        # Sample ID ve Emotion
        header_match = re.match(r' #(\d+)\s+Emotion:\s+(\w+)', block)
        if header_match:
            sample_data['id'] = header_match.group(1)
            sample_data['emotion'] = header_match.group(2)
            
        # Input Prompt
        input_match = re.search(r'INPUT PROMPT:\n\s+(.*?)\n\n\[A\]', block, re.DOTALL)
        if input_match:
            sample_data['input'] = input_match.group(1).strip()
            
        # Modeller ve Çıktıları
        outputs = {}
        
        vanilla_match = re.search(r'\[A\] VANILLA OUTPUT:\n\s+(.*?)(?=\n\n\[B\]|\Z)', block, re.DOTALL)
        if vanilla_match: outputs['Vanilla'] = vanilla_match.group(1).strip()
            
        sys_match = re.search(r'\[B\] SYSTEM-PROMPT OUTPUT:\n\s+(.*?)(?=\n\n\[C\]|\Z)', block, re.DOTALL)
        if sys_match: outputs['System-Prompt'] = sys_match.group(1).strip()
            
        steered_match = re.search(r'\[C\] STEERED V2 OUTPUT:\n\s+(.*?)(?=\n\n\[D\]|\Z)', block, re.DOTALL)
        if steered_match: outputs['Steered V2'] = steered_match.group(1).strip()
            
        lora_match = re.search(r'\[D\] LORA OUTPUT:\n\s+(.*?)(?=\n\n----------------|\Z)', block, re.DOTALL)
        if lora_match: outputs['LoRA'] = lora_match.group(1).strip()
            
        sample_data['outputs'] = outputs
        samples.append(sample_data)
        
    return samples

def generate_html(samples, model_name, form_filename, key_filename):
    form_html = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <title>AI Metin Üretimi - Değerlendirme Formu ({model_name})</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 11pt; color: #333; line-height: 1.5; margin: 40px auto; max-width: 800px; }}
            h2 {{ text-align: center; color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
            .instructions {{ background-color: #f8f9fa; padding: 15px; border-left: 5px solid #3498db; margin-bottom: 30px; font-size: 10pt; }}
            .sample-card {{ border: 1px solid #ccc; padding: 20px; margin-bottom: 25px; page-break-inside: avoid; border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }}
            .sample-header {{ font-weight: bold; font-size: 12pt; background-color: #e8f4f8; padding: 10px; margin: -20px -20px 15px -20px; border-radius: 8px 8px 0 0; border-bottom: 1px solid #ccc; }}
            .prompt-box {{ background-color: #fffdee; padding: 10px; border-left: 4px solid #f1c40f; margin-bottom: 15px; font-style: italic; }}
            .option-block {{ margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px dashed #eee; }}
            .option-text {{ margin-bottom: 10px; }}
            .rating-box {{ display: flex; justify-content: space-between; background-color: #fafafa; padding: 8px 15px; border-radius: 5px; border: 1px solid #eee; }}
            .rating-scale {{ font-family: monospace; font-size: 11pt; letter-spacing: 2px; }}
            @media print {{
                body {{ margin: 0; padding: 10px; }}
                .instructions {{ display: block; }}
            }}
        </style>
    </head>
    <body>
        <h2>AI Metin Üretimi - İnsan Değerlendirme Formu</h2>
        <div class="instructions">
            <p>Bu çalışmaya katıldığınız için teşekkür ederiz. Aşağıda çeşitli senaryolar için bir <b>'Girdi (Input)'</b> ve yapay zeka tarafından üretilmiş 4 farklı <b>'Çıktı (Option)'</b> göreceksiniz. Lütfen her bir çıktıyı aşağıdaki 2 kritere göre 1 ile 5 arasında puanlayınız (İlgili rakamı yuvarlak içine alınız):</p>
            <ul style="margin-top: 5px; margin-bottom: 0;">
                <li><b>Duygu (Emotion):</b> Üretilen metin, istenen hedef duyguyu ne kadar başarılı yansıtıyor? (1: Çok Kötü/Alakasız - 5: Çok Başarılı/Doğal)</li>
                <li><b>Akıcılık (Fluency):</b> Üretilen metin dilbilgisi açısından ne kadar düzgün ve akıcı? (1: Çok Bozuk/Tekrarlayan - 5: Kusursuz/İnsansı)</li>
            </ul>
        </div>
    """

    key_html = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <title>Master Key - {model_name}</title>
        <style>
            body {{ font-family: monospace; font-size: 12pt; line-height: 1.6; margin: 40px; }}
            .sample {{ margin-bottom: 15px; border-bottom: 1px dashed #ccc; padding-bottom: 10px; }}
        </style>
    </head>
    <body>
        <h2>Gizli Cevap Anahtarı (Master Key) - {model_name}</h2>
    """

    for sample in samples:
        form_html += f"""
        <div class="sample-card">
            <div class="sample-header">
                Soru #{sample['id']} | Hedef Duygu: <span style="color: #c0392b;">{sample['emotion'].upper()}</span>
            </div>
            <div class="prompt-box">
                <b>Girdi (Input):</b> {sample['input']}
            </div>
        """
        
        # Seçenekleri Karıştır
        outputs = list(sample['outputs'].items())
        random.shuffle(outputs)
        
        key_html += f"<div class='sample'><b>Sample #{sample['id']} ({model_name})</b><br>"
        
        for idx, (model_id, text) in enumerate(outputs):
            option_num = idx + 1
            if not text: text = "(Boş Çıktı / Üretim Yok)"
            
            form_html += f"""
            <div class="option-block">
                <div style="font-weight: bold; margin-bottom: 5px; color: #2980b9;">Option {option_num}:</div>
                <div class="option-text">"{text}"</div>
                <div class="rating-box">
                    <div><b>Duygu (Emotion):</b> <span class="rating-scale">[1] [2] [3] [4] [5]</span></div>
                    <div><b>Akıcılık (Fluency):</b> <span class="rating-scale">[1] [2] [3] [4] [5]</span></div>
                </div>
            </div>
            """
            
            key_html += f"Option {option_num}: {model_id} | "
            
        form_html += "</div>" # sample-card div'i kapat
        key_html += "</div>"

    form_html += "</body></html>"
    key_html += "</body></html>"

    with open(form_filename, 'w', encoding='utf-8') as f:
        f.write(form_html)
    with open(key_filename, 'w', encoding='utf-8') as f:
        f.write(key_html)

# Ana Çalıştırma
phi4_samples = parse_file('human_eval_phi4.txt')
generate_html(phi4_samples, 'Phi-4-mini', 'guzel_form_phi4.html', 'master_key_phi4.html')

qwen_samples = parse_file('human_eval_qwen2.5.txt')
generate_html(qwen_samples, 'Qwen-2.5', 'guzel_form_qwen.html', 'master_key_qwen.html')

print("İşlem tamam! 'guzel_form_phi4.html' ve 'guzel_form_qwen.html' dosyalarını tarayıcıda açıp Ctrl+P ile yazdırabilirsin.")