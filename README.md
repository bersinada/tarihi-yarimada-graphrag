# GraphRAG — Tarihi Yarımada Chatbot Motoru

> **Graph-based Retrieval-Augmented Generation** sistemi. Neo4j bilgi grafiği, vektör benzerlik araması ve Google Gemini LLM'i birleştirerek İstanbul Tarihi Yarımada'sındaki tarihi yapılar hakkında akıllı sorgulamaya olanak tanır.

Bu proje kendi başına bağımsız bir **REST API** olarak çalışmaktadır. Herhangi bir web sitesine veya uygulamaya kolayca entegre edilebilir.

---

## ✨ Özellikler

- **Hibrit Arama** — Vektör benzerliği (Sentence Transformers) + Graf geçişi (Neo4j) + Reciprocal Rank Fusion
- **Mekansal Sorgular** — İki yapı arasındaki mesafeyi ve yönü Türkçe olarak hesaplar
- **Niyete Duyarlı Yanıtlar** — Sorgunun `factual`, `spatial`, `temporal`, `origin` vb. niyetini otomatik tespit eder
- **LLM Destekli Analiz** — Google Gemini ile doğal dil anlama ve yanıt üretimi
- **REST API** — FastAPI tabanlı, kolayca entegre edilebilir endpoint'ler

---

## 🏗️ Mimari

```
Kullanıcı Sorusu
       │
       ▼
┌─────────────────┐
│  QueryAnalyzer  │  → Niyet tespiti + Varlık çıkarımı (LLM + kural tabanlı)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│           HybridRetriever           │
│  ┌─────────────┐  ┌──────────────┐  │
│  │VectorSearch │  │ GraphSearch  │  │  RRF ile birleştirme
│  │(Embeddings) │  │ (Neo4j Cypher│  │
│  └─────────────┘  └──────────────┘  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ResponseGenerator│  → Google Gemini ile Türkçe yanıt üretimi
└─────────────────┘
```

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.10+
- Docker Desktop
- Google Gemini API anahtarı ([buradan alın](https://aistudio.google.com/app/apikey))

### 1. Depoyu Klonlayın

```bash
git clone https://github.com/kullanici-adi/tarihi-yarimada-graph.git
cd tarihi-yarimada-graph
```

### 2. Sanal Ortam Oluşturun

```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# veya
source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
```

### 3. Ortam Değişkenlerini Ayarlayın

```bash
cp .env.example .env
```

`.env` dosyasını açıp doldurun:

```env
GOOGLE_API_KEY=your_google_api_key_here
NEO4J_PASSWORD=your_strong_password_here
```

### 4. Neo4j'yi Başlatın

```bash
docker compose up -d
```

> Neo4j arayüzü: http://localhost:7474 (kullanıcı: `neo4j`, şifre: `.env`'de belirlediğiniz)

### 5. Veritabanını Doldurun

`son-veri/` klasörüne kendi kaynak metin dosyalarınızı `.txt` formatında ekleyin, ardından:

```bash
python yukleme.py
```

> Bu adım Ollama + llama3.1 kullanır. Kendi LLM'inizi kullanmak istiyorsanız `yukleme.py` dosyasını düzenleyin.

### 6. Vektör İndeksini Oluşturun

```bash
python scripts/setup_vector_index.py
```

### 7. API'yi Başlatın

```bash
.\.venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8002
```

API hazır olduğunda: `Application startup complete` mesajını göreceksiniz.

---

## 📡 API Kullanımı

### Health Check

```bash
GET http://localhost:8002/health
```

```json
{"status": "healthy", "service": "graphrag-api"}
```

### Sorgulama

```bash
POST http://localhost:8002/query
Content-Type: application/json

{
  "query": "Ayasofya'yı kim yaptırdı?",
  "alpha": 0.5
}
```

**Yanıt:**

```json
{
  "success": true,
  "query": "Ayasofya'yı kim yaptırdı?",
  "response": "Ayasofya, Bizans İmparatoru I. Justinianus tarafından 532-537 yılları arasında yaptırılmıştır.",
  "analysis": {
    "intent": "builder",
    "entities": ["Ayasofya"],
    "confidence": 0.95
  },
  "sources": ["Ayasofya"],
  "metadata": {"result_count": 5}
}
```

**`alpha` parametresi:**
| Değer | Anlam |
|-------|-------|
| `0.0` | Yalnızca vektör araması |
| `0.5` | Dengeli (varsayılan) |
| `1.0` | Yalnızca graf araması |

### Sistem Durumu

```bash
GET http://localhost:8002/status
```

---

## 🌐 Web Entegrasyonu

`web/chatbot-widget.js` dosyası herhangi bir HTML sayfasına eklenebilir:

```html
<script src="chatbot-widget.js"></script>
<script>
  ChatbotWidget.init({
    apiUrl: 'http://localhost:8002',
    title: 'Tarihi Yarımada Rehberi'
  });
</script>
```

Tam örnek için `web/example.html` dosyasına bakın.

---

## 📂 Proje Yapısı

```
tarihi-yarimada-graph/
├── api.py                  # FastAPI uygulaması
├── config.yaml             # Sistem konfigürasyonu
├── yukleme.py              # Graf veritabanı doldurma scripti
├── requirements.txt
├── docker-compose.yaml     # Neo4j container
│
├── graphrag/               # Ana kütüphane
│   ├── facade.py           # Üst seviye arayüz (ana giriş noktası)
│   ├── config.py           # Konfigürasyon yönetimi
│   ├── database/           # Neo4j bağlantısı
│   ├── embeddings/         # Sentence Transformer tabanlı vektörleştirme
│   ├── query/              # Niyet ve varlık analizi
│   ├── retrieval/          # Hibrit arama (vektör + graf)
│   ├── generation/         # LLM yanıt üretimi
│   └── utils/              # Mekansal hesaplamalar (haversine vb.)
│
├── son-veri/               # Kaynak metin dosyaları (.txt)
├── scripts/                # Yardımcı araçlar
└── web/                    # Chatbot widget ve örnek sayfa
```

---

## ⚙️ Konfigürasyon

`config.yaml` üzerinden tüm sistem parametreleri ayarlanabilir:

```yaml
embeddings:
  model: "paraphrase-multilingual-MiniLM-L12-v2"  # Çok dilli model

retrieval:
  hybrid_alpha: 0.4      # 0=vektör ağırlıklı, 1=graf ağırlıklı
  vector_top_k: 8        # Vektör aramada kaç sonuç getirilsin
  graph_max_hops: 3      # Graf geçişinde maksimum adım sayısı

llm:
  model: "gemini-2.5-flash"
  temperature: 0.1
```

---

## 🗺️ Port Haritası

| Servis        | Port | Açıklama              |
|---------------|------|-----------------------|
| Neo4j Browser | 7474 | Veritabanı arayüzü   |
| Neo4j Bolt    | 7687 | Veritabanı bağlantısı |
| GraphRAG API  | 8002 | Chatbot backend       |

---

## 🔄 Kendi Verinizle Kullanmak

Bu sistem **herhangi bir alan** için uyarlanabilir:

1. `son-veri/` klasörüne kendi `.txt` kaynaklarınızı ekleyin
2. `graphrag/query/analyzer.py` içindeki `KNOWN_STRUCTURES` ve `KNOWN_PERSONS` listelerini güncelleyin
3. `config.yaml` içindeki `vector_index.indexed_labels` alanını düzenleyin
4. `yukleme.py` scriptini çalıştırarak yeni grafı oluşturun

---

## 📄 Lisans

MIT License — Serbestçe kullanabilir, değiştirebilir ve dağıtabilirsiniz.
