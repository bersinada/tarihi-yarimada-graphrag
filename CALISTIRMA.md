# GraphRAG Chatbot - Çalıştırma Kılavuzu

## Ön Koşullar
- Docker Desktop açık olmalı
- `.env` dosyası mevcut olmalı (API anahtarları ve Neo4j şifresi)
  - `GOOGLE_API_KEY` - Google Gemini API anahtarı
  - `NEO4J_PASSWORD` - Neo4j veritabanı şifresi

## Çalıştırma (3 Adım)

### 1. Neo4j Veritabanını Başlat
```powershell
docker compose up -d
```
Kontrol: http://localhost:7474 adresinden Neo4j Browser açılmalı.

### 2. GraphRAG API'yi Başlat
```powershell
cd c:\Users\Sinan\gis-projects\tarihi-yarimada\tarihi-yarimada-graph
.\.venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8002
```
Terminalde `Application startup complete` yazısını görene kadar bekle (~15 sn).  
Kontrol: http://localhost:8002/health

### 3. Tarihi Yarımada Web Sitesini Başlat
Frontend'i ayrı bir terminalden port 8080'de çalıştır (tarihi-yarimada-cbs projesi).

## Port Haritası

| Servis           | Port | Açıklama              |
|------------------|------|-----------------------|
| Neo4j Browser    | 7474 | Veritabanı arayüzü   |
| Neo4j Bolt       | 7687 | Veritabanı bağlantısı |
| GraphRAG API     | 8002 | Chatbot backend       |
| Web Sitesi       | 8080 | Frontend              |

## Kapatma
```powershell
# Terminaldeki API'yi kapat: Ctrl+C
# Neo4j'i kapat:
docker compose down
```

## Sık Karşılaşılan Sorunlar

| Sorun | Çözüm |
|-------|-------|
| `No module named uvicorn` | `.venv` ile çalıştırın, sistem Python'u değil |
| Port meşgul hatası | `netstat -ano \| findstr ":8002"` ile kontrol edin |
| Chatbot "Bağlantı kurulamadı" diyor | API port 8002'de çalışıyor mu kontrol edin |

> **Önemli:** Conda (base) ortamında değil, `.venv` sanal ortamında çalıştırın.
> Conda kurulumu sonrası varsayılan Python değişebilir; her zaman `.venv\Scripts\python.exe` kullanın.
