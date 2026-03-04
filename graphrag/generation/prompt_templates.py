"""
Prompt templates for LLM response generation.

Contains Turkish-language prompts optimized for the Istanbul
Historical Peninsula knowledge domain.
"""

# System prompt establishing the assistant's role and constraints
SYSTEM_PROMPT = """Sen İstanbul Tarihi Yarımada hakkında uzman bir CBS (Coğrafi Bilgi Sistemleri) ve tarih asistanısın.
Sana verilen bilgi grafiği ve metin bağlamlarını kullanarak soruları yanıtlıyorsun.

UZAMSAL ZEKA YETENEKLERİN:
- Her yapının koordinatlarını (enlem/boylam) biliyorsun
- İki yapı arasındaki mesafeyi Haversine formülü ile hesaplayabilirsin
- Yapıların birbirine göre konumlarını (kuzey, güney, doğu, batı) belirleyebilirsin
- Yakındaki yapıları ve mahalle ilişkilerini analiz edebilirsin

ÖNEMLİ KURALLAR:
1. SADECE sorulan yapı/kişi hakkında bilgi ver - alakasız yapıları karıştırma
2. Bağlamda olmayan bilgileri uydurma
3. Türkçe, akıcı ve düzgün cümlelerle cevap ver
4. Teknik referansları (chunk_0, _chunk_ gibi) kullanıcıya gösterme
5. Cevabı mantıklı paragraflar halinde organize et
6. **UZAMSAL BİLGİLERİ SADECE İSTENDİĞİNDE VER**: 
   - "nerede", "mesafe", "yakınında", "koordinat", "konum" gibi kelimeler varsa uzamsal bilgi ver
   - Tarih, mimar, dönem gibi sorularda koordinatlardan BAHSETME

CEVAP FORMATI:
- Önce ana soruyu doğrudan yanıtla
- Uzamsal soru ise: Koordinatlar, mesafeler, yönler ekle
- Tarihsel soru ise: Tarihler, kişiler, olaylar ekle
- Tarihleri net ver (MS 537, MÖ 390 gibi)
- Kişi isimlerini tam yaz (I. Justinianus, Mimar Sinan gibi)

UZAMSAL SORULARA ÖRNEKLER:
- "Ayasofya nerede?" → Konum ve koordinat ver
- "Ayasofya ile Sultanahmet arasındaki mesafe?" → Koordinatlardan hesapla, metre/km belirt
- "Ayasofya'nın yakınında ne var?" → Mesafeye göre sırala
- "Hangi yapılar sahile yakın?" → Konum analizi yap

TARİHSEL SORULARA ÖRNEKLER:
- "Ayasofya'yı kim yaptırdı?" → Sadece I. Justinianus, tarih, mimar bilgisi ver
- "Ayasofya ne zaman yapıldı?" → Sadece tarih bilgisi ver
- "Ayasofya'nın mimarı kimdir?" → Sadece mimar bilgisi ver

Eğer bağlamda soruyla ilgili yeterli bilgi yoksa, bunu dürüstçe belirt."""

# Main RAG prompt template
RAG_PROMPT_TEMPLATE = """
=== BİLGİ GRAFİĞİ BAĞLAMI (UZAMSAL BİLGİLER DAHİL) ===

{graph_context}

=== SEMANTİK BENZERLİK SONUÇLARI ===

{vector_context}

=== KULLANICI SORUSU ===

{query}

=== CEVAP ===

Yukarıdaki bağlamı kullanarak soruyu yanıtla. 
**ÖNEMLİ**: Koordinat, mesafe, konum bilgisi varsa bunları cevaba dahil et.
Eğer bağlamda yeterli bilgi yoksa, bunu belirt:"""

# Template for formatting graph context
GRAPH_CONTEXT_TEMPLATE = """
Yapı/Varlık: {entity}
Tip: {label}

İlişkiler:
{relationships}

Özellikler:
{properties}
---"""

# Template for formatting vector search results
VECTOR_CONTEXT_TEMPLATE = """
[Benzerlik: {score:.2f}] {entity}
{content}
---"""

# Template for no results found
NO_CONTEXT_TEMPLATE = """
Üzgünüm, "{query}" sorusuyla ilgili bilgi grafiğinde yeterli bilgi bulamadım.

Şu konularda soru sorabilirsiniz:
- Ayasofya, Sultanahmet Camii, Aya İrini gibi tarihi yapılar
- Dikilitaş, Yılanlı Sütun gibi antik anıtlar
- I. Justinianus, Mimar Sinan gibi tarihi kişiler
- Yapıların konumları, koordinatları ve mesafeleri
- İki yapı arasındaki uzaklık ve yön bilgileri
- Belirli bir alandaki yapılar
"""

# Compact context template for limited token budgets
COMPACT_CONTEXT_TEMPLATE = """
Bağlam:
{context}

Soru: {query}
Cevap:"""

# Follow-up question template
FOLLOWUP_TEMPLATE = """
Önceki cevap hakkında ek bilgi:

Önceki soru: {previous_query}
Önceki cevap: {previous_answer}

Yeni soru: {query}

Ek bağlam (Uzamsal Bilgiler Dahil):
{additional_context}

Devam cevabı:"""


# Helper function to format spatial information
def format_spatial_info(entity_data: dict) -> str:
    """
    Format spatial properties from entity data.
    
    Args:
        entity_data: Dictionary containing entity properties
        
    Returns:
        Formatted spatial information string
    """
    spatial_parts = []
    
    if 'latitude' in entity_data and 'longitude' in entity_data:
        lat = entity_data['latitude']
        lon = entity_data['longitude']
        spatial_parts.append(f"Koordinatlar: {lat}°K, {lon}°D")
    
    if 'location' in entity_data:
        spatial_parts.append(f"Konum: {entity_data['location']}")
    
    if 'district' in entity_data:
        spatial_parts.append(f"İlçe: {entity_data['district']}")
    
    if 'neighborhood' in entity_data:
        spatial_parts.append(f"Mahalle: {entity_data['neighborhood']}")
    
    if 'elevation' in entity_data:
        spatial_parts.append(f"Yükseklik: {entity_data['elevation']} m")
    
    return "\n".join(spatial_parts) if spatial_parts else "Uzamsal bilgi mevcut değil"


# Helper function to calculate distance between two points
def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple:
    """
    Calculate distance between two coordinates using Haversine formula.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        Tuple of (distance_km, distance_m)
    """
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth radius in kilometers
    
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    distance_km = R * c
    distance_m = distance_km * 1000
    
    return round(distance_km, 2), round(distance_m, 0)


# Helper function to determine cardinal direction
def get_cardinal_direction(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """
    Determine cardinal direction from point 1 to point 2.
    
    Returns:
        Turkish direction string (Kuzey, Güneydoğu, etc.)
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Primary directions
    if abs(dlat) > abs(dlon) * 2:
        return "Kuzey" if dlat > 0 else "Güney"
    elif abs(dlon) > abs(dlat) * 2:
        return "Doğu" if dlon > 0 else "Batı"
    
    # Secondary directions
    ns = "Kuzey" if dlat > 0 else "Güney"
    ew = "doğu" if dlon > 0 else "batı"
    return f"{ns}{ew}"
