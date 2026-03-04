import os
import time
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document

load_dotenv()

# --- 1. AYARLAR ---
KLASOR_YOLU = "son-veri"

# --- 2. OLLAMA (Rate limit YOK!) ---
print("🔌 Ollama ve Neo4j bağlantıları kuruluyor...")
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

# --- 3. ŞEMA (İNGİLİZCE - Model bunu anlıyor) ---
allowed_nodes = [
    "Structure",
    "Person", 
    "Location",
    "Period",
    "Event",
    "Style",
    "Material",
]

allowed_relationships = [
    "LOCATED_IN",
    "BUILT_BY",
    "DESIGNED_BY",
    "BUILT_IN",
    "RESTORED_BY",
    "NEXT_TO",
    "OPPOSITE_TO",
    "PART_OF",
    "REPLACED",
    "HAS_STYLE",
    "MADE_OF",
    "DAMAGED_BY",
    "BURIED_IN",
    "STUDENT_OF",
    "BROUGHT_BY",
    "BROUGHT_FROM",
]

# --- 4. TRANSFORMER ---
llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=allowed_nodes,
    allowed_relationships=allowed_relationships,
    strict_mode=False,
)

# --- 5. NEO4J BAĞLANTISI ---
graph = Neo4jGraph(
    url=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    username=os.environ.get("NEO4J_USERNAME", "neo4j"),
    password=os.environ["NEO4J_PASSWORD"]
)


def graph_temizle():
    print("🗑️  Graph siliniyor...")
    graph.query("MATCH (n) DETACH DELETE n")
    print("✅ Graph temizlendi.\n")


def main():
    print("\n" + "="*50)
    mevcut = graph.query("MATCH (n) RETURN count(n) as sayi")[0]["sayi"]
    print(f"📊 Mevcut graph'te {mevcut} node var.")
    
    if mevcut > 0:
        cevap = input("❓ Mevcut graph silinsin mi? (e/h): ").strip().lower()
        if cevap == 'e':
            graph_temizle()
        else:
            print("ℹ️  Mevcut graph korunuyor.\n")
    print("="*50 + "\n")
    
    if not os.path.exists(KLASOR_YOLU):
        print(f"❌ '{KLASOR_YOLU}' klasörü yok!")
        return

    dosyalar = [f for f in os.listdir(KLASOR_YOLU) if f.endswith(".txt")]
    print(f"📂 {len(dosyalar)} dosya bulundu. İşlem başlıyor...\n")

    toplam_node = 0
    toplam_relationship = 0

    for i, dosya_adi in enumerate(dosyalar):
        print(f"🔄 [{i+1}/{len(dosyalar)}] İşleniyor: {dosya_adi}")
        
        try:
            with open(os.path.join(KLASOR_YOLU, dosya_adi), "r", encoding="utf-8") as f:
                text = f.read()
            
            print(f"   📄 {len(text)} karakter okundu")
            
            doc = Document(page_content=text, metadata={"kaynak": dosya_adi})
            start = time.time()
            
            graph_documents = llm_transformer.convert_to_graph_documents([doc])
            
            if graph_documents and graph_documents[0].nodes:
                print(f"   🔍 Örnek: {graph_documents[0].nodes[0]}")
            
            for gd in graph_documents:
                toplam_node += len(gd.nodes)
                toplam_relationship += len(gd.relationships)
            
            graph.add_graph_documents(graph_documents, include_source=True)
            
            node_sayisi = len(graph_documents[0].nodes) if graph_documents else 0
            rel_sayisi = len(graph_documents[0].relationships) if graph_documents else 0
            
            print(f"   ✅ Tamamlandı ({time.time() - start:.2f} sn)")
            print(f"      📊 Node: {node_sayisi}, İlişki: {rel_sayisi}")
            
        except Exception as e:
            print(f"   ❌ HATA: {e}")

    print(f"\n{'='*50}")
    print(f"🏁 YÜKLEME BİTTİ!")
    print(f"📊 TOPLAM: {toplam_node} node, {toplam_relationship} ilişki")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()