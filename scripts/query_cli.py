#!/usr/bin/env python3
"""
Interactive CLI for querying the GraphRAG system.

Run with:
    python scripts/query_cli.py

Commands:
    quit/q/exit - Exit the CLI
    status      - Show system status
    alpha <val> - Set retrieval alpha (0-1)
    help        - Show help
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphrag.facade import GraphRAGFacade


WELCOME_TEXT = """
╔══════════════════════════════════════════════════════════════╗
║     İstanbul Tarihi Yarımada GraphRAG Sistemi                ║
║     Hybrid Graph + Vector Retrieval                          ║
╚══════════════════════════════════════════════════════════════╝

Örnek sorular:
  • Ayasofya'yı kim yaptırdı?
  • Dikilitaşı Mısır'dan kim getirtti?
  • Sultanahmet Camii'nin yanında ne var?
  • Mimar Sinan'ın öğrencileri kimler?

Komutlar:
  • quit/q/exit - Çıkış
  • status      - Sistem durumu
  • alpha <0-1> - Vektör/graf dengesi (0=vektör, 1=graf)
  • help        - Yardım
"""

HELP_TEXT = """
Yardım:
  Bu sistem İstanbul Tarihi Yarımada hakkındaki sorularınızı
  hem semantik benzerlik (vektör) hem de graf ilişkileri
  kullanarak yanıtlar.

Soru türleri:
  • Olgusal: "Ayasofya ne zaman yapıldı?"
  • İlişkisel: "Sultanahmet Camii'ni kim yaptırdı?"
  • Mekansal: "Ayasofya'nın yanında ne var?"
  • Köken: "Dikilitaş nereden getirildi?"
  • Dönemsel: "Bizans döneminde hangi yapılar yapıldı?"

Bilinen yapılar:
  Ayasofya, Sultanahmet Camii, Aya İrini, Dikilitaş,
  Yılanlı Sütun, Örme Dikilitaş, Alman Çeşmesi,
  III. Ahmed Çeşmesi, Sultan Ahmed Türbesi, Firuzağa Camii

Bilinen kişiler:
  I. Justinianus, Fatih Sultan Mehmed, I. Ahmed, Mimar Sinan,
  Sedefkâr Mehmed Ağa, I. Theodosius, III. Thutmose
"""


def print_separator():
    """Print a visual separator."""
    print("-" * 60)


def show_status(rag: GraphRAGFacade):
    """Display system status."""
    print("\n📊 Sistem Durumu:")
    print_separator()

    status = rag.get_system_status()

    if status.get("neo4j_connected"):
        print("✓ Neo4j bağlantısı: Aktif")
    else:
        print("✗ Neo4j bağlantısı: Kapalı")
        return

    print(f"\nDüğüm sayıları:")
    for label, count in status.get("node_counts", {}).items():
        if count > 0:
            print(f"  • {label}: {count}")

    indexes = status.get("vector_indexes", [])
    print(f"\nVektör indeksleri: {len(indexes)}")

    doc_stats = status.get("document_stats", {})
    if doc_stats.get("total", 0) > 0:
        print(f"\nDoküman istatistikleri:")
        print(f"  • Toplam chunk: {doc_stats.get('total', 0)}")
        print(f"  • Bağlı yapı: {doc_stats.get('linked_structures', 0)}")

    print_separator()


def process_query(rag: GraphRAGFacade, query: str):
    """Process and display query results."""
    print("\n⏳ İşleniyor...\n")

    try:
        result = rag.query(query)

        # Show analysis
        print("📋 Analiz:")
        print(f"   Intent: {result.analysis.intent.value}")
        if result.analysis.entities:
            print(f"   Varlıklar: {', '.join(result.analysis.entities)}")
        print(f"   Güven: {result.analysis.confidence:.0%}")

        print_separator()

        # Show response
        print("\n💬 Yanıt:\n")
        print(result.response)

        print_separator()

        # Show sources
        if result.sources:
            print(f"\n📚 Kaynaklar: {', '.join(result.sources)}")

        # Show metadata
        meta = result.metadata
        print(f"\n📈 Sonuç: {meta.get('result_count', 0)} kayıt bulundu")

    except Exception as e:
        print(f"\n❌ Hata: {e}")


def main():
    """Run the interactive CLI."""
    print(WELCOME_TEXT)

    print("Sistem başlatılıyor...")

    try:
        rag = GraphRAGFacade()
        print("✓ Sistem hazır!\n")
    except Exception as e:
        print(f"✗ Başlatma hatası: {e}")
        print("\nKontrol edin:")
        print("  1. Neo4j çalışıyor mu? (docker-compose up -d)")
        print("  2. .env dosyasında GROQ_API_KEY var mı?")
        print("  3. config.yaml doğru ayarlanmış mı?")
        return 1

    try:
        while True:
            try:
                # Get user input
                query = input("\n🔍 Soru: ").strip()

                # Handle empty input
                if not query:
                    continue

                # Handle commands
                query_lower = query.lower()

                if query_lower in ['quit', 'q', 'exit', 'çıkış', 'cikis']:
                    print("\nGüle güle! 👋")
                    break

                elif query_lower == 'help' or query_lower == 'yardım':
                    print(HELP_TEXT)
                    continue

                elif query_lower == 'status' or query_lower == 'durum':
                    show_status(rag)
                    continue

                elif query_lower.startswith('alpha '):
                    try:
                        alpha = float(query_lower.split()[1])
                        rag.set_retrieval_alpha(alpha)
                        print(f"✓ Alpha değeri {alpha} olarak ayarlandı")
                        print(f"  (0 = sadece vektör, 1 = sadece graf)")
                    except (ValueError, IndexError):
                        print("Kullanım: alpha <0-1 arası değer>")
                    continue

                # Process as query
                process_query(rag, query)

            except KeyboardInterrupt:
                print("\n\nGüle güle! 👋")
                break

            except Exception as e:
                print(f"\n❌ Hata: {e}")

    finally:
        rag.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
