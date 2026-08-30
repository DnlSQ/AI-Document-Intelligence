from src.document_loader import extract_text_from_pdf
from src.text_cleaner import clean_text

pdf_path = "data/documents/sample.pdf"

pages = extract_text_from_pdf(pdf_path)

print("=" * 40)
print("       PDF CLEANING TEST")
print("=" * 40)

for page in pages[:3]:
    cleaned_text = clean_text(page["text"])

    print(f"\nPage: {page['page']}")
    print("-" * 40)
    print(cleaned_text[:700])
