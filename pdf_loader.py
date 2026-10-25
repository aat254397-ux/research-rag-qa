from pathlib import Path

from pypdf import PdfReader


def load_pdf_pages(file_path):
    pdf_path = Path(file_path).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF file does not exist: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("file must be a PDF")

    document_pages = []

    with pdf_path.open("rb") as pdf_file:
        reader = PdfReader(pdf_file)

        if reader.is_encrypted:
            raise ValueError("暂不支持加密 PDF，请使用未加密版本。")

        for page_number, pdf_page in enumerate(
            reader.pages,
            start=1,
        ):
            text = (pdf_page.extract_text() or "").strip()

            document_pages.append({
                "source": pdf_path.name,
                "path": str(pdf_path),
                "page": page_number,
                "text": text,
            })

    if not any(page["text"] for page in document_pages):
        raise ValueError(
            "未提取到文字；文件可能需要 OCR 或其他解析方式。"
        )

    return document_pages


if __name__ == "__main__":
    file_path = input(
        "请粘贴一篇 PDF 的完整文件路径："
    ).strip().strip('"')

    pages = load_pdf_pages(file_path)

    text_pages = [
        page for page in pages
        if page["text"]
    ]

    print("文件：", pages[0]["source"])
    print("PDF 总页数：", len(pages))
    print("提取到文字的页数：", len(text_pages))

    first_text_page = text_pages[0]

    print("预览页码：", first_text_page["page"])
    print("正文预览：")
    print(first_text_page["text"][:1000])
