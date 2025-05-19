import csv
from pathlib import Path
from bs4 import BeautifulSoup


def extract_sections(html_content):
    """
    Extracts the problem statement, input specification, and output specification
    from a CodeNet problem HTML.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    sections = {
        "problem_id": None,
        "statement": "",
        "input_spec": "",
        "output_spec": "",
    }

    # Attempt to get the problem ID from title or filename
    title_tag = soup.find("title")
    if title_tag and title_tag.text.strip():
        # Assuming title like "Problem p00001"
        sections["problem_id"] = title_tag.text.strip().split()[-1]

    # Extract main statement (assuming in <div class="problem-statement"> or first <p>)
    stmt_div = soup.find("div", class_="problem-statement")
    if stmt_div:
        sections["statement"] = stmt_div.get_text(separator="\n").strip()
    else:
        # Fallback: all paragraphs until Input specification
        paras = soup.find_all("p")
        text = []
        for p in paras:
            if "Input" in p.text or "Output" in p.text:
                break
            text.append(p.get_text())
        sections["statement"] = "\n".join(text).strip()

    # Extract Input and Output specs by heading
    for header in soup.find_all(["h2", "h3", "strong"]):
        header_text = header.get_text().strip().lower()
        if "input specification" in header_text or header_text.startswith("input"):
            # gather following siblings until next header
            spec = []
            sib = header.find_next_sibling()
            while sib and sib.name not in ["h2", "h3", "strong"]:
                spec.append(sib.get_text(separator="\n").strip())
                sib = sib.find_next_sibling()
            sections["input_spec"] = "\n".join(spec).strip()
        if "output specification" in header_text or header_text.startswith("output"):
            spec = []
            sib = header.find_next_sibling()
            while sib and sib.name not in ["h2", "h3", "strong"]:
                spec.append(sib.get_text(separator="\n").strip())
                sib = sib.find_next_sibling()
            sections["output_spec"] = "\n".join(spec).strip()

    return sections


def main():
    html_folder = Path("../Project_CodeNet/problem_descriptions")
    output_csv = Path("problems_statements.csv")

    # Prepare CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["problem_id", "statement", "input_spec", "output_spec"]
        )
        writer.writeheader()

        # Iterate through HTML files
        for html_file in html_folder.glob("*.html"):
            with open(html_file, "r", encoding="utf-8") as f:
                content = f.read()
            sections = extract_sections(content)
            # If no ID from HTML title, fallback to filename
            if not sections["problem_id"]:
                sections["problem_id"] = html_file.stem
            writer.writerow(sections)
            print(f"Processed {sections['problem_id']}")


if __name__ == "__main__":
    main()
