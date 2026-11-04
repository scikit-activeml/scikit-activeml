import re
import sys

from pathlib import Path


def link_code_snippets(text: str, ref: str | None = None) -> str:
    """
    Replace the collapsible code snippets with links named by their
    summaries. The links lead to the README on GitHub at the release tag
    `ref`, or at the development branch without a tag.

    PyPI cannot collapse the snippets, so they would push the installation
    far down the page.
    """
    url = (
        "https://github.com/scikit-activeml/scikit-activeml/blob/"
        f"{ref or 'development'}/README.rst"
    )

    def links(match):
        titles = re.findall(r"<b>(.*?)</b>", match[1])
        return "\n\n".join(f"`{title} <{url}>`__" for title in titles) + "\n"

    return re.sub(
        r"^\.\. code_snippets_start\n(.*?)^\.\. code_snippets_end\n",
        links,
        text,
        flags=re.DOTALL | re.MULTILINE,
    )


def transform_raw_blocks(text):
    """
    Process `.. raw::` blocks:

    - Remove HTML tags like <details>, <summary>, </details>, </summary>.
    - Convert <b>...</b> to **...**.
    - Keep non-HTML content, dedented.
    - End raw block when indentation breaks (so following directives
      stay valid).
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    in_raw = False

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # Start of a raw block
        if not in_raw and stripped.startswith(".. raw::"):
            in_raw = True
            i += 1
            continue

        if in_raw:
            stripped = line.lstrip()

            # End of raw block when we see a non-blank, non-indented line
            if stripped != "" and not (
                line.startswith(" ") or line.startswith("\t")
            ):
                in_raw = False
                # re-process this line outside the raw block
                continue

            # Blank line inside raw: keep as blank
            if stripped == "":
                out.append("")
                i += 1
                continue

            s = stripped

            # Convert <b>...</b> to **...**
            m = re.search(r"<b>(.*?)</b>", s)
            if m:
                out.append(f"**{m.group(1).strip()}**")
                out.append("")  # blank line after the heading
            elif s.startswith("<"):
                # Drop lines that are just HTML tags:
                # <details>, <summary>, etc.
                pass
            else:
                # Non-HTML text inside the raw block: keep, but dedented
                out.append(s)

            i += 1
            continue

        # Normal line outside raw block
        out.append(line)
        i += 1

    return "\n".join(out) + "\n"


def cleanup_html_fragments(text: str) -> str:
    """
    Clean up stray HTML fragments outside raw blocks if any remain.
    """
    # Kill <div style="clear: both;"></div> lines if they ever leak out
    text = re.sub(
        r"^\s*<div[^>]*>\s*</div>\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Drop standalone <details> / </details> if still present
    text = re.sub(
        r"^\s*</?details>\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Drop standalone <summary ...> / </summary>
    text = re.sub(
        r"^\s*</?summary[^>]*>\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Convert any remaining <b>...</b> to **...**
    text = re.sub(
        r"<b>(.*?)</b>",
        r"**\1**",
        text,
        flags=re.DOTALL,
    )

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def ensure_blank_before_code_blocks(text: str) -> str:
    """
    Ensure there is a blank line before any `.. code-block::` directive.
    If the previous line is non-blank, insert a blank line.
    """
    lines = text.splitlines()
    out: list[str] = []

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(".. code-block::"):
            if out and out[-1].strip() != "":
                out.append("")  # insert a blank line
        out.append(line)

    return "\n".join(out) + "\n"


def resolve_image_paths(text: str, ref: str | None = None) -> str:
    """
    Turn the README's relative image paths into raw GitHub URLs at the
    release tag `ref`, or at the development branch without a tag.

    PyPI keeps each release's README, but loads its images from their URLs.
    Tagged URLs show the images of that release.
    """
    target = f"refs/tags/{ref}" if ref else "refs/heads/development"
    return re.sub(
        r"((?:image|figure)::\s+)(docs/)",
        rf"\1https://raw.githubusercontent.com/scikit-activeml/"
        rf"scikit-activeml/{target}/\2",
        text,
    )


def main(src, dst, ref=None):
    src_path = Path(src)
    dst_path = Path(dst)
    text = src_path.read_text(encoding="utf8")
    text = link_code_snippets(text, ref)
    text = transform_raw_blocks(text)
    text = cleanup_html_fragments(text)
    text = ensure_blank_before_code_blocks(text)
    text = resolve_image_paths(text, ref)
    dst_path.write_text(text, encoding="utf8")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(
            "Usage: clean_readme_for_pypi.py SRC_RST DST_RST [RELEASE_TAG]"
        )
    main(*sys.argv[1:])
