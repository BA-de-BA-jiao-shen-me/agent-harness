from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field


@dataclass
class ParentChunk:
    parent_id: str
    doc_id: str
    source: str
    section_path: str
    content: str


@dataclass
class ChildChunk:
    chunk_id: str
    parent_id: str
    doc_id: str
    source: str
    section_path: str
    content: str
    is_table: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


class HierarchicalChunker:
    """
    Structure-aware Parent-Child Markdown/Document Chunker.

    Production highlights:
    1. Splits by Markdown heading hierarchy (`# / ## / ###`) preserving `section_path` context.
    2. Detects Markdown tables (`| col | ... |`) and keeps each table intact inside a chunk
       instead of slicing tables mid-row (a notorious naive RAG failure mode!).
    3. Emits small `ChildChunk`s for high-precision retrieval while linking to full `ParentChunk`s
       for rich context expansion at generation time.
    """

    def __init__(self, child_max_chars: int = 240, child_overlap: int = 40) -> None:
        self.child_max_chars = child_max_chars
        self.child_overlap = child_overlap

    @staticmethod
    def _split_into_sections(text: str) -> list[tuple[str, str]]:
        lines = text.splitlines()
        heading_stack: list[tuple[int, str]] = []
        sections: list[tuple[str, str]] = []
        buf: list[str] = []

        def flush() -> None:
            content = "\n".join(buf).strip()
            if content:
                sec_path = " > ".join(h for _, h in heading_stack) or "Root"
                sections.append((sec_path, content))
            buf.clear()

        for line in lines:
            m = re.match(r"^(#{1,4})\s+(.+)$", line.strip())
            if m:
                flush()
                level = len(m.group(1))
                title = m.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
            else:
                buf.append(line)
        flush()
        return sections

    @staticmethod
    def _split_blocks_preserving_tables(section_text: str) -> list[tuple[str, bool]]:
        """Group consecutive table lines together so tables are never broken mid-row."""
        blocks: list[tuple[str, bool]] = []
        cur_lines: list[str] = []
        in_table = False

        def flush() -> None:
            txt = "\n".join(cur_lines).strip()
            if txt:
                blocks.append((txt, in_table))
            cur_lines.clear()

        for line in section_text.splitlines():
            stripped = line.strip()
            line_is_table = stripped.startswith("|") and stripped.endswith("|")
            if line_is_table != in_table:
                flush()
                in_table = line_is_table
            if not stripped and not in_table:
                flush()
                continue
            cur_lines.append(line)
        flush()
        return blocks

    def chunk_document(
        self,
        text: str,
        source: str,
        doc_id: str | None = None,
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        did = doc_id or f"doc_{uuid.uuid4().hex[:8]}"
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        for sec_idx, (sec_path, sec_content) in enumerate(self._split_into_sections(text)):
            pid = f"{did}_p{sec_idx}"
            full_parent = f"[{sec_path}]\n{sec_content}"
            parents.append(
                ParentChunk(
                    parent_id=pid,
                    doc_id=did,
                    source=source,
                    section_path=sec_path,
                    content=full_parent,
                )
            )
            blocks = self._split_blocks_preserving_tables(sec_content)
            c_idx = 0
            for block_text, is_table in blocks:
                if is_table or len(block_text) <= self.child_max_chars:
                    children.append(
                        ChildChunk(
                            chunk_id=f"{pid}_c{c_idx}",
                            parent_id=pid,
                            doc_id=did,
                            source=source,
                            section_path=sec_path,
                            content=f"[{sec_path}] {block_text}",
                            is_table=is_table,
                        )
                    )
                    c_idx += 1
                else:
                    step = max(1, self.child_max_chars - self.child_overlap)
                    for start in range(0, len(block_text), step):
                        part = block_text[start : start + self.child_max_chars].strip()
                        if not part:
                            continue
                        children.append(
                            ChildChunk(
                                chunk_id=f"{pid}_c{c_idx}",
                                parent_id=pid,
                                doc_id=did,
                                source=source,
                                section_path=sec_path,
                                content=f"[{sec_path}] {part}",
                                is_table=False,
                            )
                        )
                        c_idx += 1
                        if start + self.child_max_chars >= len(block_text):
                            break

        return parents, children
