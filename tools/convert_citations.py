"""Convert plain-text citations to pandoc-citeproc [@key] format.

Uses a hard-coded mapping of every citation string in the manuscript to its bib key.
Rewrites paper/draft/*.md files in place. Logs every replacement to stderr.

Usage:  python tools/convert_citations.py [--dry-run]
"""
import re, sys, glob

DRY_RUN = "--dry-run" in sys.argv

# Complete mapping: normalized "FirstAuthor, Year[suffix]" -> bib key
# Built by cross-referencing every citation in the manuscript against references.bib
CITE_MAP = {
    "Ackerman & Panickssery, 2025": "ackerman2025inspection",
    "Arditi et al., 2024": "arditi2024refusal",
    "Bansal et al., 2021": "bansal2021stitching",
    "Bhattacharjee & Liu, 2024": "bhattacharjee2024fighting",
    "Bigoulaeva et al., 2026": "bigoulaeva2026instruction",
    "Binder et al., 2024": "binder2024looking",
    "Block, 2019": "block2019gnad",
    "Brunswik, 1956": "brunswik1956perception",
    "Burns et al., 2024": "burns2023discovering",
    "Clark et al., 2021": "clark2021human",
    "Desimone & Duncan, 1995": "desimone1995biased",
    "Elazar et al., 2021": "elazar2021amnesic",
    "Funder, 1995": "funder1995accuracy",
    "Gupta et al., 2026": "gupta2026cuerouting",
    "Gurnee et al., 2026": "gurnee2026verbalizable",
    "Hautus, 1995": "hautus1995corrections",
    "Heidari et al., 2026": "heidari2026evalawareness",
    "Heimersheim & Nanda, 2024": "heimersheim2024patching",
    "Hewitt & Liang, 2019": "hewitt2019control",
    "Hu & Levy, 2023": "hu2023prompting",
    "Hu et al., 2022": "hu2022lora",
    "Jakesch et al., 2023": "jakesch2023human",
    "Jones & Bergen, 2025": "jones2025turing",
    "Kadavath et al., 2022": "kadavath2022know",
    "Kahneman & Tversky, 1973": "kahneman1973prediction",
    "Karelaia & Hogarth, 2008": "karelaia2008lens",
    "Kissane et al., 2024": "kissane2024saetransfer",
    "Kornblith et al., 2019": "kornblith2019similarity",
    "Lambert et al., 2024": "lambert2024tulu3",
    "Lee et al., 2024": "lee2024dpo",
    "Lenc & Vedaldi, 2015": "lenc2015equivariance",
    "Leong et al., 2025": "leong2025template",
    "Lin et al., 2023": "lin2023urial",
    "Lindsey et al., 2024": "lindsey2024crosscoders",
    "Lindsey, 2025": "lindsey2025introspection",
    "Liu et al., 2023": "liu2023cognitive",
    "Loos & Eberle, 2026": "loos2026kvpatching",
    "Macmillan & Creelman, 2004": "macmillan2005detection",
    "Makelov et al., 2023": "makelov2023subspace",
    "Marks & Tegmark, 2024": "marks2024geometry",
    "Marks and Tegmark, 2024": "marks2024geometry",
    "Meng et al., 2023": "meng2022rome",
    "Merchant et al., 2020": "merchant2020bert",
    "Minder et al., 2026": "minder2025crosscoders",
    "Orgad et al., 2025": "orgad2025know",
    "Ott et al., 2011": "ott2011finding",
    "Ott et al., 2013": "ott2013negative",
    "Panickssery et al., 2024": "panickssery2024evaluators",
    "Panzeri et al., 2017": "panzeri2017cracking",
    "Phillips & Edwards, 1966": "phillips1966conservatism",
    "Pimentel & Meister, 2024": "pimentel2024compute",
    "Porter & Machery, 2024": "porter2024poetry",
    "Prakash et al., 2024": "prakash2024finetuning",
    "Raghavendra et al., 2024": "raghavendra2024revisiting",
    "Rathi et al., 2024": "rathi2024gpt4",
    "Rosin & Laubrock, in preparation": "rosin_inprep",
    "Russell et al., 2025": "russell2025frequent",
    "Schabus et al., 2017": "schabus2017onemillion",
    "Templeton et al., 2024": "templeton2024scaling",
    "Thai et al., 2025": "thai2026editlens",
    "Torop et al., 2026": "torop2026inverted",
    "Tucker, 1964": "tucker1964lme",
    "Tulving & Pearlstone, 1966": "tulving1966availability",
    "Vergara-Browne et al., 2026": "vergarabrowne2026sah",
    "Vig et al., 2020": "vig2020causal",
    "Villuri et al., 2026": "villuri2026probes",
    "Waldis et al., 2026": "waldis2026instructions",
    "Wang et al., 2022": "wang2022ioi",
    "Weidmann, 2025": "weidmann2025heretic",
    "Wortsman et al., 2022a": "wortsman2022wiseft",
    "Wortsman et al., 2022b": "wortsman2022soups",
    "Yuan et al., 2026": "yuan2026crossgen",
    "Zhao et al., 2026": "zhao2026layerwise",
    "Zhou et al., 2023": "zhou2023lima",
    "Zhou et al., 2026": "zhou2025cosur",
    "Zhou, 2026a": "zhou2026convergence",
    "Zhou, 2026b": "zhou2026fearcaut",
    # Köbis with special char
    "Köbis & Mossink, 2021": "kobis2021angelou",
}

def normalize_cite(s):
    """Normalize whitespace in a citation string."""
    return re.sub(r'\s+', ' ', s).strip()

def lookup(cite_str):
    """Look up a citation string in the mapping."""
    n = normalize_cite(cite_str)
    if n in CITE_MAP:
        return CITE_MAP[n]
    # Try without line breaks
    for k, v in CITE_MAP.items():
        if normalize_cite(k) == n:
            return v
    return None

def process_file(path):
    text = open(path).read()
    changes = []

    # A. Narrative citations: "Author et al. (Year)" or "Author and Author (Year)"
    def replace_narrative(m):
        full = m.group(0)
        author_part = m.group(1).strip()
        year = m.group(2)
        # Build lookup string
        cite_str = f"{author_part}, {year}"
        key = lookup(cite_str)
        if key:
            repl = f"@{key}"
            changes.append((full, repl))
            return repl
        print(f"  NARRATIVE NOT FOUND: {cite_str}", file=sys.stderr)
        return full

    text = re.sub(
        r'([A-ZÀ-Ž][a-zà-žöüé\'-]+(?:\s+(?:et\s+al\.|and\s+[A-ZÀ-Ž][a-zà-žöüé\'-]+))?)(?:\s+)\((\d{4}[a-b]?)\)',
        replace_narrative,
        text
    )

    # B. Parenthetical citations
    def replace_paren(m):
        full = m.group(0)
        inner = m.group(1)

        # Must contain author-like pattern + year
        if not re.search(r'[A-ZÀ-Ž][a-zà-ž].*\d{4}', inner):
            return full

        # Split by semicolons
        parts = re.split(r';\s*', inner)
        converted = []
        has_failure = False

        for part in parts:
            part = part.strip()
            part_prefix = ""

            # Check for prefix
            pfx_match = re.match(r'^(see also|cf\.|e\.g\.,?|for a review,? see)\s+', part, re.I)
            if pfx_match:
                part_prefix = pfx_match.group(1) + " "
                part = part[pfx_match.end():]

            # Handle "Author et al., Year1, Year2" (same author, multiple years with suffixes)
            m_multi = re.match(
                r'^([A-ZÀ-Ž][a-zà-žöüé\'-]+(?:\s+et\s+al\.)?),\s*(\d{4}[a-b]?)(?:,\s*(\d{4}[a-b]?))+$',
                part
            )
            if m_multi:
                author = re.match(r'^([A-ZÀ-Ž][a-zà-žöüé\'-]+(?:\s+et\s+al\.)?)', part).group(1)
                years = re.findall(r'(\d{4}[a-b]?)', part)
                for yr in years:
                    cite_str = f"{author}, {yr}"
                    key = lookup(cite_str)
                    if key:
                        entry = f"@{key}"
                        if part_prefix and years.index(yr) == 0:
                            entry = part_prefix + entry
                        converted.append(entry)
                    else:
                        print(f"  MULTI-YEAR NOT FOUND: {cite_str}", file=sys.stderr)
                        has_failure = True
                continue

            # Single citation
            # Normalize: collapse whitespace
            part_norm = normalize_cite(part)
            key = lookup(part_norm)
            if key:
                entry = f"@{key}"
                if part_prefix:
                    entry = part_prefix + entry
                converted.append(entry)
            else:
                # Not a pure citation — mixed text like "Tülu 3" or "CKA"
                # or "in preparation" variant
                # Try matching just the citation part at end
                tail_m = re.search(
                    r'([A-ZÀ-Ž][a-zà-žöüé\'-]+(?:\s+(?:et\s+al\.|&\s+[A-ZÀ-Ž][a-zà-žöüé\'-]+))?,\s*(?:\d{4}[a-b]?|in preparation))\s*$',
                    part_norm
                )
                if tail_m:
                    cite_part = tail_m.group(1)
                    key = lookup(cite_part)
                    if key:
                        # Keep the non-citation prefix text
                        prefix_text = part_norm[:tail_m.start()].strip()
                        if prefix_text:
                            # Can't cleanly convert mixed text+citation in pandoc syntax
                            # Leave as-is
                            has_failure = True
                        else:
                            entry = f"@{key}"
                            if part_prefix:
                                entry = part_prefix + entry
                            converted.append(entry)
                    else:
                        has_failure = True
                else:
                    has_failure = True

        if has_failure or not converted:
            return full

        result = "[" + "; ".join(converted) + "]"
        changes.append((full, result))
        return result

    text = re.sub(
        r'\(([^()]*?(?:\d{4}|in preparation)[a-b]?[^()]*)\)',
        replace_paren,
        text
    )

    if changes:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"FILE: {path}", file=sys.stderr)
        for old, new in changes:
            old_short = normalize_cite(old)[:80]
            print(f"  {old_short}", file=sys.stderr)
            print(f"  → {new}", file=sys.stderr)

        if not DRY_RUN:
            open(path, 'w').write(text)
            print(f"  [{len(changes)} replacements written]", file=sys.stderr)
        else:
            print(f"  [{len(changes)} replacements (dry run)]", file=sys.stderr)
    return len(changes)


total = 0
for f in sorted(glob.glob("paper/draft/*.md")):
    total += process_file(f)

print(f"\nTotal: {total} replacements {'(dry run)' if DRY_RUN else 'applied'}", file=sys.stderr)
