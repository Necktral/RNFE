import os
import datetime

CHANGELOG_PATH = os.path.join(os.path.dirname(__file__), "../CHANGELOG.md")

ENTRY_TEMPLATE = """
## [{tag}] {date}
{changes}
"""

def add_changelog_entry(tag: str, changes: str):
    date = datetime.date.today().strftime("%d-%m-%Y")
    entry = ENTRY_TEMPLATE.format(tag=tag, date=date, changes=changes)
    with open(CHANGELOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, required=True, help="Etiqueta/version del cambio")
    ap.add_argument("--changes", type=str, required=True, help="Descripción de los cambios")
    args = ap.parse_args()
    add_changelog_entry(args.tag, args.changes)
    print(f"Registro agregado a CHANGELOG.md: {args.tag}")
