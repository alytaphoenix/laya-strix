#!/usr/bin/env python3
"""Clear the executable-stack bit on Ryzen AI ELF libraries.

Several Ryzen AI 1.8 shared objects ship with PT_GNU_STACK's PF_X bit set.
Ubuntu 26.04 refuses to load them ("cannot enable executable stack as shared
object requires: Invalid argument").  This patches the program header in place
and keeps a ``.execstack.bak`` copy.  It is idempotent and must be re-run after
every Ryzen AI reinstall.

    python3 deploy/fix_execstack.py [venv_dir]
"""

from __future__ import annotations

import shutil
import struct
import sys
from pathlib import Path

PT_GNU_STACK = 0x6474E551
PF_X = 0x1


def _gnu_stack_offset(data: bytes):
    if data[:4] != b"\x7fELF" or data[4] != 2:  # 64-bit ELF only
        return None
    little = data[5] == 1
    end = "<" if little else ">"
    e_phoff, = struct.unpack_from(end + "Q", data, 0x20)
    e_phentsize, e_phnum = struct.unpack_from(end + "HH", data, 0x36)
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, = struct.unpack_from(end + "I", data, off)
        if p_type == PT_GNU_STACK:
            return off + 4, end
    return None


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    root = Path(argv[0]) if argv else Path(__file__).resolve().parent.parent / "venv"
    if not root.is_dir():
        raise SystemExit("not a directory: %s" % root)

    patched = clean = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.name.endswith(".execstack.bak"):
            continue
        try:
            with path.open("rb") as f:
                head = f.read(64)
                if head[:4] != b"\x7fELF":
                    continue
                data = bytearray(head + f.read())
        except OSError:
            continue

        found = _gnu_stack_offset(data)
        if found is None:
            continue
        flags_off, end = found
        flags, = struct.unpack_from(end + "I", data, flags_off)
        if not flags & PF_X:
            clean += 1
            continue

        backup = path.with_name(path.name + ".execstack.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        struct.pack_into(end + "I", data, flags_off, flags & ~PF_X)
        path.write_bytes(data)
        print("patched %s" % path)
        patched += 1

    print("\n%d file(s) patched, %d ELF(s) already non-exec-stack." % (patched, clean))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
