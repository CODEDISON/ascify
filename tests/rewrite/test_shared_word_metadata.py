"""CPU-only malformed/alignment regression inputs for the real ELF gate."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('shared_word_check', Path(__file__).with_name('check_shared_word_alignment.py'))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def elf(section, payload, alignment=1, size=None):
    names = b'\0.shstrtab\0' + section.encode() + b'\0'
    body = bytes(64) + names + payload
    start = (len(body) + 7) & ~7
    header = bytearray(64)
    header[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<Q', header, 40, start)
    struct.pack_into('<HHH', header, 58, 64, 3, 1)
    body = bytes(header) + body[64:] + bytes(start - len(body))
    body += bytes(64)
    body += struct.pack('<IIQQQQIIQQ', 1, 3, 0, 0, 64, len(names), 0, 0, 1, 0)
    body += struct.pack('<IIQQQQIIQQ', 11, 1, 0, 0, 64+len(names), len(payload) if size is None else size, 0, 0, alignment, 0)
    return body


class MetadataChecks(unittest.TestCase):
    def check(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'object.o';path.write_bytes(data)
            return CHECK.ub_metadata(path,'sharedWordPairs',2176)

    def test_real_section_align4_and8(self):
        for alignment in (4, 8):
            data=elf('.ubuf.sharedWordPairs.simt_entry',bytes(2176),alignment)
            self.assertEqual(self.check(data)[0]['alignment'],alignment)

    def test_nested_device_elf(self):
        embedded=elf('.ubuf.sharedWordPairs.simt_entry',bytes(2176),4)
        self.assertEqual(len(self.check(elf('.aicore_binary',embedded))),1)

    def test_align2_cannot_promote(self):
        with self.assertRaises(AssertionError):
            self.check(elf('.ubuf.sharedWordPairs.simt_entry',bytes(2176),2))

    def test_extent_change_rejected(self):
        with self.assertRaises(AssertionError):
            self.check(elf('.ubuf.sharedWordPairs.simt_entry',bytes(2178),4))

    def test_missing_kernel_rejected(self):
        with self.assertRaises(AssertionError):
            self.check(elf('.ubuf.unrelated.simt_entry',bytes(2176),4))

    def test_host_text_does_not_prove_device_alignment(self):
        with self.assertRaises(AssertionError):
            self.check(elf('.text',b'ASCIFY_ALIGN(4)'))

    def test_invalid_format_rejected(self):
        with self.assertRaises(AssertionError):
            self.check(b'not an ELF object')


if __name__=='__main__':
    unittest.main()
