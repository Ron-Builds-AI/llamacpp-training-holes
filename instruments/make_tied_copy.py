"""Write a copy of a GGUF without output.weight, so llama.cpp ties the output layer to token_embd.weight.

Same copy logic as gguf-py/gguf/scripts/gguf_new_metadata.py (copy_with_new_metadata), with one tensor left out.

usage: PYTHONPATH=<your llama.cpp checkout>/gguf-py python make_tied_copy.py IN.gguf OUT.gguf

Edited for publication: the one line that put a local gguf-py path on sys.path was removed, nothing else. The edited
script was not re-run; the model in the proof runs was written by the unedited script.
"""
import sys

import gguf


def main():
    src, dst = sys.argv[1], sys.argv[2]
    reader = gguf.GGUFReader(src, "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    tensors = [t for t in reader.tensors if t.name != "output.weight"]
    print(f"{src}: {len(reader.tensors)} tensors, writing {len(tensors)} (output.weight left out)")
    for t in tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
