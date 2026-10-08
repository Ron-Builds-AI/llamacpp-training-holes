# Data

Training texts for the `llama-finetune` rows. Both are synthetic text from a fixed template and contain no third-party text.

| file | what | SHA256 |
|---|---|---|
| `tied_small.txt` | the first 45 lines of the companion repository's `REPRO_window_synthetic.txt`, 5,866 bytes. Its last line ends in a carriage return and a line feed, and the other lines in a line feed only, as in the file the runs used. | `40174901bf3aa2653f60906175580e725c7c3fdb2dadeffee0a00435a8d55796` |
| `short.txt` | the first 1,000 bytes of the same file | `8e4c2991985c1c0057a14d42338c29872832381b9aa40e52bb089c3b4766e894` |

`.gitattributes` keeps `*.txt` byte for byte, which matters for `tied_small.txt`. `short.txt` was checked against `head -c 1000` of the
companion's file and it matches.
