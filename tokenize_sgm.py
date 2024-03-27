import os
from bs4 import BeautifulSoup
from fairseq import options, tasks

from fairseq.fairseq.dataclass.utils import convert_namespace_to_omegaconf

def tokenize_sgm_file(tokenizer, bpe, sgm_fpath):
    tokenized_lines = []
    with open(sgm_fpath, encoding="utf-8") as f:
        contents = f.read()
    soup = BeautifulSoup(contents, "lxml-xml")
    for seg in soup.find_all('seg'):
        # Tokenize the text
        tokenized_text = tokenizer.encode(seg.text)
        tokenized_text = bpe.encode(tokenized_text)
        tokenized_lines.append(tokenized_text)

    return tokenized_lines

if __name__ == "__main__":
    # We use the same parser as interactive.py to build tokenizer and bpe
    parser = options.get_interactive_generation_parser()
    parser.add_argument('--sgm_fpath', type=str, help='Path to the sgm file')
    parser.add_argument('--output_fpath', type=str, help='Path to the output file')
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)

    task = tasks.setup_task(cfg.task)
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)
    # Tokenize the source and target files
    tokenized_src_lines = tokenize_sgm_file(tokenizer, bpe, args.sgm_fpath)

    # Write the tokenized lines to the output file
    os.makedirs(os.path.dirname(args.output_fpath), exist_ok=True)
    with open(args.output_fpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tokenized_src_lines))
