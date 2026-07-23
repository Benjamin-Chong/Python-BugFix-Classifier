import tokenize
import io

def tokenize_diff(changes):
    tokens = []
    lines = changes.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        holder = ''
        if line[0] == '-':
            holder = '<DELETE>'
            line = line[1:].strip()
        elif line[0] == '+':
            holder = '<ADD>'
            line = line[1:].strip()
        if holder:
            tokens.append(holder)
        try:
            line = line.strip()
            token_generator = tokenize.generate_tokens(io.StringIO(line).readline)
            for token in token_generator:
                if token.type == tokenize.ENDMARKER or token.type == tokenize.NEWLINE or token.type == tokenize.NL:
                    continue
                tokens.append(token.string)
        except tokenize.TokenError:
            pass

    return tokens